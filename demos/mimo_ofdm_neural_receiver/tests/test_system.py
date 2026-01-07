# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import tensorflow as tf
import pytest
from sionna.phy.utils import compute_ber
from demos.mimo_ofdm_neural_receiver.src.config import BitsPerSym
from demos.mimo_ofdm_neural_receiver.src.system import System

# Dimension reference for System:
#
# Input:
#   batch_size: scalar tensor
#   ebno_db: scalar (call_scalar) or [batch] tensor (__call__)
#
# Output (inference mode):
#   b (transmitted bits): [batch, 1, num_streams_per_tx, k]
#   b_hat (decoded bits): [batch, 1, num_streams_per_tx, k]
#
# Output (training mode):
#   loss: scalar


@pytest.mark.parametrize("modulation", [BitsPerSym.QPSK, BitsPerSym.QAM16])
@pytest.mark.parametrize("perfect_csi", [True, False])
@pytest.mark.parametrize("use_neural_rx", [True, False])
def test_system_inference(modulation, perfect_csi, use_neural_rx):
    """Test System forward pass in inference mode."""
    system = System(
        training=False,
        perfect_csi=perfect_csi,
        cdl_model="D",
        delay_spread=300e-9,
        carrier_frequency=2.6e9,
        speed=0.0,
        num_bits_per_symbol=modulation,
        use_neural_rx=use_neural_rx,
        name="system",
    )

    batch_size = tf.constant(4, tf.int32)
    ebno_db = tf.constant(40.0, tf.float32)

    b, b_hat = system.call_scalar(batch_size, ebno_db)
    ber = compute_ber(b, b_hat)

    # Print results
    csi_mode = "perfect" if perfect_csi else "estimated"
    rx_mode = "neural" if use_neural_rx else "baseline"
    print(f"\n[System] {modulation.name}, CSI={csi_mode}, RX={rx_mode}:")
    print(f"  b shape:     {b.shape}")
    print(f"  b_hat shape: {b_hat.shape}")
    print(f"  BER:         {float(ber):.6f}")

    # Assertions
    assert b.shape == b_hat.shape
    assert len(b.shape) == 4  # [batch, num_tx, num_streams, k]


@pytest.mark.parametrize("modulation", [BitsPerSym.QPSK, BitsPerSym.QAM16])
def test_system_training(modulation):
    """Test System forward pass in training mode (returns loss)."""
    system = System(
        training=True,
        perfect_csi=True,
        cdl_model="D",
        num_bits_per_symbol=modulation,
        use_neural_rx=True,
        name="system_training",
    )

    batch_size = tf.constant(4, tf.int32)
    ebno_db = tf.constant(15.0, tf.float32)

    loss = system.call_scalar(batch_size, ebno_db)

    print(f"\n[System Training] {modulation.name}:")
    print(f"  loss: {float(loss):.6f}")

    # Assertions
    assert loss.shape == ()  # scalar
    assert not tf.math.is_nan(loss)
