# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import tensorflow as tf
import pytest
from sionna.phy.utils import ebnodb2no
from demos.mimo_ofdm_neural_receiver.src.config import Config, BitsPerSym
from demos.mimo_ofdm_neural_receiver.src.csi import CSI
from demos.mimo_ofdm_neural_receiver.src.channel import Channel

# Dimension reference for Channel:
#
# Input:
#   x_rg_tx (transmitted resource grid):
#       [batch, num_tx, num_ofdm_symbols, fft_size]
#   h_freq (channel frequency response):
#       [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
#   no (noise power):
#       scalar or [batch]
#
# Output:
#   y (received signal): [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]


def rand_cplx(shape, dtype=tf.float32):
    return tf.complex(
        tf.random.normal(shape, dtype=dtype), tf.random.normal(shape, dtype=dtype)
    )


@pytest.mark.parametrize("modulation", [BitsPerSym.QPSK, BitsPerSym.QAM16])
def test_channel(modulation):
    """Test Channel forward pass for different modulation schemes."""
    cfg = Config(num_bits_per_symbol=modulation)
    batch_size = tf.constant(4, dtype=tf.int32)
    ebno_db = tf.constant(10.0, tf.float32)

    csi = CSI(cfg)
    h_freq = csi.build(batch_size)

    channel = Channel()

    # Dummy transmitted resource grid
    x_shape = (4, cfg.rg.num_tx, cfg.num_ofdm_symbols, cfg.fft_size)
    x_rg_tx = rand_cplx(x_shape)
    no = ebnodb2no(ebno_db, cfg.num_bits_per_symbol, cfg.coderate, cfg.rg)

    out = channel(x_rg_tx, h_freq, no)

    # Print shapes
    print(f"\n[Channel] {modulation.name}:")
    print(f"  x_rg_tx shape: {x_rg_tx.shape}")
    print(f"  h_freq shape:  {h_freq.shape}")
    print(f"  y shape:       {out['y'].shape}")

    # Assertions
    assert "y" in out
    assert (
        len(out["y"].shape) == 5
    )  # [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
    assert out["y"].shape[0] == 4  # batch size
