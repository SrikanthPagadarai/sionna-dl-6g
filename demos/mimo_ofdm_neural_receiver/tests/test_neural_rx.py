# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import tensorflow as tf
import pytest
from sionna.phy.utils import ebnodb2no
from demos.mimo_ofdm_neural_receiver.src.config import Config, BitsPerSym
from demos.mimo_ofdm_neural_receiver.src.csi import CSI
from demos.mimo_ofdm_neural_receiver.src.neural_rx import NeuralRx

# Dimension reference for NeuralRx:
#
# Input:
#   y (received signal): [batch, 1, num_streams, num_ofdm_symbols, fft_size]
#   no (noise power): scalar or [batch]
#   batch_size: scalar tensor
#
# Output:
#   llr (log-likelihood ratios): [batch, 1, num_ut_ant, n]
#   b_hat (decoded bits): [batch, 1, num_ut_ant, k] or None if channel_coding_off


def rand_cplx(shape, dtype=tf.float32):
    return tf.complex(
        tf.random.normal(shape, dtype=dtype), tf.random.normal(shape, dtype=dtype)
    )


@pytest.mark.parametrize("modulation", [BitsPerSym.QPSK, BitsPerSym.QAM16])
@pytest.mark.parametrize("perfect_csi", [True, False])
def test_neural_rx(modulation, perfect_csi):
    """Test NeuralRx forward pass for different configurations."""
    cfg = Config(num_bits_per_symbol=modulation, perfect_csi=perfect_csi)
    batch_size = tf.constant(4, dtype=tf.int32)
    ebno_db = tf.constant(15.0, tf.float32)

    csi = CSI(cfg)
    csi.build(batch_size)

    # Dims from config
    n_sym = cfg.num_ofdm_symbols
    n_sc = cfg.fft_size
    num_streams = cfg.num_streams_per_tx

    # Dummy received signal
    y = rand_cplx((4, 1, num_streams, n_sym, n_sc))
    no = ebnodb2no(ebno_db, cfg.num_bits_per_symbol, cfg.coderate, cfg.rg)

    # Create and run NeuralRx
    nrx = NeuralRx(cfg)
    out = nrx(y, no, batch_size)

    # Print shapes
    csi_mode = "perfect" if perfect_csi else "estimated"
    print(f"\n[NeuralRx] {modulation.name}, CSI={csi_mode}:")
    print(f"  y shape:     {y.shape}")
    print(f"  llr shape:   {out['llr'].shape}")
    print(f"  b_hat shape: {out['b_hat'].shape}")

    # Assertions
    assert out["llr"].shape == (4, 1, cfg.num_ut_ant, cfg.n)
    assert out["b_hat"].shape == (4, 1, cfg.num_ut_ant, cfg.k)
