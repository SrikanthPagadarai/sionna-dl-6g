# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import tensorflow as tf
import pytest
from sionna.phy.utils import ebnodb2no
from demos.mimo_ofdm_neural_receiver.src.config import Config, BitsPerSym
from demos.mimo_ofdm_neural_receiver.src.csi import CSI
from demos.mimo_ofdm_neural_receiver.src.rx import Rx

# Dimension reference for RX outputs:
#
# h_hat (channel estimate):
#   [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
#           num_ofdm_symbols, num_effective_subcarriers]
# err_var (estimation error):
#   scalar
#   or
#   [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
#           num_ofdm_symbols, num_effective_subcarriers]
# x_hat (equalized symbols):
#   [batch, num_tx, num_streams_per_tx, num_data_symbols]
# no_eff (effective noise):
#   [batch, num_tx, num_streams_per_tx, num_data_symbols]
# llr (log-likelihood ratios):
#   [batch, num_tx, num_streams_per_tx, n]
# b_hat (decoded bits):
#   [batch, num_tx, num_streams_per_tx, k]


def rand_cplx(shape, dtype=tf.float32):
    return tf.complex(
        tf.random.normal(shape, dtype=dtype), tf.random.normal(shape, dtype=dtype)
    )


@pytest.mark.parametrize("modulation", [BitsPerSym.QPSK, BitsPerSym.QAM16])
@pytest.mark.parametrize("perfect_csi", [True, False])
def test_rx(modulation, perfect_csi):
    """Test RX for different modulation schemes and CSI modes."""
    cfg = Config(num_bits_per_symbol=modulation, perfect_csi=perfect_csi)
    batch_size = tf.constant(4, tf.int32)
    ebno_db = tf.constant(10.0, tf.float32)

    csi = CSI(cfg)
    h_freq = csi.build(batch_size)
    rx = Rx(cfg, csi)

    # Dummy received signal
    y = rand_cplx((batch_size, 1, cfg.num_bs_ant, cfg.num_ofdm_symbols, cfg.fft_size))
    no = ebnodb2no(ebno_db, cfg.num_bits_per_symbol, cfg.coderate, cfg.rg)

    out = rx(y, h_freq, no)

    # Print shapes
    csi_mode = "perfect" if perfect_csi else "estimated"
    print(f"\n[RX] {modulation.name}, CSI={csi_mode}:")
    for k, v in out.items():
        print(f"  {k:10s}: {v.shape}")

    # Assertions
    assert out["b_hat"].shape == (4, 1, cfg.num_streams_per_tx, cfg.k)
    assert out["llr"].shape == (4, 1, cfg.num_streams_per_tx, cfg.n)
    assert out["x_hat"].shape == (4, 1, cfg.num_streams_per_tx, cfg.rg.num_data_symbols)
