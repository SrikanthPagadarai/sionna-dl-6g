# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import pytest
import tensorflow as tf
from demos.mimo_ofdm_neural_receiver.src.config import Config, BitsPerSym
from demos.mimo_ofdm_neural_receiver.src.tx import Tx

# Dimension reference for TX outputs:
#
# b (info bits):     [batch, num_tx, num_streams_per_tx, k]
# c (coded bits):    [batch, num_tx, num_streams_per_tx, n]
# x (symbols):       [batch, num_tx, num_streams_per_tx, num_data_symbols]
# x_rg (resource grid): [batch, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
#
# Where:
#   batch = number of parallel channel realizations
#   num_tx = number of transmitters (1 for single-user)
#   num_streams_per_tx = spatial streams per TX (= num_ut_ant)
#   k = information bits per codeword (n * coderate)
#   n = codeword length (num_data_symbols * num_bits_per_symbol)
#   num_data_symbols = data-carrying REs in the resource grid
#   num_ofdm_symbols = OFDM symbols per slot (14 for normal CP)
#   fft_size = subcarriers (76)


@pytest.mark.parametrize("modulation", [BitsPerSym.QPSK, BitsPerSym.QAM16])
def test_tx(modulation):
    """Test TX for different modulation schemes."""
    cfg = Config(num_bits_per_symbol=modulation)
    batch_size = tf.constant(4, dtype=tf.int32)

    tx = Tx(cfg)
    out = tx(batch_size)

    # Expected dimensions from config
    expected = {
        "b": (4, 1, cfg.num_streams_per_tx, cfg.k),
        "c": (4, 1, cfg.num_streams_per_tx, cfg.n),
        "x": (4, 1, cfg.num_streams_per_tx, cfg.rg.num_data_symbols),
        "x_rg": (4, 1, cfg.num_streams_per_tx, cfg.num_ofdm_symbols, cfg.fft_size),
    }

    # Print shapes with meanings
    print(f"\n[TX] {modulation.name} (bits_per_symbol={modulation.value}):")
    print(f"  b (info bits):      {out['b'].shape} -> k={cfg.k}")
    print(f"  c (coded bits):     {out['c'].shape} -> n={cfg.n}")
    print(f"  x (symbols):        {out['x'].shape}")
    print(f"  x_rg (resource grid): {out['x_rg'].shape}")

    # Assertions
    assert out["b"].shape == expected["b"]
    assert out["c"].shape == expected["c"]
    assert out["x"].shape == expected["x"]
    assert out["x_rg"].shape == expected["x_rg"]
