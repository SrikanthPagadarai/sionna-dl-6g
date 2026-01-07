# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""Tests for DPD Tx OFDM transmitter."""

import pytest
import tensorflow as tf
from demos.dpd.src.config import Config
from demos.dpd.src.tx import Tx

# Dimension reference for TX outputs:
#
# bits:
#   [batch, num_tx, num_streams_per_tx, k]
# codewords:
#   [batch, num_tx, num_streams_per_tx, n]
# x_rg:
#   [batch, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
# x_time:
#   [batch, num_tx, num_streams_per_tx, (fft_size + cp_length) * num_ofdm_symbols]


def test_tx_initialization():
    """Test Tx initialization with Config."""
    cfg = Config()
    tx = Tx(cfg)

    # Verify resource grid is created
    assert tx.rg is not None
    assert tx.rg.num_tx == cfg.num_ut
    assert tx.rg.num_streams_per_tx == cfg.num_streams_per_tx

    # Verify LDPC dimensions
    assert tx.k > 0  # info bits
    assert tx.n > tx.k  # coded bits > info bits

    print("\n[Tx Initialization]:")
    print(f"  k (info bits): {tx.k}")
    print(f"  n (coded bits): {tx.n}")
    print(f"  Code rate: {tx.k / tx.n:.4f}")


def test_tx_forward_pass():
    """Test Tx forward pass output shapes."""
    cfg = Config()
    tx = Tx(cfg)
    batch_size = tf.constant(4, dtype=tf.int32)

    out = tx(batch_size)

    # Verify all expected keys are present
    assert "bits" in out
    assert "codewords" in out
    assert "x_rg" in out
    assert "x_time" in out

    # Verify output shapes
    assert out["bits"].shape[0] == 4  # batch
    assert out["bits"].shape[1] == cfg.num_ut
    assert out["bits"].shape[2] == cfg.num_streams_per_tx
    assert out["bits"].shape[3] == tx.k

    assert out["codewords"].shape[0] == 4
    assert out["codewords"].shape[3] == tx.n

    assert out["x_rg"].shape[0] == 4
    assert out["x_rg"].shape[3] == cfg.num_ofdm_symbols
    assert out["x_rg"].shape[4] == cfg.fft_size

    # Time-domain samples per frame
    samples_per_frame = (cfg.fft_size + cfg.cyclic_prefix_length) * cfg.num_ofdm_symbols
    assert out["x_time"].shape[3] == samples_per_frame

    print("\n[Tx Forward Pass]:")
    print(f"  bits shape: {out['bits'].shape}")
    print(f"  codewords shape: {out['codewords'].shape}")
    print(f"  x_rg shape: {out['x_rg'].shape}")
    print(f"  x_time shape: {out['x_time'].shape}")


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_tx_batch_sizes(batch_size):
    """Test Tx with different batch sizes."""
    cfg = Config()
    tx = Tx(cfg)

    out = tx(tf.constant(batch_size, dtype=tf.int32))

    assert out["bits"].shape[0] == batch_size
    assert out["x_time"].shape[0] == batch_size

    print(f"\n[Tx Batch Size {batch_size}]:")
    print(f"  x_time shape: {out['x_time'].shape}")


def test_tx_resource_grid_properties():
    """Test Tx resource grid configuration."""
    cfg = Config()
    tx = Tx(cfg)

    rg = tx.rg

    # Verify resource grid properties match config
    assert rg.num_ofdm_symbols == cfg.num_ofdm_symbols
    assert rg.fft_size == cfg.fft_size
    assert rg.cyclic_prefix_length == cfg.cyclic_prefix_length

    print("\n[Tx Resource Grid Properties]:")
    print(f"  num_ofdm_symbols: {rg.num_ofdm_symbols}")
    print(f"  fft_size: {rg.fft_size}")
    print(f"  cyclic_prefix_length: {rg.cyclic_prefix_length}")
    print(f"  num_data_symbols: {rg.num_data_symbols}")


def test_tx_output_dtype():
    """Test Tx output data types."""
    cfg = Config()
    tx = Tx(cfg)
    batch_size = tf.constant(2, dtype=tf.int32)

    out = tx(batch_size)

    # bits should be integer-like (but stored as float for LDPC)
    # x_rg and x_time should be complex
    assert out["x_rg"].dtype == tf.complex64
    assert out["x_time"].dtype == tf.complex64

    print("\n[Tx Output Dtypes]:")
    print(f"  bits dtype: {out['bits'].dtype}")
    print(f"  codewords dtype: {out['codewords'].dtype}")
    print(f"  x_rg dtype: {out['x_rg'].dtype}")
    print(f"  x_time dtype: {out['x_time'].dtype}")
