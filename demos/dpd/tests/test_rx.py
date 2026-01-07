# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""Tests for DPD Rx OFDM receiver."""

import tensorflow as tf
import numpy as np
from demos.dpd.src.config import Config
from demos.dpd.src.rx import Rx


def test_rx_initialization():
    """Test Rx initialization with typical DPD parameters."""
    cfg = Config()

    rx = Rx(
        signal_fs=cfg.signal_sample_rate,
        pa_sample_rate=122.88e6,
        fft_size=cfg.fft_size,
        cp_length=cfg.cyclic_prefix_length,
        num_ofdm_symbols=cfg.num_ofdm_symbols,
        num_guard_lower=cfg.num_guard_carriers[0],
        num_guard_upper=cfg.num_guard_carriers[1],
        dc_null=cfg.dc_null,
    )

    # Verify initialization
    assert rx._signal_fs == cfg.signal_sample_rate
    assert rx._fft_size == cfg.fft_size
    assert rx._cp_length == cfg.cyclic_prefix_length

    print("\n[Rx Initialization]:")
    print(f"  signal_fs: {rx._signal_fs / 1e6:.2f} MHz")
    print(f"  pa_sample_rate: {rx._pa_sample_rate / 1e6:.2f} MHz")
    print(f"  fft_size: {rx._fft_size}")
    print(f"  cp_length: {rx._cp_length}")


def test_rx_subcarrier_indices():
    """Test Rx subcarrier index calculation."""
    cfg = Config()

    rx = Rx(
        signal_fs=cfg.signal_sample_rate,
        pa_sample_rate=122.88e6,
        fft_size=cfg.fft_size,
        cp_length=cfg.cyclic_prefix_length,
        num_ofdm_symbols=cfg.num_ofdm_symbols,
        num_guard_lower=cfg.num_guard_carriers[0],
        num_guard_upper=cfg.num_guard_carriers[1],
        dc_null=cfg.dc_null,
    )

    # Verify subcarrier indices are set correctly
    assert rx._lower_start == cfg.num_guard_carriers[0]
    assert rx._lower_end == cfg.fft_size // 2
    assert rx._upper_start == cfg.fft_size // 2 + (1 if cfg.dc_null else 0)
    assert rx._upper_end == cfg.fft_size - cfg.num_guard_carriers[1]

    # Calculate number of data subcarriers
    num_lower = rx._lower_end - rx._lower_start
    num_upper = rx._upper_end - rx._upper_start
    total_data_subcarriers = num_lower + num_upper

    print("\n[Rx Subcarrier Indices]:")
    print(f"  lower: [{rx._lower_start}, {rx._lower_end})")
    print(f"  upper: [{rx._upper_start}, {rx._upper_end})")
    print(f"  total data subcarriers: {total_data_subcarriers}")


def test_rx_demod():
    """Test Rx OFDM demodulation."""
    cfg = Config()

    rx = Rx(
        signal_fs=cfg.signal_sample_rate,
        pa_sample_rate=122.88e6,
        fft_size=cfg.fft_size,
        cp_length=cfg.cyclic_prefix_length,
        num_ofdm_symbols=cfg.num_ofdm_symbols,
        num_guard_lower=cfg.num_guard_carriers[0],
        num_guard_upper=cfg.num_guard_carriers[1],
        dc_null=cfg.dc_null,
    )

    # Create a test signal with correct length
    num_samples = (cfg.fft_size + cfg.cyclic_prefix_length) * cfg.num_ofdm_symbols
    signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    signal = signal.astype(np.complex64)

    # Demodulate
    symbols = rx._demod(signal)

    # Calculate expected number of data subcarriers
    num_lower = cfg.fft_size // 2 - cfg.num_guard_carriers[0]
    num_upper = (cfg.fft_size - cfg.num_guard_carriers[1]) - (
        cfg.fft_size // 2 + (1 if cfg.dc_null else 0)
    )
    num_data_subcarriers = num_lower + num_upper

    assert symbols.shape[0] == num_data_subcarriers
    assert symbols.shape[1] == cfg.num_ofdm_symbols

    print("\n[Rx Demodulation]:")
    print(f"  input samples: {num_samples}")
    print(f"  output symbols shape: {symbols.shape}")


def test_rx_equalize():
    """Test Rx per-subcarrier equalization."""
    cfg = Config()

    rx = Rx(
        signal_fs=cfg.signal_sample_rate,
        pa_sample_rate=122.88e6,
        fft_size=cfg.fft_size,
        cp_length=cfg.cyclic_prefix_length,
        num_ofdm_symbols=cfg.num_ofdm_symbols,
        num_guard_lower=cfg.num_guard_carriers[0],
        num_guard_upper=cfg.num_guard_carriers[1],
        dc_null=cfg.dc_null,
    )

    # Create test symbols with a known channel
    num_subcarriers = 100
    num_symbols = 8

    # Transmitted symbols (unit power)
    tx = tf.complex(
        tf.random.normal([num_subcarriers, num_symbols]),
        tf.random.normal([num_subcarriers, num_symbols]),
    )

    # Apply a per-subcarrier channel
    H = tf.complex(
        tf.random.uniform([num_subcarriers, 1], 0.5, 2.0),
        tf.random.uniform([num_subcarriers, 1], -0.5, 0.5),
    )
    rx_symbols = tx * H

    # Equalize
    equalized = rx._equalize(rx_symbols, tx)

    # After equalization, should be close to original tx
    error = tf.reduce_mean(tf.abs(equalized - tx) ** 2)
    assert error < 1e-6, f"Equalization error too high: {float(error)}"

    print("\n[Rx Equalization]:")
    print(f"  tx shape: {tx.shape}")
    print(f"  rx shape: {rx_symbols.shape}")
    print(f"  equalized shape: {equalized.shape}")
    print(f"  equalization MSE: {float(error):.2e}")


def test_rx_compute_evm():
    """Test Rx EVM computation."""
    # Create reference and received symbols
    num_symbols = 1000

    # Perfect transmission (no error)
    tx_perfect = np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)
    evm_perfect = Rx._compute_evm(tx_perfect, tx_perfect)
    assert evm_perfect < 1e-10, "EVM should be ~0 for perfect transmission"

    # Add noise
    noise_power = 0.01
    rx_noisy = tx_perfect + np.sqrt(noise_power) * (
        np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)
    )
    evm_noisy = Rx._compute_evm(rx_noisy, tx_perfect)

    # EVM should be positive and reasonable
    assert evm_noisy > 0
    assert evm_noisy < 100  # Should be less than 100%

    print("\n[Rx EVM Computation]:")
    print(f"  EVM (perfect): {evm_perfect:.6f}%")
    print(f"  EVM (noisy): {evm_noisy:.2f}%")


def test_rx_evm_scaling():
    """Test that EVM scales correctly with noise level."""
    num_symbols = 1000
    tx = np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)

    noise_levels = [0.001, 0.01, 0.1]
    evms = []

    for noise_power in noise_levels:
        rx = tx + np.sqrt(noise_power) * (
            np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)
        )
        evm = Rx._compute_evm(rx, tx)
        evms.append(evm)

    # EVM should increase with noise level
    assert evms[0] < evms[1] < evms[2]

    print("\n[Rx EVM Scaling]:")
    for nl, evm in zip(noise_levels, evms):
        print(f"  noise_power={nl:.3f}: EVM={evm:.2f}%")
