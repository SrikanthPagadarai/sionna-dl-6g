# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""Tests for DPD Config dataclass."""

from demos.dpd.src.config import Config


def test_config_initialization():
    """Test Config initialization with default values."""
    cfg = Config()

    # Test mutable parameters have default values
    assert cfg.seed == 42
    assert cfg.batch_size == 100

    print("\n[Config Initialization]:")
    print(f"  seed: {cfg.seed}")
    print(f"  batch_size: {cfg.batch_size}")


def test_config_custom_initialization():
    """Test Config initialization with custom mutable parameters."""
    cfg = Config(seed=123, batch_size=64)

    assert cfg.seed == 123
    assert cfg.batch_size == 64

    print("\n[Config Custom Initialization]:")
    print(f"  seed: {cfg.seed}")
    print(f"  batch_size: {cfg.batch_size}")


def test_config_system_parameters():
    """Test immutable system parameters."""
    cfg = Config()

    # System parameters
    assert cfg.num_ut == 1
    assert cfg.num_ut_ant == 1
    assert cfg.num_streams_per_tx == 1

    print("\n[Config System Parameters]:")
    print(f"  num_ut: {cfg.num_ut}")
    print(f"  num_ut_ant: {cfg.num_ut_ant}")
    print(f"  num_streams_per_tx: {cfg.num_streams_per_tx}")


def test_config_resource_grid_parameters():
    """Test immutable resource grid parameters."""
    cfg = Config()

    # Resource grid parameters
    assert cfg.num_ofdm_symbols == 8
    assert cfg.fft_size == 1024
    assert cfg.subcarrier_spacing == 15000.0
    assert cfg.num_guard_carriers == (200, 199)
    assert cfg.dc_null is True
    assert cfg.cyclic_prefix_length == 72
    assert cfg.pilot_pattern == "kronecker"
    assert cfg.pilot_ofdm_symbol_indices == [2, 6]

    print("\n[Config Resource Grid Parameters]:")
    print(f"  num_ofdm_symbols: {cfg.num_ofdm_symbols}")
    print(f"  fft_size: {cfg.fft_size}")
    print(f"  subcarrier_spacing: {cfg.subcarrier_spacing} Hz")
    print(f"  num_guard_carriers: {cfg.num_guard_carriers}")
    print(f"  dc_null: {cfg.dc_null}")
    print(f"  cyclic_prefix_length: {cfg.cyclic_prefix_length}")


def test_config_modulation_parameters():
    """Test immutable modulation and coding parameters."""
    cfg = Config()

    assert cfg.num_bits_per_symbol == 4  # 16-QAM
    assert cfg.coderate == 0.5

    print("\n[Config Modulation Parameters]:")
    print(f"  num_bits_per_symbol: {cfg.num_bits_per_symbol}")
    print(f"  coderate: {cfg.coderate}")


def test_config_derived_properties():
    """Test derived properties computed from base parameters."""
    cfg = Config()

    # signal_sample_rate = fft_size * subcarrier_spacing
    expected_sample_rate = cfg.fft_size * cfg.subcarrier_spacing
    assert cfg.signal_sample_rate == expected_sample_rate

    print("\n[Config Derived Properties]:")
    print(f"  signal_sample_rate: {cfg.signal_sample_rate / 1e6:.2f} MHz")
    print(
        f"    (fft_size={cfg.fft_size} × subcarrier_spacing={cfg.subcarrier_spacing})"
    )


def test_config_all_properties_accessible():
    """Test that all property getters work correctly."""
    cfg = Config()

    # List of all properties to verify
    properties_to_test = [
        "num_ut",
        "num_ut_ant",
        "num_streams_per_tx",
        "num_ofdm_symbols",
        "fft_size",
        "subcarrier_spacing",
        "num_guard_carriers",
        "dc_null",
        "cyclic_prefix_length",
        "pilot_pattern",
        "pilot_ofdm_symbol_indices",
        "num_bits_per_symbol",
        "coderate",
        "signal_sample_rate",
    ]

    for prop in properties_to_test:
        value = getattr(cfg, prop)
        assert value is not None, f"Property {prop} should not be None"

    print("\n[Config All Properties]:")
    print(f"  All {len(properties_to_test)} properties accessible")
