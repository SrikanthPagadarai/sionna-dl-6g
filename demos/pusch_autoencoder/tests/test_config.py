# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

from demos.pusch_autoencoder.src.config import Config


def test_config_initialization():
    """Test Config initialization and default values."""
    cfg = Config()

    # Test basic system parameters
    assert cfg.subcarrier_spacing == 30e3
    assert cfg.num_time_steps == 14
    assert cfg.num_ue == 4
    assert cfg.num_bs == 1
    assert cfg.num_ue_ant == 4
    assert cfg.num_bs_ant == 16  # Default value

    # Test PUSCH parameters
    assert cfg.num_prb == 16
    assert cfg.mcs_index == 14
    assert cfg.num_layers == 1
    assert cfg.mcs_table == 1
    assert cfg.domain == "freq"

    print("\n[Config Initialization]:")
    print(f"  Subcarrier spacing: {cfg.subcarrier_spacing} Hz")
    print(f"  Num UE: {cfg.num_ue}, Num BS: {cfg.num_bs}")
    print(f"  UE antennas: {cfg.num_ue_ant}, BS antennas: {cfg.num_bs_ant}")
    print(f"  MCS index: {cfg.mcs_index}, MCS table: {cfg.mcs_table}")


def test_config_num_bs_ant_configurable():
    """Test that num_bs_ant can be set at initialization."""
    # Test custom values
    cfg_32 = Config(num_bs_ant=32)
    assert cfg_32.num_bs_ant == 32

    cfg_8 = Config(num_bs_ant=8)
    assert cfg_8.num_bs_ant == 8

    cfg_64 = Config(num_bs_ant=64)
    assert cfg_64.num_bs_ant == 64

    # Verify default still works
    cfg_default = Config()
    assert cfg_default.num_bs_ant == 16

    print("\n[Config num_bs_ant Configurable]:")
    print(f"  Default: {cfg_default.num_bs_ant}")
    print(f"  Custom (32): {cfg_32.num_bs_ant}")
    print(f"  Custom (8): {cfg_8.num_bs_ant}")
    print(f"  Custom (64): {cfg_64.num_bs_ant}")


def test_config_mcs_decoding():
    """Test that MCS decoding properly sets modulation and coderate."""
    cfg = Config()

    # Check that num_bits_per_symbol and target_coderate are properly set
    assert hasattr(cfg, "_num_bits_per_symbol")
    assert hasattr(cfg, "_target_coderate")
    assert cfg.num_bits_per_symbol > 0
    assert 0 < cfg.target_coderate <= 1

    print("\n[Config MCS Decoding]:")
    print(f"  Num bits per symbol: {cfg.num_bits_per_symbol}")
    print(f"  Target coderate: {cfg.target_coderate:.4f}")


def test_config_pusch_pilot_indices():
    """Test PUSCH pilot indices initialization."""
    cfg = Config()

    # Check that pilot indices are initialized
    assert isinstance(cfg.pusch_pilot_indices, list)
    assert len(cfg.pusch_pilot_indices) == 2

    print("\n[Config PUSCH Pilot]:")
    print(f"  Pilot indices: {cfg.pusch_pilot_indices}")


def test_config_properties():
    """Test that all property getters work correctly."""
    cfg = Config()

    # Test read-only properties
    properties_to_test = [
        "subcarrier_spacing",
        "num_time_steps",
        "num_ue",
        "num_bs",
        "num_ue_ant",
        "num_bs_ant",
        "batch_size_cir",
        "target_num_cirs",
        "max_depth",
        "min_gain_db",
        "max_gain_db",
        "min_dist_m",
        "max_dist_m",
        "batch_size",
        "seed",
        "num_prb",
        "mcs_index",
        "num_layers",
        "mcs_table",
        "domain",
        "num_bits_per_symbol",
        "target_coderate",
    ]

    for prop in properties_to_test:
        value = getattr(cfg, prop)
        assert value is not None, f"Property {prop} should not be None"

    print("\n[Config Properties]:")
    print(f"  All {len(properties_to_test)} properties accessible")
