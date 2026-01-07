# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import pytest
import tensorflow as tf
from sionna.phy.nr import PUSCHConfig
from demos.pusch_autoencoder.src.config import Config
from demos.pusch_autoencoder.src.pusch_trainable_transmitter import (
    PUSCHTrainableTransmitter,
)


def get_pusch_config(cfg):
    """Helper to create PUSCHConfig."""
    pusch_config = PUSCHConfig()
    pusch_config.carrier.subcarrier_spacing = cfg.subcarrier_spacing / 1000.0
    pusch_config.carrier.n_size_grid = cfg.num_prb
    pusch_config.num_antenna_ports = cfg.num_ue_ant
    pusch_config.num_layers = cfg.num_layers
    pusch_config.precoding = "codebook"
    pusch_config.tpmi = 1
    pusch_config.dmrs.dmrs_port_set = list(range(cfg.num_layers))
    pusch_config.dmrs.config_type = 1
    pusch_config.dmrs.length = 1
    pusch_config.dmrs.additional_position = 1
    pusch_config.dmrs.num_cdm_groups_without_data = 2
    pusch_config.tb.mcs_index = cfg.mcs_index
    pusch_config.tb.mcs_table = cfg.mcs_table
    return pusch_config


@pytest.mark.parametrize("training", [True, False])
def test_trainable_transmitter_initialization(training):
    """Test PUSCHTrainableTransmitter initialization."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    tx = PUSCHTrainableTransmitter(
        [pusch_config], output_domain="freq", training=training
    )

    # Check that constellation variables are created
    assert hasattr(tx, "_points_r")
    assert hasattr(tx, "_points_i")
    assert isinstance(tx._points_r, tf.Variable)
    assert isinstance(tx._points_i, tf.Variable)

    # Check trainability
    assert tx._points_r.trainable == training
    assert tx._points_i.trainable == training

    print(f"\n[Trainable TX Init] Training={training}:")
    print(f"  Constellation real: {tx._points_r.shape}")
    print(f"  Constellation imag: {tx._points_i.shape}")
    print(f"  Trainable: {tx._points_r.trainable}")


@pytest.mark.parametrize("training", [True, False])
def test_trainable_transmitter_forward(training):
    """Test PUSCHTrainableTransmitter forward pass."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)
    batch_size = 4

    tx = PUSCHTrainableTransmitter(
        [pusch_config], output_domain="freq", training=training
    )

    # Forward pass
    output = tx(batch_size)

    # The trainable transmitter returns (x_map, x, b, c) when return_bits=True (default)
    # The training parameter only controls trainability, not output structure
    x_map, x, b, c = output
    assert x_map.shape[0] == batch_size
    assert x.shape[0] == batch_size
    assert b.shape[0] == batch_size
    assert c.shape[0] == batch_size

    print(f"\n[Trainable TX Forward] Training={training}:")
    print(f"  x_map shape: {x_map.shape}")
    print(f"  x shape: {x.shape}")
    print(f"  b shape: {b.shape}")
    print(f"  c shape: {c.shape}")


def test_normalized_constellation():
    """Test constellation normalization."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    tx = PUSCHTrainableTransmitter([pusch_config], output_domain="freq", training=True)

    # Get normalized constellation
    constellation = tx.get_normalized_constellation()

    # Check unit power (average energy should be 1)
    energy = tf.reduce_mean(tf.square(tf.abs(constellation)))

    print("\n[Constellation Normalization]:")
    print(f"  Constellation shape: {constellation.shape}")
    print(f"  Average energy: {float(energy):.6f} (should be ~1.0)")
    print(f"  Mean: {tf.reduce_mean(constellation)}")

    # Energy should be close to 1.0
    assert tf.abs(energy - 1.0) < 0.1


def test_trainable_variables():
    """Test that trainable variables are correctly exposed."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    # Training mode
    tx_train = PUSCHTrainableTransmitter(
        [pusch_config], output_domain="freq", training=True
    )
    train_vars = tx_train.trainable_variables
    assert len(train_vars) == 2  # _points_r and _points_i
    assert all(isinstance(v, tf.Variable) for v in train_vars)
    assert all(v.trainable for v in train_vars)

    # Inference mode
    tx_infer = PUSCHTrainableTransmitter(
        [pusch_config], output_domain="freq", training=False
    )
    infer_vars = tx_infer.trainable_variables
    assert len(infer_vars) == 2
    assert not any(v.trainable for v in infer_vars)

    print("\n[Trainable Variables]:")
    print(f"  Training mode vars: {len(train_vars)} trainable")
    print(f"  Inference mode vars: {len(infer_vars)} non-trainable")


def test_constellation_update_during_call():
    """Test that constellation is updated with normalized values during call."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)
    batch_size = 2

    tx = PUSCHTrainableTransmitter([pusch_config], output_domain="freq", training=True)

    # Modify constellation variables
    tx._points_r.assign(tx._points_r * 2.0)
    tx._points_i.assign(tx._points_i * 2.0)

    # Call should normalize internally
    _ = tx(batch_size)

    # Get the constellation from the mapper (which should be normalized)
    constellation = tx._constellation.points

    # Check that it's normalized (unit power)
    energy = tf.reduce_mean(tf.square(tf.abs(constellation)))

    print("\n[Constellation Update]:")
    print(f"  Energy after scaling: {float(energy):.6f}")

    assert tf.abs(energy - 1.0) < 0.1
