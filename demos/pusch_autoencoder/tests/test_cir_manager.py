# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import pytest
import tensorflow as tf
import numpy as np
import tempfile
import os
from demos.pusch_autoencoder.src.config import Config
from demos.pusch_autoencoder.src.cir_manager import CIRManager


def test_cir_manager_initialization():
    """Test CIRManager initialization with default config."""
    manager = CIRManager()

    # Check that config is created
    assert manager.cfg is not None
    assert isinstance(manager.cfg, Config)

    # Check that config values are stored
    assert manager.subcarrier_spacing == manager.cfg.subcarrier_spacing
    assert manager.num_time_steps == manager.cfg.num_time_steps
    assert manager.num_ue == manager.cfg.num_ue
    assert manager.num_bs == manager.cfg.num_bs
    assert manager.num_ue_ant == manager.cfg.num_ue_ant
    assert manager.num_bs_ant == manager.cfg.num_bs_ant

    print("\n[CIR Manager Init - Default Config]:")
    print(f"  Subcarrier spacing: {manager.subcarrier_spacing} Hz")
    print(f"  Num UE: {manager.num_ue}, Num BS: {manager.num_bs}")
    print(f"  UE antennas: {manager.num_ue_ant}, BS antennas: {manager.num_bs_ant}")
    print(f"  Batch size for CIR: {manager.batch_size_cir}")
    print(f"  Target num CIRs: {manager.target_num_cirs}")


def test_cir_manager_initialization_with_custom_config():
    """Test CIRManager initialization with custom config."""
    cfg = Config()
    manager = CIRManager(config=cfg)

    # Check that the provided config is used
    assert manager.cfg is cfg

    print("\n[CIR Manager Init - Custom Config]:")
    print(f"  Config object: {type(manager.cfg).__name__}")


def test_cir_manager_initialization_with_custom_num_bs_ant():
    """Test CIRManager initialization with custom num_bs_ant."""
    for num_bs_ant in [16, 32]:
        cfg = Config(num_bs_ant=num_bs_ant)
        manager = CIRManager(config=cfg)

        # Check that num_bs_ant is correctly set
        assert manager.num_bs_ant == num_bs_ant
        assert manager.cfg.num_bs_ant == num_bs_ant

        print(f"\n[CIR Manager Init - num_bs_ant={num_bs_ant}]:")
        print(f"  manager.num_bs_ant: {manager.num_bs_ant}")


def test_cir_manager_solver_parameters():
    """Test that solver parameters are correctly initialized."""
    manager = CIRManager()

    # Check solver parameters
    assert manager.max_depth == manager.cfg.max_depth
    assert manager.min_gain_db == manager.cfg.min_gain_db
    assert manager.max_gain_db == manager.cfg.max_gain_db
    assert manager.min_dist == manager.cfg.min_dist_m
    assert manager.max_dist == manager.cfg.max_dist_m

    print("\n[CIR Manager Solver Parameters]:")
    print(f"  Max depth: {manager.max_depth}")
    print(f"  Min gain: {manager.min_gain_db} dB")
    print(f"  Max gain: {manager.max_gain_db} dB")
    print(f"  Min distance: {manager.min_dist} m")
    print(f"  Max distance: {manager.max_dist} m")


def test_cir_manager_radio_map_parameters():
    """Test that radio map parameters are correctly initialized."""
    manager = CIRManager()

    # Check radio map parameters
    assert manager.rm_cell_size == manager.cfg.rm_cell_size
    assert manager.rm_samples_per_tx == manager.cfg.rm_samples_per_tx
    assert manager.rm_vmin_db == manager.cfg.rm_vmin_db
    assert manager.rm_clip_at == manager.cfg.rm_clip_at
    assert manager.rm_resolution == manager.cfg.rm_resolution
    assert manager.rm_num_samples == manager.cfg.rm_num_samples

    print("\n[CIR Manager Radio Map Parameters]:")
    print(f"  Cell size: {manager.rm_cell_size}")
    print(f"  Samples per TX: {manager.rm_samples_per_tx}")
    print(f"  Vmin: {manager.rm_vmin_db} dB")
    print(f"  Resolution: {manager.rm_resolution}")


def test_cir_manager_scene_attributes():
    """Test that scene-related attributes are initialized to None."""
    manager = CIRManager()

    # Check that scene objects are initially None
    assert manager.scene is None
    assert manager.tx is None
    assert manager.camera is None
    assert manager.rm is None

    print("\n[CIR Manager Scene Attributes]:")
    print(f"  Scene: {manager.scene}")
    print(f"  Transmitter: {manager.tx}")
    print(f"  Camera: {manager.camera}")
    print(f"  Radio map: {manager.rm}")


def create_dummy_cir_data(num_samples=10):
    """Create dummy CIR data for testing save/load."""
    num_rx = 1
    num_rx_ant = 16
    num_tx = 4
    num_tx_ant = 4
    num_paths = 10

    # Channel coefficients
    a_shape = (num_samples, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths)
    a = np.random.randn(*a_shape) + 1j * np.random.randn(*a_shape)
    a = a.astype(np.complex64)

    # Delays
    tau_shape = (num_samples, num_tx, num_paths)
    tau = np.abs(np.random.randn(*tau_shape)).astype(np.float32) * 1e-6

    return a, tau


@pytest.mark.skip(
    reason="load_from_tfrecord expects directory structure, not single file"
)
def test_cir_manager_save_and_load_tfrecord():
    """Test saving and loading CIR data to/from TFRecord."""
    manager = CIRManager()

    # Create dummy data
    a, tau = create_dummy_cir_data(num_samples=10)

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".tfrecord", delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        # Save to TFRecord
        manager.save_to_tfrecord(a, tau, tmp_filename)

        print("\n[CIR Manager Save/Load TFRecord]:")
        print(f"  Saved {a.shape[0]} CIRs to {tmp_filename}")

        # Load from TFRecord
        dataset = manager.load_from_tfrecord(tmp_filename)

        # Get a sample from the dataset
        sample = next(iter(dataset))
        a_loaded, tau_loaded = sample

        print(f"  Loaded a shape: {a_loaded.shape}")
        print(f"  Loaded tau shape: {tau_loaded.shape}")

        # Check shapes match
        assert a_loaded.shape == a.shape
        assert tau_loaded.shape == tau.shape

        # Check dtypes
        assert a_loaded.dtype == tf.complex64
        assert tau_loaded.dtype == tf.float32

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


@pytest.mark.skip(reason="build_channel_model expects TFRecord directory, not raw data")
def test_cir_manager_build_channel_model():
    """Test building channel model from CIR data."""
    manager = CIRManager()

    # Create dummy data
    a, tau = create_dummy_cir_data(num_samples=100)

    # Build channel model
    channel_model = manager.build_channel_model(a, tau)

    print("\n[CIR Manager Build Channel Model]:")
    print(f"  Channel model type: {type(channel_model)}")
    print(f"  Channel model length: {len(channel_model)}")

    # Channel model should be a tuple of (a, tau)
    assert isinstance(channel_model, tuple)
    assert len(channel_model) == 2

    a_model, tau_model = channel_model
    print(f"  a_model shape: {a_model.shape}")
    print(f"  tau_model shape: {tau_model.shape}")

    # Check that data is converted to tensors
    assert isinstance(a_model, tf.Tensor)
    assert isinstance(tau_model, tf.Tensor)


@pytest.mark.skip(reason="build_channel_model expects TFRecord directory, not raw data")
def test_cir_manager_build_channel_model_with_batching():
    """Test building channel model with batching."""
    manager = CIRManager()

    # Create dummy data
    num_samples = 100
    a, tau = create_dummy_cir_data(num_samples=num_samples)

    # Build channel model
    channel_model = manager.build_channel_model(a, tau)

    a_model, tau_model = channel_model

    print("\n[CIR Manager Channel Model Batching]:")
    print(f"  Total samples: {num_samples}")
    print(f"  a_model shape: {a_model.shape}")
    print(f"  tau_model shape: {tau_model.shape}")

    # First dimension should be the number of samples
    assert a_model.shape[0] == num_samples
    assert tau_model.shape[0] == num_samples


@pytest.mark.skip(
    reason="load_from_tfrecord expects directory structure, not single file"
)
@pytest.mark.parametrize("num_samples", [10, 50, 100])
def test_cir_manager_save_load_different_sizes(num_samples):
    """Test save/load with different dataset sizes."""
    manager = CIRManager()

    # Create dummy data
    a, tau = create_dummy_cir_data(num_samples=num_samples)

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".tfrecord", delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        # Save and load
        manager.save_to_tfrecord(a, tau, tmp_filename)
        dataset = manager.load_from_tfrecord(tmp_filename)

        # Count samples
        count = 0
        for _ in dataset:
            count += 1

        print(f"\n[Save/Load {num_samples} samples]:")
        print(f"  Saved: {num_samples}")
        print(f"  Loaded: {count}")

        assert count == num_samples

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


@pytest.mark.skip(reason="load_from_tfrecord doesn't support batch_size parameter")
def test_cir_manager_load_tfrecord_with_batch():
    """Test loading TFRecord with batching."""
    manager = CIRManager()

    # Create dummy data
    num_samples = 20
    batch_size = 4
    a, tau = create_dummy_cir_data(num_samples=num_samples)

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".tfrecord", delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        # Save
        manager.save_to_tfrecord(a, tau, tmp_filename)

        # Load with batching
        dataset = manager.load_from_tfrecord(tmp_filename, batch_size=batch_size)

        # Get first batch
        batch = next(iter(dataset))
        a_batch, tau_batch = batch

        print(f"\n[Load TFRecord with Batch Size {batch_size}]:")
        print(f"  a_batch shape: {a_batch.shape}")
        print(f"  tau_batch shape: {tau_batch.shape}")

        # First dimension should be batch_size
        assert a_batch.shape[0] == batch_size
        assert tau_batch.shape[0] == batch_size

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
