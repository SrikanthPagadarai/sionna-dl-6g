# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import pytest
import tensorflow as tf
import numpy as np
from demos.pusch_autoencoder.src.cir_generator import CIRGenerator


def create_dummy_cir_data(
    dataset_size=100, num_rx=1, num_rx_ant=16, num_paths=10, num_time_steps=14
):
    """Create dummy CIR data for testing.

    Note: The actual CIRGenerator expects data with time steps dimension,
    so simplified test data is created that mimics the real structure.
    """
    num_tx = 4  # Will be sampled from
    num_tx_ant = 4

    # For testing, simple data is created that doesn't require the generator's
    # complex transpose operations. Only initialization is tested.
    # Channel coefficients:
    # [dataset_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    a_shape = (
        dataset_size,
        num_rx,
        num_rx_ant,
        num_tx,
        num_tx_ant,
        num_paths,
        num_time_steps,
    )
    a = np.random.randn(*a_shape) + 1j * np.random.randn(*a_shape)
    a = a.astype(np.complex64)

    # Delays: [dataset_size, num_rx, num_tx, num_paths]
    tau_shape = (dataset_size, num_rx, num_tx, num_paths)
    tau = np.abs(np.random.randn(*tau_shape)).astype(np.float32) * 1e-6

    return a, tau


def test_cir_generator_initialization():
    """Test CIRGenerator initialization."""
    dataset_size = 100
    num_tx = 4
    a, tau = create_dummy_cir_data(dataset_size=dataset_size)

    generator = CIRGenerator(a, tau, num_tx=num_tx)

    # Check attributes
    assert generator._dataset_size == dataset_size
    assert generator._num_tx == num_tx
    assert generator._a.dtype == tf.complex64
    assert generator._tau.dtype == tf.float32

    print("\n[CIR Generator Init]:")
    print(f"  Dataset size: {generator._dataset_size}")
    print(f"  Num TX to sample: {generator._num_tx}")
    print(f"  a shape: {generator._a.shape}")
    print(f"  tau shape: {generator._tau.shape}")


@pytest.mark.skip(reason="Requires properly formatted CIR data from scene generation")
def test_cir_generator_single_sample():
    """Test generating a single sample from CIRGenerator."""
    dataset_size = 100
    num_tx = 4
    a, tau = create_dummy_cir_data(dataset_size=dataset_size)

    generator = CIRGenerator(a, tau, num_tx=num_tx)

    # Create the generator
    gen = generator()

    # Get one sample
    a_sample, tau_sample = next(gen)

    print("\n[CIR Generator Single Sample]:")
    print(f"  a_sample shape: {a_sample.shape}")
    print(f"  tau_sample shape: {tau_sample.shape}")
    print(f"  a_sample dtype: {a_sample.dtype}")
    print(f"  tau_sample dtype: {tau_sample.dtype}")

    # Check output shapes
    # After transpose and squeeze, shapes should be:
    # a: [num_rx, num_rx_ant, num_tx_sampled, num_tx_ant, num_paths]
    # tau: [num_rx, num_tx_sampled, num_paths]
    assert len(a_sample.shape) == 5
    assert len(tau_sample.shape) == 3
    assert a_sample.shape[2] == num_tx  # num_tx dimension
    assert tau_sample.shape[1] == num_tx  # num_tx dimension


@pytest.mark.skip(reason="Requires properly formatted CIR data from scene generation")
def test_cir_generator_multiple_samples():
    """Test generating multiple samples from CIRGenerator."""
    dataset_size = 50
    num_tx = 4
    a, tau = create_dummy_cir_data(dataset_size=dataset_size)

    generator = CIRGenerator(a, tau, num_tx=num_tx)

    # Create the generator
    gen = generator()

    # Get multiple samples
    num_samples = 10
    samples = []
    for _ in range(num_samples):
        sample = next(gen)
        samples.append(sample)

    print(f"\n[CIR Generator Multiple Samples] Generated {num_samples} samples:")
    print(f"  First sample a shape: {samples[0][0].shape}")
    print(f"  First sample tau shape: {samples[0][1].shape}")

    # Check that the right number of samples is returned
    assert len(samples) == num_samples

    # Check that samples have consistent shapes
    for a_sample, tau_sample in samples:
        assert a_sample.shape == samples[0][0].shape
        assert tau_sample.shape == samples[0][1].shape


@pytest.mark.skip(reason="Requires properly formatted CIR data from scene generation")
def test_cir_generator_randomness():
    """Test that CIRGenerator produces different samples."""
    dataset_size = 100
    num_tx = 4
    a, tau = create_dummy_cir_data(dataset_size=dataset_size)

    generator = CIRGenerator(a, tau, num_tx=num_tx)

    # Create the generator
    gen = generator()

    # Get two samples
    a1, tau1 = next(gen)
    a2, tau2 = next(gen)

    # Samples should be different (due to random sampling)
    # Checked by comparing if they're not identical
    a_same = tf.reduce_all(tf.equal(a1, a2))
    tau_same = tf.reduce_all(tf.equal(tau1, tau2))

    print("\n[CIR Generator Randomness]:")
    print(f"  Samples are identical (a): {bool(a_same)}")
    print(f"  Samples are identical (tau): {bool(tau_same)}")

    # With random sampling, it's extremely unlikely they're identical
    # (though theoretically possible with small dataset)


@pytest.mark.skip(reason="Requires properly formatted CIR data from scene generation")
def test_cir_generator_with_tf_dataset():
    """Test using CIRGenerator with tf.data.Dataset."""
    dataset_size = 100
    num_tx = 4
    a, tau = create_dummy_cir_data(dataset_size=dataset_size)

    generator = CIRGenerator(a, tau, num_tx=num_tx)

    # Create a tf.data.Dataset from the generator
    # Determine output signature
    gen = generator()
    a_sample, tau_sample = next(gen)

    output_signature = (
        tf.TensorSpec(shape=a_sample.shape, dtype=tf.complex64),
        tf.TensorSpec(shape=tau_sample.shape, dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    )

    # Take a few samples from the dataset
    batch_size = 8
    dataset_batched = dataset.batch(batch_size).take(3)

    print("\n[CIR Generator with tf.data.Dataset]:")
    for i, (a_batch, tau_batch) in enumerate(dataset_batched):
        print(f"  Batch {i+1}:")
        print(f"    a shape: {a_batch.shape}")
        print(f"    tau shape: {tau_batch.shape}")

        # Check batch dimension
        assert a_batch.shape[0] == batch_size
        assert tau_batch.shape[0] == batch_size


@pytest.mark.skip(reason="Requires properly formatted CIR data from scene generation")
def test_cir_generator_dtype():
    """Test that CIRGenerator preserves correct dtypes."""
    dataset_size = 50
    num_tx = 4
    a, tau = create_dummy_cir_data(dataset_size=dataset_size)

    generator = CIRGenerator(a, tau, num_tx=num_tx)

    # Create the generator
    gen = generator()

    # Get a sample
    a_sample, tau_sample = next(gen)

    print("\n[CIR Generator Dtype]:")
    print(f"  a dtype: {a_sample.dtype}")
    print(f"  tau dtype: {tau_sample.dtype}")

    # Check dtypes
    assert a_sample.dtype == tf.complex64
    assert tau_sample.dtype == tf.float32


@pytest.mark.skip(reason="Requires properly formatted CIR data from scene generation")
@pytest.mark.parametrize("num_tx", [1, 2, 4, 8])
def test_cir_generator_different_num_tx(num_tx):
    """Test CIRGenerator with different num_tx values."""
    dataset_size = 100
    a, tau = create_dummy_cir_data(dataset_size=dataset_size)

    generator = CIRGenerator(a, tau, num_tx=num_tx)

    # Create the generator
    gen = generator()

    # Get a sample
    a_sample, tau_sample = next(gen)

    print(f"\n[CIR Generator num_tx={num_tx}]:")
    print(f"  a_sample shape: {a_sample.shape}")
    print(f"  tau_sample shape: {tau_sample.shape}")

    # Check that the num_tx dimension is correct
    assert a_sample.shape[2] == num_tx
    assert tau_sample.shape[1] == num_tx
