# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""Tests for NeuralNetworkDPD using Indirect Learning Architecture."""

import pytest
import tensorflow as tf
from demos.dpd.src.nn_dpd import NeuralNetworkDPD, ResidualBlock


# ============================================================================
# ResidualBlock Tests
# ============================================================================


def test_residual_block_initialization():
    """Test ResidualBlock initialization."""
    block = ResidualBlock(units=64, num_layers=2)

    assert block.units == 64
    assert block.num_layers == 2
    assert len(block._dense_layers) == 2
    assert len(block._layer_norms) == 2
    assert len(block._activations) == 2

    print("\n[ResidualBlock Initialization]:")
    print(f"  units: {block.units}")
    print(f"  num_layers: {block.num_layers}")


def test_residual_block_forward():
    """Test ResidualBlock forward pass."""
    block = ResidualBlock(units=64, num_layers=2)

    # Input shape: [batch, features]
    x = tf.random.normal([4, 64])
    y = block(x)

    assert y.shape == x.shape

    print("\n[ResidualBlock Forward]:")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {y.shape}")


def test_residual_block_skip_connection():
    """Test that ResidualBlock has skip connection."""
    block = ResidualBlock(units=64, num_layers=2)

    x = tf.random.normal([4, 64])
    y = block(x)

    # With random initialization, output should not be identical to input
    # but should have similar magnitude due to skip connection
    diff = tf.reduce_mean(tf.abs(y - x))
    assert diff > 0, "Output should differ from input"
    assert diff < 10, "Output should be similar magnitude to input (skip connection)"

    print("\n[ResidualBlock Skip Connection]:")
    print(f"  mean diff from input: {float(diff):.4f}")


def test_residual_block_invalid_layers():
    """Test that ResidualBlock raises for invalid num_layers."""
    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        ResidualBlock(units=64, num_layers=0)


# ============================================================================
# NeuralNetworkDPD Tests
# ============================================================================


def test_nn_dpd_initialization():
    """Test NeuralNetworkDPD initialization with default parameters."""
    dpd = NeuralNetworkDPD()

    assert dpd._memory_depth == 4
    assert dpd._num_filters == 64
    assert dpd._num_layers_per_block == 2
    assert dpd._num_res_blocks == 3
    assert dpd._input_size == 20  # 5 * memory_depth (real, imag, |x|^2, |x|^4, |x|^6)

    print("\n[NN-DPD Initialization]:")
    print(f"  memory_depth: {dpd._memory_depth}")
    print(f"  num_filters: {dpd._num_filters}")
    print(f"  num_res_blocks: {dpd._num_res_blocks}")
    print(f"  input_size: {dpd._input_size}")


def test_nn_dpd_initialization_custom():
    """Test NeuralNetworkDPD initialization with custom parameters."""
    dpd = NeuralNetworkDPD(
        memory_depth=8,
        num_filters=128,
        num_layers_per_block=3,
        num_res_blocks=4,
    )

    assert dpd._memory_depth == 8
    assert dpd._num_filters == 128
    assert dpd._num_layers_per_block == 3
    assert dpd._num_res_blocks == 4

    print("\n[NN-DPD Initialization Custom]:")
    print(f"  memory_depth: {dpd._memory_depth}")
    print(f"  num_filters: {dpd._num_filters}")
    print(f"  num_res_blocks: {dpd._num_res_blocks}")


def test_nn_dpd_sliding_windows():
    """Test sliding window feature extraction."""
    dpd = NeuralNetworkDPD(memory_depth=4)

    batch_size = 2
    num_samples = 100
    x = tf.complex(
        tf.random.normal([batch_size, num_samples]),
        tf.random.normal([batch_size, num_samples]),
    )

    features = dpd._create_sliding_windows_batched(x)

    # Output shape: [batch, num_samples, 5 * memory_depth]
    # Features: real, imag, |x|^2, |x|^4, |x|^6 for each memory tap
    assert features.shape == (batch_size, num_samples, 20)
    assert features.dtype == tf.float32

    print("\n[NN-DPD Sliding Windows]:")
    print(f"  input shape: {x.shape}")
    print(f"  features shape: {features.shape}")


def test_nn_dpd_forward_1d():
    """Test NN-DPD forward pass with 1D input."""
    dpd = NeuralNetworkDPD()

    num_samples = 100
    x = tf.complex(
        tf.random.normal([num_samples], stddev=0.1),
        tf.random.normal([num_samples], stddev=0.1),
    )

    y = dpd(x)

    assert y.shape == x.shape
    assert y.dtype == tf.complex64

    print("\n[NN-DPD Forward 1D]:")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {y.shape}")


def test_nn_dpd_forward_2d_batched():
    """Test NN-DPD forward pass with 2D batched input."""
    dpd = NeuralNetworkDPD()

    batch_size = 4
    num_samples = 100
    x = tf.complex(
        tf.random.normal([batch_size, num_samples], stddev=0.1),
        tf.random.normal([batch_size, num_samples], stddev=0.1),
    )

    y = dpd(x)

    assert y.shape == x.shape
    assert y.shape[0] == batch_size
    assert y.shape[1] == num_samples

    print("\n[NN-DPD Forward 2D Batched]:")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {y.shape}")


def test_nn_dpd_skip_connection_initial():
    """Test that NN-DPD starts as near-identity due to zero initialization."""
    dpd = NeuralNetworkDPD()

    x = tf.complex(
        tf.random.normal([100], stddev=0.1),
        tf.random.normal([100], stddev=0.1),
    )

    y = dpd(x)

    # With zero initialization of output layer, should be close to identity
    mse = tf.reduce_mean(tf.abs(y - x) ** 2)
    assert mse < 0.01, f"Initial NN should be near-identity, MSE={float(mse)}"

    print("\n[NN-DPD Skip Connection Initial]:")
    print(f"  MSE from input: {float(mse):.6f}")


def test_nn_dpd_trainable_variables():
    """Test that NN-DPD has trainable variables."""
    dpd = NeuralNetworkDPD()

    # Build the model by calling it
    x = tf.complex(
        tf.random.normal([10], stddev=0.1),
        tf.random.normal([10], stddev=0.1),
    )
    _ = dpd(x)

    trainable_vars = dpd.trainable_variables

    # Should have multiple trainable variables
    assert len(trainable_vars) > 0

    # Count parameters
    total_params = sum(tf.reduce_prod(v.shape) for v in trainable_vars)

    print("\n[NN-DPD Trainable Variables]:")
    print(f"  num variables: {len(trainable_vars)}")
    print(f"  total parameters: {int(total_params)}")


def test_nn_dpd_differentiable():
    """Test that NN-DPD is differentiable."""
    dpd = NeuralNetworkDPD()

    x = tf.complex(
        tf.random.normal([2, 100], stddev=0.1),
        tf.random.normal([2, 100], stddev=0.1),
    )

    with tf.GradientTape() as tape:
        y = dpd(x, training=True)
        loss = tf.reduce_mean(tf.abs(y) ** 2)

    grads = tape.gradient(loss, dpd.trainable_variables)

    # All gradients should be non-None
    for i, g in enumerate(grads):
        assert g is not None, f"Gradient {i} should not be None"

    print("\n[NN-DPD Differentiability]:")
    print(f"  loss: {float(loss):.6f}")
    print(f"  num gradients: {len(grads)}")
    print("  all gradients non-None: True")


def test_nn_dpd_output_to_complex():
    """Test output to complex conversion."""
    dpd = NeuralNetworkDPD()

    # Test with 2D output
    output_2d = tf.random.normal([100, 2])
    complex_2d = dpd._output_to_complex(output_2d)
    assert complex_2d.shape == (100,)
    assert complex_2d.dtype == tf.complex64

    # Test with 3D output
    output_3d = tf.random.normal([4, 100, 2])
    complex_3d = dpd._output_to_complex(output_3d)
    assert complex_3d.shape == (4, 100)

    print("\n[NN-DPD Output to Complex]:")
    print(f"  2D output {output_2d.shape} -> {complex_2d.shape}")
    print(f"  3D output {output_3d.shape} -> {complex_3d.shape}")


@pytest.mark.parametrize("memory_depth", [2, 4, 8])
def test_nn_dpd_various_memory_depths(memory_depth):
    """Test NN-DPD with various memory depths."""
    dpd = NeuralNetworkDPD(memory_depth=memory_depth)

    x = tf.complex(
        tf.random.normal([50], stddev=0.1),
        tf.random.normal([50], stddev=0.1),
    )

    y = dpd(x)

    assert y.shape == x.shape

    print(f"\n[NN-DPD Memory Depth {memory_depth}]:")
    print(f"  input_size: {dpd._input_size}")
    print(f"  output shape: {y.shape}")


def test_nn_dpd_graph_mode():
    """Test that NN-DPD works in TensorFlow graph mode."""
    dpd = NeuralNetworkDPD()

    @tf.function
    def forward_fn(x):
        return dpd(x, training=False)

    x = tf.complex(
        tf.random.normal([2, 100], stddev=0.1),
        tf.random.normal([2, 100], stddev=0.1),
    )

    y = forward_fn(x)

    assert y.shape == x.shape

    print("\n[NN-DPD Graph Mode]:")
    print("  Successfully ran in tf.function")
    print(f"  output shape: {y.shape}")
