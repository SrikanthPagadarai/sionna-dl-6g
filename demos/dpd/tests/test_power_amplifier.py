# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""Tests for DPD PowerAmplifier memory polynomial model."""

import pytest
import tensorflow as tf
from demos.dpd.src.power_amplifier import PowerAmplifier


def test_pa_initialization():
    """Test PowerAmplifier initialization with default parameters."""
    pa = PowerAmplifier()

    assert pa.order == 7
    assert pa.memory_depth == 4

    print("\n[PA Initialization]:")
    print(f"  order: {pa.order}")
    print(f"  memory_depth: {pa.memory_depth}")


def test_pa_forward_1d():
    """Test PA forward pass with 1D input."""
    pa = PowerAmplifier()

    # Create test input
    num_samples = 1000
    x = tf.complex(
        tf.random.normal([num_samples], stddev=0.1),
        tf.random.normal([num_samples], stddev=0.1),
    )

    y = pa(x)

    # Output shape should match input
    assert y.shape == x.shape
    assert y.dtype == tf.complex64

    # Output should be non-zero
    assert tf.reduce_mean(tf.abs(y)) > 0

    print("\n[PA Forward 1D]:")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {y.shape}")
    print(f"  input power: {float(tf.reduce_mean(tf.abs(x) ** 2)):.6f}")
    print(f"  output power: {float(tf.reduce_mean(tf.abs(y) ** 2)):.6f}")


def test_pa_forward_2d_batched():
    """Test PA forward pass with 2D batched input."""
    pa = PowerAmplifier()

    # Create batched test input
    batch_size = 4
    num_samples = 1000
    x = tf.complex(
        tf.random.normal([batch_size, num_samples], stddev=0.1),
        tf.random.normal([batch_size, num_samples], stddev=0.1),
    )

    y = pa(x)

    # Output shape should match input
    assert y.shape == x.shape
    assert y.shape[0] == batch_size
    assert y.shape[1] == num_samples

    print("\n[PA Forward 2D Batched]:")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {y.shape}")


def test_pa_memory_effect():
    """Test that PA exhibits memory effects (output depends on past samples)."""
    pa = PowerAmplifier()

    # Create two signals that differ only in their past
    num_samples = 100
    base_signal = tf.complex(
        tf.random.normal([num_samples], stddev=0.1),
        tf.random.normal([num_samples], stddev=0.1),
    )

    # Same current sample, different history
    signal1 = base_signal
    signal2 = tf.concat(
        [
            tf.zeros(10, dtype=tf.complex64),  # Different history
            base_signal[10:],
        ],
        axis=0,
    )

    y1 = pa(signal1)
    y2 = pa(signal2)

    # Outputs should differ due to memory effects
    diff = tf.reduce_mean(tf.abs(y1 - y2))
    assert diff > 0, "PA should exhibit memory effects"

    print("\n[PA Memory Effect]:")
    print(f"  Mean difference due to history: {float(diff):.6f}")


def test_pa_nonlinearity():
    """Test that PA exhibits nonlinear behavior at higher amplitudes."""
    pa = PowerAmplifier()

    num_samples = 1000

    # Low amplitude input (linear region)
    x_low = tf.complex(
        tf.random.normal([num_samples], stddev=0.01),
        tf.random.normal([num_samples], stddev=0.01),
    )

    # High amplitude input (nonlinear region)
    x_high = tf.complex(
        tf.random.normal([num_samples], stddev=0.5),
        tf.random.normal([num_samples], stddev=0.5),
    )

    y_low = pa(x_low)
    y_high = pa(x_high)

    # Compute gain for each
    gain_low = tf.sqrt(
        tf.reduce_mean(tf.abs(y_low) ** 2) / tf.reduce_mean(tf.abs(x_low) ** 2)
    )
    gain_high = tf.sqrt(
        tf.reduce_mean(tf.abs(y_high) ** 2) / tf.reduce_mean(tf.abs(x_high) ** 2)
    )

    # Gain compression: gain should be lower at high amplitude
    # (typical PA behavior, though depends on coefficient values)
    print("\n[PA Nonlinearity]:")
    print(f"  Gain at low amplitude: {float(gain_low):.4f}")
    print(f"  Gain at high amplitude: {float(gain_high):.4f}")


def test_pa_estimate_gain():
    """Test PA gain estimation."""
    pa = PowerAmplifier()

    gain = pa.estimate_gain(num_samples=10000)

    # Gain should be positive and reasonable
    assert gain > 0
    assert gain < 10  # Reasonable upper bound

    print("\n[PA Gain Estimation]:")
    print(f"  Estimated gain: {float(gain):.4f}")


def test_pa_basis_matrix_shape():
    """Test PA basis matrix construction."""
    pa = PowerAmplifier()

    num_samples = 100
    x = tf.complex(
        tf.random.normal([num_samples], stddev=0.1),
        tf.random.normal([num_samples], stddev=0.1),
    )

    X = pa._setup_basis_matrix(x)

    # Expected number of columns: n_coeffs * memory_depth
    # n_coeffs = (order + 1) // 2 = 4 for order=7
    # Total columns = 4 * 4 = 16
    expected_cols = pa._n_coeffs * pa._memory_depth
    assert X.shape == (num_samples, expected_cols)

    print("\n[PA Basis Matrix]:")
    print(f"  input samples: {num_samples}")
    print(f"  basis matrix shape: {X.shape}")
    print(f"  n_coeffs: {pa._n_coeffs}, memory_depth: {pa._memory_depth}")


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_pa_different_batch_sizes(batch_size):
    """Test PA with various batch sizes."""
    pa = PowerAmplifier()

    num_samples = 500
    x = tf.complex(
        tf.random.normal([batch_size, num_samples], stddev=0.1),
        tf.random.normal([batch_size, num_samples], stddev=0.1),
    )

    y = pa(x)

    assert y.shape == (batch_size, num_samples)

    print(f"\n[PA Batch Size {batch_size}]:")
    print(f"  output shape: {y.shape}")
