# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""Tests for DPD Interpolator using Sionna primitives."""

import pytest
import tensorflow as tf
from demos.dpd.src.interpolator import Interpolator


def test_interpolator_initialization():
    """Test Interpolator initialization with default parameters."""
    input_rate = 15.36e6
    output_rate = 122.88e6

    interp = Interpolator(input_rate=input_rate, output_rate=output_rate)

    # Check rate ratio is computed correctly (8x upsampling)
    assert interp._upsample_factor == 8
    assert interp._downsample_factor == 1

    print("\n[Interpolator Initialization]:")
    print(f"  input_rate: {input_rate / 1e6:.2f} MHz")
    print(f"  output_rate: {interp._output_rate / 1e6:.2f} MHz")
    print(f"  upsample_factor: {interp._upsample_factor}")
    print(f"  downsample_factor: {interp._downsample_factor}")
    print(f"  filter_length: {interp._filter_length}")


def test_interpolator_fractional_ratio():
    """Test Interpolator with fractional rate ratio."""
    input_rate = 10e6
    output_rate = 15e6  # 3/2 ratio

    interp = Interpolator(input_rate=input_rate, output_rate=output_rate)

    # Should simplify to 3/2 ratio
    assert interp._upsample_factor == 3
    assert interp._downsample_factor == 2

    print("\n[Interpolator Fractional Ratio]:")
    print(f"  ratio: {output_rate / input_rate}")
    print(f"  upsample_factor: {interp._upsample_factor}")
    print(f"  downsample_factor: {interp._downsample_factor}")


def test_interpolator_forward_1d():
    """Test Interpolator forward pass with 1D input."""
    input_rate = 15.36e6
    output_rate = 122.88e6
    interp = Interpolator(input_rate=input_rate, output_rate=output_rate)

    # Create test input
    num_samples = 1000
    x = tf.complex(
        tf.random.normal([1, num_samples]),
        tf.random.normal([1, num_samples]),
    )

    y, out_rate = interp(x)

    # Output should be upsampled by factor of 8
    expected_out_samples = num_samples * 8
    assert y.shape[1] == expected_out_samples
    assert out_rate == output_rate

    print("\n[Interpolator Forward 1D]:")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {y.shape}")
    print(f"  output_rate: {out_rate / 1e6:.2f} MHz")


def test_interpolator_forward_batched():
    """Test Interpolator forward pass with batched input."""
    input_rate = 15.36e6
    output_rate = 122.88e6
    interp = Interpolator(input_rate=input_rate, output_rate=output_rate)

    # Create batched test input
    batch_size = 4
    num_samples = 500
    x = tf.complex(
        tf.random.normal([batch_size, num_samples]),
        tf.random.normal([batch_size, num_samples]),
    )

    y, out_rate = interp(x)

    assert y.shape[0] == batch_size
    assert y.shape[1] == num_samples * 8

    print("\n[Interpolator Forward Batched]:")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {y.shape}")


def test_interpolator_power_preservation():
    """Test that interpolation approximately preserves signal power."""
    input_rate = 15.36e6
    output_rate = 122.88e6
    interp = Interpolator(input_rate=input_rate, output_rate=output_rate)

    # Create test input
    num_samples = 1000
    x = tf.complex(
        tf.random.normal([1, num_samples]),
        tf.random.normal([1, num_samples]),
    )

    y, _ = interp(x)

    # Compute power
    input_power = tf.reduce_mean(tf.abs(x) ** 2)
    output_power = tf.reduce_mean(tf.abs(y) ** 2)

    # Power should be approximately preserved (within 10%)
    power_ratio = float(output_power / input_power)
    assert 0.9 < power_ratio < 1.1, f"Power ratio {power_ratio} out of expected range"

    print("\n[Interpolator Power Preservation]:")
    print(f"  input power: {float(input_power):.6f}")
    print(f"  output power: {float(output_power):.6f}")
    print(f"  power ratio: {power_ratio:.4f}")


def test_interpolator_custom_filter_params():
    """Test Interpolator with custom filter parameters."""
    input_rate = 15.36e6
    output_rate = 122.88e6

    interp = Interpolator(
        input_rate=input_rate,
        output_rate=output_rate,
        half_len_mult=10,  # Shorter filter
        kaiser_beta=6.0,  # Lower stopband attenuation
    )

    # Filter should still work
    x = tf.complex(
        tf.random.normal([1, 100]),
        tf.random.normal([1, 100]),
    )

    y, _ = interp(x)

    assert y.shape[1] == 800  # 100 * 8

    print("\n[Interpolator Custom Filter]:")
    print(f"  filter_length: {interp._filter_length}")


@pytest.mark.parametrize(
    "input_rate,output_rate",
    [
        (15.36e6, 122.88e6),  # 8x upsample
        (30.72e6, 122.88e6),  # 4x upsample
        (122.88e6, 15.36e6),  # 8x downsample
    ],
)
def test_interpolator_various_rates(input_rate, output_rate):
    """Test Interpolator with various rate ratios."""
    interp = Interpolator(input_rate=input_rate, output_rate=output_rate)

    num_samples = 800  # Divisible by common factors
    x = tf.complex(
        tf.random.normal([1, num_samples]),
        tf.random.normal([1, num_samples]),
    )

    y, out_rate = interp(x)

    # Check output length matches expected ratio
    expected_ratio = output_rate / input_rate
    actual_ratio = y.shape[1] / num_samples

    # Should be close (exact for integer ratios)
    assert abs(actual_ratio - expected_ratio) < 0.01

    print(
        f"\n[Interpolator Rate {input_rate / 1e6:.2f} -> {output_rate / 1e6:.2f} MHz]:"
    )
    print(f"  expected ratio: {expected_ratio:.4f}")
    print(f"  actual ratio: {actual_ratio:.4f}")
    print(f"  output shape: {y.shape}")


def test_interpolator_graph_mode():
    """Test that Interpolator works in TensorFlow graph mode."""
    input_rate = 15.36e6
    output_rate = 122.88e6
    interp = Interpolator(input_rate=input_rate, output_rate=output_rate)

    @tf.function
    def interpolate_fn(x):
        y, rate = interp(x)
        return y

    x = tf.complex(
        tf.random.normal([2, 100]),
        tf.random.normal([2, 100]),
    )

    y = interpolate_fn(x)

    assert y.shape == (2, 800)

    print("\n[Interpolator Graph Mode]:")
    print("  Successfully ran in tf.function")
    print("  output shape: {y.shape}")
