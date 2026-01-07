# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""Tests for LeastSquaresDPD using Indirect Learning Architecture."""

import pytest
import tensorflow as tf
from demos.dpd.src.ls_dpd import LeastSquaresDPD


def test_ls_dpd_initialization_defaults():
    """Test LeastSquaresDPD initialization with default parameters."""
    dpd = LeastSquaresDPD()

    assert dpd._order == 7
    assert dpd._memory_depth == 4
    assert dpd._lag_depth == 0
    assert dpd._nIterations == 3
    assert dpd._learning_rate == 0.75
    assert dpd._learning_method == "newton"
    assert dpd._use_even is False
    assert dpd._use_conj is False
    assert dpd._use_dc_term is False

    print("\n[LS-DPD Initialization Defaults]:")
    print(f"  order: {dpd._order}")
    print(f"  memory_depth: {dpd._memory_depth}")
    print(f"  n_coeffs: {dpd.n_coeffs}")


def test_ls_dpd_initialization_custom():
    """Test LeastSquaresDPD initialization with custom parameters."""
    params = {
        "order": 5,
        "memory_depth": 3,
        "nIterations": 5,
        "learning_rate": 0.5,
        "learning_method": "ema",
    }
    dpd = LeastSquaresDPD(params=params)

    assert dpd._order == 5
    assert dpd._memory_depth == 3
    assert dpd._nIterations == 5
    assert dpd._learning_rate == 0.5
    assert dpd._learning_method == "ema"

    print("\n[LS-DPD Initialization Custom]:")
    print(f"  order: {dpd._order}")
    print(f"  memory_depth: {dpd._memory_depth}")
    print(f"  n_coeffs: {dpd.n_coeffs}")


def test_ls_dpd_even_order_raises():
    """Test that even order raises ValueError."""
    with pytest.raises(ValueError, match="Order of the DPD must be odd"):
        LeastSquaresDPD(params={"order": 6})


def test_ls_dpd_n_coeffs_calculation():
    """Test number of coefficients calculation."""
    # Default: order=7, memory=4, no even, no conj, no dc
    # n_order = (7+1)//2 = 4
    # n = 4 * 4 = 16
    dpd = LeastSquaresDPD()
    assert dpd.n_coeffs == 16

    # With smaller order
    dpd2 = LeastSquaresDPD(params={"order": 5, "memory_depth": 2})
    # n_order = (5+1)//2 = 3, n = 3 * 2 = 6
    assert dpd2.n_coeffs == 6

    print("\n[LS-DPD Coefficient Count]:")
    print(f"  order=7, memory=4: {dpd.n_coeffs} coeffs")
    print(f"  order=5, memory=2: {dpd2.n_coeffs} coeffs")


def test_ls_dpd_build_and_coeffs():
    """Test layer build and coefficient access."""
    dpd = LeastSquaresDPD()

    # Build the layer
    dpd.build(input_shape=(1000,))

    # Check coefficients
    coeffs = dpd.coeffs
    assert coeffs.shape == (dpd.n_coeffs, 1)
    assert coeffs.dtype == tf.complex64

    # First coefficient should be 1+0j (identity initialization)
    assert tf.abs(coeffs[0, 0] - tf.constant(1.0 + 0j, dtype=tf.complex64)) < 1e-6

    print("\n[LS-DPD Coefficients]:")
    print(f"  shape: {coeffs.shape}")
    print(f"  first coeff: {coeffs[0, 0].numpy()}")


def test_ls_dpd_coeffs_setter():
    """Test coefficient setter."""
    dpd = LeastSquaresDPD()
    dpd.build(input_shape=(1000,))

    # Set new coefficients
    new_coeffs = tf.complex(
        tf.random.normal([dpd.n_coeffs, 1]),
        tf.random.normal([dpd.n_coeffs, 1]),
    )
    dpd.coeffs = new_coeffs

    # Verify they were set
    diff = tf.reduce_max(tf.abs(dpd.coeffs - new_coeffs))
    assert diff < 1e-6

    print("\n[LS-DPD Coefficient Setter]:")
    print(f"  Successfully set {dpd.n_coeffs} coefficients")


def test_ls_dpd_basis_matrix_shape():
    """Test GMP basis matrix shape."""
    dpd = LeastSquaresDPD()

    num_samples = 1000
    x = tf.complex(
        tf.random.normal([num_samples]),
        tf.random.normal([num_samples]),
    )

    X = dpd.setup_basis_matrix(x)

    assert X.shape == (num_samples, dpd.n_coeffs)
    assert X.dtype == tf.complex64

    print("\n[LS-DPD Basis Matrix]:")
    print(f"  input samples: {num_samples}")
    print(f"  matrix shape: {X.shape}")


def test_ls_dpd_predistort_1d():
    """Test predistortion with 1D input."""
    dpd = LeastSquaresDPD()

    num_samples = 1000
    x = tf.complex(
        tf.random.normal([num_samples], stddev=0.1),
        tf.random.normal([num_samples], stddev=0.1),
    )

    y = dpd.predistort(x)

    assert y.shape == x.shape
    assert y.dtype == tf.complex64

    # With identity initialization, output should be close to input
    mse = tf.reduce_mean(tf.abs(y - x) ** 2)
    assert mse < 0.1, f"Initial predistortion should be near-identity, MSE={float(mse)}"

    print("\n[LS-DPD Predistort 1D]:")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {y.shape}")
    print(f"  MSE from input: {float(mse):.6f}")


def test_ls_dpd_predistort_2d_batched():
    """Test predistortion with 2D batched input."""
    dpd = LeastSquaresDPD()

    batch_size = 4
    num_samples = 500
    x = tf.complex(
        tf.random.normal([batch_size, num_samples], stddev=0.1),
        tf.random.normal([batch_size, num_samples], stddev=0.1),
    )

    y = dpd.predistort(x)

    assert y.shape == x.shape
    assert y.shape[0] == batch_size
    assert y.shape[1] == num_samples

    print("\n[LS-DPD Predistort 2D Batched]:")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {y.shape}")


def test_ls_dpd_call():
    """Test Keras layer call interface."""
    dpd = LeastSquaresDPD()

    x = tf.complex(
        tf.random.normal([100], stddev=0.1),
        tf.random.normal([100], stddev=0.1),
    )

    # Call should work like predistort
    y = dpd(x)
    y_pred = dpd.predistort(x)

    diff = tf.reduce_max(tf.abs(y - y_pred))
    assert diff < 1e-6

    print("\n[LS-DPD Call Interface]:")
    print(f"  call() matches predistort(): diff={float(diff):.2e}")


def test_ls_dpd_ls_estimation():
    """Test least-squares estimation."""
    dpd = LeastSquaresDPD()
    dpd.build(input_shape=(1000,))

    num_samples = 1000
    x = tf.complex(
        tf.random.normal([num_samples], stddev=0.1),
        tf.random.normal([num_samples], stddev=0.1),
    )

    # Build basis matrix
    X = dpd.setup_basis_matrix(x)

    # Create target signal
    y = tf.complex(
        tf.random.normal([num_samples], stddev=0.1),
        tf.random.normal([num_samples], stddev=0.1),
    )

    # Estimate coefficients
    coeffs = dpd._ls_estimation(X, y)

    assert coeffs.shape == (dpd.n_coeffs, 1)
    assert coeffs.dtype == tf.complex64

    print("\n[LS-DPD LS Estimation]:")
    print(f"  coefficient shape: {coeffs.shape}")


@pytest.mark.parametrize("order", [3, 5, 7, 9])
def test_ls_dpd_various_orders(order):
    """Test DPD with various polynomial orders."""
    dpd = LeastSquaresDPD(params={"order": order, "memory_depth": 4})

    x = tf.complex(
        tf.random.normal([500], stddev=0.1),
        tf.random.normal([500], stddev=0.1),
    )

    y = dpd.predistort(x)

    assert y.shape == x.shape

    print(f"\n[LS-DPD Order {order}]:")
    print(f"  n_coeffs: {dpd.n_coeffs}")
    print(f"  output shape: {y.shape}")


def test_ls_dpd_differentiable():
    """Test that predistortion is differentiable."""
    dpd = LeastSquaresDPD()
    dpd.build(input_shape=(100,))

    x = tf.complex(
        tf.random.normal([100], stddev=0.1),
        tf.random.normal([100], stddev=0.1),
    )

    with tf.GradientTape() as tape:
        y = dpd.predistort(x)
        # Loss based on output power
        loss = tf.reduce_mean(tf.abs(y) ** 2)

    # Should be able to compute gradients w.r.t. coefficients
    grads = tape.gradient(loss, dpd.trainable_variables)
    assert len(grads) == 2  # real and imag parts
    assert grads[0] is not None
    assert grads[1] is not None

    print("\n[LS-DPD Differentiability]:")
    print(f"  loss: {float(loss):.6f}")
    print(f"  gradient shapes: {[g.shape for g in grads]}")
