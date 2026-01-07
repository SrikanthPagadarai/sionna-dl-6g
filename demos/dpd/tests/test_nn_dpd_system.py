# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""Tests for NN_DPDSystem using Neural Network Indirect Learning Architecture."""

import pytest
import tensorflow as tf
from demos.dpd.src.config import Config
from demos.dpd.src.nn_dpd_system import NN_DPDSystem


def test_nn_dpd_system_initialization_inference():
    """Test NN_DPDSystem initialization in inference mode."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=False, config=cfg)

    # Verify components are initialized
    assert system._dpd is not None
    assert system._pa is not None
    assert system._tx is not None
    assert system._interpolator is not None
    assert system._rx is not None  # Available in inference mode

    print("\n[NN_DPDSystem Initialization (Inference)]:")
    print(f"  DPD memory_depth: {system._dpd._memory_depth}")
    print(f"  DPD num_filters: {system._dpd._num_filters}")


def test_nn_dpd_system_initialization_training():
    """Test NN_DPDSystem initialization in training mode."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    # Verify components
    assert system._dpd is not None
    assert system._rx is None  # Not available in training mode

    print("\n[NN_DPDSystem Initialization (Training)]:")
    print(f"  DPD num_res_blocks: {system._dpd._num_res_blocks}")


def test_nn_dpd_system_custom_dpd_params():
    """Test NN_DPDSystem with custom DPD parameters."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(
        training=True,
        config=cfg,
        dpd_memory_depth=8,
        dpd_num_filters=128,
        dpd_num_layers_per_block=3,
        dpd_num_res_blocks=4,
    )

    assert system._dpd._memory_depth == 8
    assert system._dpd._num_filters == 128
    assert system._dpd._num_layers_per_block == 3
    assert system._dpd._num_res_blocks == 4

    print("\n[NN_DPDSystem Custom DPD Params]:")
    print(f"  memory_depth: {system._dpd._memory_depth}")
    print(f"  num_filters: {system._dpd._num_filters}")
    print(f"  num_res_blocks: {system._dpd._num_res_blocks}")


def test_nn_dpd_system_generate_signal():
    """Test signal generation."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    signal = system.generate_signal(batch_size=4)

    # Signal should be 2D: [batch, samples]
    assert len(signal.shape) == 2
    assert signal.shape[0] == 4
    assert signal.dtype == tf.complex64

    print("\n[NN_DPDSystem Generate Signal]:")
    print(f"  signal shape: {signal.shape}")


def test_nn_dpd_system_estimate_pa_gain():
    """Test PA gain estimation."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    gain = system.estimate_pa_gain(num_samples=10000)

    assert gain > 0
    assert system._pa_gain_initialized is True

    print("\n[NN_DPDSystem PA Gain Estimation]:")
    print(f"  estimated gain: {gain:.4f}")


def test_nn_dpd_system_training_forward():
    """Test training forward pass returns loss."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    # Estimate PA gain first
    system.estimate_pa_gain()

    # Run training forward pass (use batch_size_or_signal keyword for Keras layer)
    loss = system(batch_size_or_signal=4)

    # Should return scalar loss
    assert loss.shape == ()
    assert not tf.math.is_nan(loss)
    assert loss >= 0

    print("\n[NN_DPDSystem Training Forward]:")
    print(f"  loss: {float(loss):.6f}")


def test_nn_dpd_system_inference_forward():
    """Test inference forward pass."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=False, config=cfg)

    # Estimate PA gain first
    system.estimate_pa_gain()

    # Run inference (use batch_size_or_signal keyword for Keras layer)
    result = system(batch_size_or_signal=4)

    assert "pa_input" in result
    assert "pa_output_no_dpd" in result
    assert "pa_output_with_dpd" in result
    assert "predistorted" in result

    # All should have same shape
    assert result["pa_input"].shape == result["pa_output_no_dpd"].shape
    assert result["pa_input"].shape == result["pa_output_with_dpd"].shape

    print("\n[NN_DPDSystem Inference Forward]:")
    print(f"  pa_input shape: {result['pa_input'].shape}")
    print(f"  pa_output_no_dpd shape: {result['pa_output_no_dpd'].shape}")
    print(f"  pa_output_with_dpd shape: {result['pa_output_with_dpd'].shape}")


def test_nn_dpd_system_trainable_variables():
    """Test that system has trainable variables for optimization."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    # Estimate PA gain and run forward to build model
    system.estimate_pa_gain()
    _ = system(batch_size_or_signal=4)

    trainable_vars = system.trainable_variables

    # Should have trainable variables from the NN-DPD
    assert len(trainable_vars) > 0

    # Count parameters
    total_params = sum(tf.reduce_prod(v.shape) for v in trainable_vars)

    print("\n[NN_DPDSystem Trainable Variables]:")
    print(f"  num variables: {len(trainable_vars)}")
    print(f"  total parameters: {int(total_params)}")


def test_nn_dpd_system_gradient_flow():
    """Test that gradients flow through the system."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    system.estimate_pa_gain()

    with tf.GradientTape() as tape:
        loss = system(batch_size_or_signal=4)

    grads = tape.gradient(loss, system.trainable_variables)

    # All gradients should be non-None
    non_none_grads = [g for g in grads if g is not None]
    assert len(non_none_grads) > 0

    print("\n[NN_DPDSystem Gradient Flow]:")
    print(f"  loss: {float(loss):.6f}")
    print(f"  non-None gradients: {len(non_none_grads)} / {len(grads)}")


def test_nn_dpd_system_forward_signal_path():
    """Test forward signal path through DPD and PA."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    system.estimate_pa_gain()

    x = system.generate_signal(batch_size=4)
    signals = system._forward_signal_path(x)

    assert "u" in signals
    assert "u_norm" in signals
    assert "y_comp" in signals
    assert "x_scale" in signals

    # Shapes should match
    assert signals["u"].shape == x.shape
    assert signals["y_comp"].shape == x.shape

    print("\n[NN_DPDSystem Forward Signal Path]:")
    print(f"  u shape: {signals['u'].shape}")
    print(f"  y_comp shape: {signals['y_comp'].shape}")
    print(f"  x_scale: {float(signals['x_scale']):.4f}")


def test_nn_dpd_system_normalize_to_unit_power():
    """Test unit power normalization."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    # Create test signal
    x = tf.complex(
        tf.random.normal([4, 1000]),
        tf.random.normal([4, 1000]),
    )

    x_norm, scale = system._normalize_to_unit_power(x)

    # Normalized signal should have approximately unit power
    power = tf.reduce_mean(tf.abs(x_norm) ** 2)
    assert abs(float(power) - 1.0) < 0.1, f"Power should be ~1, got {float(power)}"

    print("\n[NN_DPDSystem Normalize to Unit Power]:")
    print(f"  output power: {float(power):.4f}")
    print(f"  scale factor: {float(scale):.4f}")


def test_nn_dpd_system_loss_scaling():
    """Test that loss scaling is applied."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    system.estimate_pa_gain()

    # The loss should be scaled by _loss_scale (1000.0)
    loss = system(batch_size_or_signal=4)

    # Loss should be reasonable magnitude (scaled up)
    assert float(loss) > 0
    assert float(loss) < 1e6  # Upper bound check

    print("\n[NN_DPDSystem Loss Scaling]:")
    print(f"  loss: {float(loss):.4f}")
    print(f"  loss_scale: {system._loss_scale}")


def test_nn_dpd_system_properties():
    """Test system property accessors."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=False, config=cfg)

    # Test properties
    assert system.dpd is not None
    assert system.minimal_ofdm_receiver is not None
    assert system.signal_fs == cfg.signal_sample_rate
    assert system.pa_sample_rate == 122.88e6
    assert system.fft_size == cfg.fft_size
    assert system.cp_length == cfg.cyclic_prefix_length

    print("\n[NN_DPDSystem Properties]:")
    print(f"  signal_fs: {system.signal_fs / 1e6:.2f} MHz")
    print(f"  pa_sample_rate: {system.pa_sample_rate / 1e6:.2f} MHz")
    print(f"  fft_size: {system.fft_size}")


def test_nn_dpd_system_with_pre_generated_signal():
    """Test system with pre-generated signal input."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    system.estimate_pa_gain()

    # Generate signal first
    x = system.generate_signal(batch_size=4)

    # Use pre-generated signal
    loss = system(x, training=True)

    assert loss.shape == ()
    assert not tf.math.is_nan(loss)

    print("\n[NN_DPDSystem Pre-generated Signal]:")
    print(f"  input signal shape: {x.shape}")
    print(f"  loss: {float(loss):.6f}")


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_nn_dpd_system_various_batch_sizes(batch_size):
    """Test system with various batch sizes."""
    cfg = Config(batch_size=batch_size)
    system = NN_DPDSystem(training=True, config=cfg)

    system.estimate_pa_gain()
    loss = system(batch_size_or_signal=batch_size)

    assert not tf.math.is_nan(loss)

    print(f"\n[NN_DPDSystem Batch Size {batch_size}]:")
    print(f"  loss: {float(loss):.6f}")


def test_nn_dpd_system_training_step():
    """Test a complete training step with optimizer."""
    cfg = Config(batch_size=4)
    system = NN_DPDSystem(training=True, config=cfg)

    system.estimate_pa_gain()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Initial loss
    initial_loss = system(batch_size_or_signal=4)

    # Training step
    with tf.GradientTape() as tape:
        loss = system(batch_size_or_signal=4)

    grads = tape.gradient(loss, system.trainable_variables)
    optimizer.apply_gradients(zip(grads, system.trainable_variables))

    print("\n[NN_DPDSystem Training Step]:")
    print(f"  initial loss: {float(initial_loss):.6f}")
    print(f"  loss after step: {float(loss):.6f}")
    print(f"  gradients applied to {len(grads)} variables")
