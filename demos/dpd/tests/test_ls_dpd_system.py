# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""Tests for LS_DPDSystem using Least-Squares Indirect Learning Architecture."""

import pytest
import tensorflow as tf
from demos.dpd.src.config import Config
from demos.dpd.src.ls_dpd_system import LS_DPDSystem


def test_ls_dpd_system_initialization_inference():
    """Test LS_DPDSystem initialization in inference mode."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=False, config=cfg)

    # Verify components are initialized
    assert system._dpd is not None
    assert system._pa is not None
    assert system._tx is not None
    assert system._interpolator is not None
    assert system._rx is not None  # Available in inference mode

    print("\n[LS_DPDSystem Initialization (Inference)]:")
    print(f"  DPD order: {system._dpd._order}")
    print(f"  DPD memory_depth: {system._dpd._memory_depth}")


def test_ls_dpd_system_initialization_training():
    """Test LS_DPDSystem initialization in training mode."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=True, config=cfg)

    # Verify components
    assert system._dpd is not None
    assert system._rx is None  # Not available in training mode

    print("\n[LS_DPDSystem Initialization (Training)]:")
    print(f"  DPD n_coeffs: {system._dpd.n_coeffs}")


def test_ls_dpd_system_custom_dpd_params():
    """Test LS_DPDSystem with custom DPD parameters."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(
        training=True,
        config=cfg,
        dpd_order=5,
        dpd_memory_depth=3,
        ls_nIterations=5,
        ls_learning_rate=0.5,
        ls_learning_method="ema",
    )

    assert system._dpd._order == 5
    assert system._dpd._memory_depth == 3
    assert system._dpd._nIterations == 5
    assert system._dpd._learning_rate == 0.5
    assert system._dpd._learning_method == "ema"

    print("\n[LS_DPDSystem Custom DPD Params]:")
    print(f"  order: {system._dpd._order}")
    print(f"  memory_depth: {system._dpd._memory_depth}")
    print(f"  learning_method: {system._dpd._learning_method}")


def test_ls_dpd_system_generate_signal():
    """Test signal generation."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=True, config=cfg)

    signal = system.generate_signal(batch_size=4)

    # Signal should be 2D: [batch, samples]
    assert len(signal.shape) == 2
    assert signal.shape[0] == 4
    assert signal.dtype == tf.complex64

    print("\n[LS_DPDSystem Generate Signal]:")
    print(f"  signal shape: {signal.shape}")


def test_ls_dpd_system_generate_signal_with_extras():
    """Test signal generation with extra outputs."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=False, config=cfg)

    result = system.generate_signal(batch_size=4, return_extras=True)

    assert "tx_upsampled" in result
    assert "tx_baseband" in result
    assert "x_rg" in result
    assert "fd_symbols" in result

    print("\n[LS_DPDSystem Generate Signal with Extras]:")
    print(f"  tx_upsampled shape: {result['tx_upsampled'].shape}")
    print(f"  tx_baseband shape: {result['tx_baseband'].shape}")
    print(f"  fd_symbols shape: {result['fd_symbols'].shape}")


def test_ls_dpd_system_estimate_pa_gain():
    """Test PA gain estimation."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=True, config=cfg)

    gain = system.estimate_pa_gain(num_samples=10000)

    assert gain > 0
    assert system._pa_gain_initialized is True

    print("\n[LS_DPDSystem PA Gain Estimation]:")
    print(f"  estimated gain: {gain:.4f}")


def test_ls_dpd_system_inference_forward():
    """Test inference forward pass."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=False, config=cfg)

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

    print("\n[LS_DPDSystem Inference Forward]:")
    print(f"  pa_input shape: {result['pa_input'].shape}")
    print(f"  pa_output_no_dpd shape: {result['pa_output_no_dpd'].shape}")
    print(f"  pa_output_with_dpd shape: {result['pa_output_with_dpd'].shape}")


def test_ls_dpd_system_training_forward_raises():
    """Test that training forward raises for LS-DPD."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=True, config=cfg)

    with pytest.raises(ValueError, match="Use perform_ls_learning"):
        system(batch_size_or_signal=4)


def test_ls_dpd_system_ls_training_iteration():
    """Test single LS training iteration."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=True, config=cfg)

    # Estimate PA gain first
    system.estimate_pa_gain()

    # Generate signal
    x = system.generate_signal(batch_size=4)

    # Run one iteration
    result = system._ls_training_iteration(x)

    assert "y_power" in result
    assert isinstance(result["y_power"], float)

    print("\n[LS_DPDSystem LS Training Iteration]:")
    print(f"  y_power: {result['y_power']:.2f} dB")


def test_ls_dpd_system_perform_ls_learning():
    """Test full LS learning procedure."""
    cfg = Config(batch_size=10)
    system = LS_DPDSystem(
        training=True,
        config=cfg,
        dpd_order=5,
        dpd_memory_depth=3,
        ls_nIterations=2,
    )

    # Estimate PA gain first
    system.estimate_pa_gain()

    # Build the DPD layer by running a forward pass
    x = system.generate_signal(batch_size=10)
    _ = system._dpd(x)  # This builds the layer

    # Perform learning
    result = system.perform_ls_learning(batch_size=10, verbose=False)

    assert "coeffs" in result
    assert "coeff_history" in result

    # Coeffs should have changed from initial
    # (at least some should be non-zero beyond first)
    coeffs = result["coeffs"]
    assert coeffs.shape[0] == system._dpd.n_coeffs

    print("\n[LS_DPDSystem LS Learning]:")
    print("  n_iterations: 2")
    print(f"  final coeffs shape: {coeffs.shape}")
    print(f"  coeff_history shape: {result['coeff_history'].shape}")


def test_ls_dpd_system_perform_ls_learning_custom_iterations():
    """Test LS learning with custom iteration count."""
    cfg = Config(batch_size=10)
    system = LS_DPDSystem(training=True, config=cfg, ls_nIterations=3)

    system.estimate_pa_gain()

    # Build the DPD layer by running a forward pass
    x = system.generate_signal(batch_size=10)
    _ = system._dpd(x)  # This builds the layer

    # Override with custom iteration count
    result = system.perform_ls_learning(batch_size=10, nIterations=5, verbose=False)

    # Coeff history should have initial + 5 iterations = 6 columns
    assert result["coeff_history"].shape[1] == 6

    print("\n[LS_DPDSystem Custom Iterations]:")
    print(f"  coeff_history shape: {result['coeff_history'].shape}")


def test_ls_dpd_system_normalize_to_rms():
    """Test RMS normalization."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=True, config=cfg)

    # Create test signal
    x = tf.complex(
        tf.random.normal([4, 1000]),
        tf.random.normal([4, 1000]),
    )

    target_rms_dbm = 0.0
    x_norm, scale = system.normalize_to_rms(x, target_rms_dbm)

    # Verify output shape matches input
    assert x_norm.shape == x.shape

    print("\n[LS_DPDSystem Normalize to RMS]:")
    print(f"  target RMS: {target_rms_dbm} dBm")
    print(f"  scale factor: {float(scale):.4f}")


def test_ls_dpd_system_forward_signal_path():
    """Test forward signal path through DPD and PA."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=True, config=cfg)

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

    print("\n[LS_DPDSystem Forward Signal Path]:")
    print(f"  u shape: {signals['u'].shape}")
    print(f"  y_comp shape: {signals['y_comp'].shape}")


def test_ls_dpd_system_properties():
    """Test system property accessors."""
    cfg = Config(batch_size=4)
    system = LS_DPDSystem(training=False, config=cfg)

    # Test properties
    assert system.dpd is not None
    assert system.minimal_ofdm_receiver is not None
    assert system.signal_fs == cfg.signal_sample_rate
    assert system.pa_sample_rate == 122.88e6
    assert system.fft_size == cfg.fft_size
    assert system.cp_length == cfg.cyclic_prefix_length

    print("\n[LS_DPDSystem Properties]:")
    print(f"  signal_fs: {system.signal_fs / 1e6:.2f} MHz")
    print(f"  pa_sample_rate: {system.pa_sample_rate / 1e6:.2f} MHz")
    print(f"  fft_size: {system.fft_size}")


@pytest.mark.parametrize("learning_method", ["newton", "ema"])
def test_ls_dpd_system_learning_methods(learning_method):
    """Test both LS learning methods."""
    cfg = Config(batch_size=10)
    system = LS_DPDSystem(
        training=True,
        config=cfg,
        ls_learning_method=learning_method,
        ls_nIterations=2,
    )

    system.estimate_pa_gain()

    # Build the DPD layer by running a forward pass
    x = system.generate_signal(batch_size=10)
    _ = system._dpd(x)  # This builds the layer

    result = system.perform_ls_learning(batch_size=10, verbose=False)

    assert result["coeffs"] is not None

    print(f"\n[LS_DPDSystem Learning Method: {learning_method}]:")
    print("  Successfully completed learning")
