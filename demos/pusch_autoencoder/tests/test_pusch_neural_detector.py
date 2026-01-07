# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import pytest
import tensorflow as tf
from sionna.phy.nr import PUSCHConfig
from demos.pusch_autoencoder.src.config import Config
from demos.pusch_autoencoder.src.pusch_neural_detector import (
    PUSCHNeuralDetector,
    Conv2DResBlock,
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


def test_conv2d_res_block():
    """Test Conv2DResBlock forward pass."""
    filters = 64
    kernel_size = (3, 3)
    batch_size = 4
    height, width = 14, 192
    channels = filters  # Must match filters for residual connection

    # Create block
    block = Conv2DResBlock(filters=filters, kernel_size=kernel_size)

    # Random input (channels must equal filters for residual add)
    x = tf.random.normal((batch_size, height, width, channels))

    # Forward pass
    y = block(x)

    print("\n[Conv2D ResBlock]:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Filters: {filters}")

    # Output should have same shape as input (residual connection)
    assert y.shape == x.shape


def test_neural_detector_initialization():
    """Test PUSCHNeuralDetector initialization."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    # Update config with PUSCH parameters
    cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
    cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
    cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

    detector = PUSCHNeuralDetector(
        cfg, num_conv2d_filters=128, num_shared_res_blocks=4, num_det_res_blocks=6
    )

    # Check attributes
    assert detector.num_conv2d_filters == 128
    assert detector.num_shared_res_blocks == 4
    assert detector.num_det_res_blocks == 6

    print("\n[Neural Detector Init]:")
    print(f"  Conv2D filters: {detector.num_conv2d_filters}")
    print(f"  Shared res blocks: {detector.num_shared_res_blocks}")
    print(f"  Det res blocks: {detector.num_det_res_blocks}")
    print(f"  Num streams total: {detector._num_streams_total}")
    print(f"  Num RX antennas: {detector._num_rx_ant}")


def rand_cplx(shape, dtype=tf.float32):
    """Generate random complex tensor."""
    return tf.complex(
        tf.random.normal(shape, dtype=dtype), tf.random.normal(shape, dtype=dtype)
    )


@pytest.mark.skip(reason="Forward pass needs proper channel model integration")
def test_neural_detector_forward():
    """Test PUSCHNeuralDetector forward pass with dummy inputs."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    # Update config
    cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
    cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
    cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

    detector = PUSCHNeuralDetector(cfg)

    batch_size = 4
    num_rx = cfg.num_bs
    num_rx_ant = cfg.num_bs_ant
    num_tx = cfg.num_ue
    num_streams_per_tx = cfg.num_layers
    num_ofdm_symbols = cfg.pusch_num_symbols_per_slot
    fft_size = cfg.pusch_num_subcarriers

    # Create dummy inputs
    # y: received signal [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
    y = rand_cplx((batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size))

    # h_hat: channel estimate
    # [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_syms, fft_size]
    h_hat = rand_cplx(
        (
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_streams_per_tx,
            num_ofdm_symbols,
            fft_size,
        )
    )

    # err_var: estimation error variance
    # [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_syms, fft_size]
    err_var = tf.abs(
        tf.random.normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                fft_size,
            )
        )
    )

    # no: noise variance (scalar)
    no = tf.constant(0.1, tf.float32)

    # Forward pass
    llr = detector(y, h_hat, err_var, no)

    print("\n[Neural Detector Forward]:")
    print(f"  y shape: {y.shape}")
    print(f"  h_hat shape: {h_hat.shape}")
    print(f"  err_var shape: {err_var.shape}")
    print(f"  llr shape: {llr.shape}")

    # LLR shape should be [batch, num_tx, num_streams_per_tx, n_coded_bits]
    assert llr.shape[0] == batch_size
    assert llr.shape[1] == num_tx
    assert llr.shape[2] == num_streams_per_tx
    assert len(llr.shape) == 4


@pytest.mark.skip(reason="Forward pass needs proper channel model integration")
def test_neural_detector_with_constellation():
    """Test PUSCHNeuralDetector forward pass with custom constellation."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    # Update config
    cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
    cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
    cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

    detector = PUSCHNeuralDetector(cfg)

    batch_size = 2
    num_rx = cfg.num_bs
    num_rx_ant = cfg.num_bs_ant
    num_tx = cfg.num_ue
    num_streams_per_tx = cfg.num_layers
    num_ofdm_symbols = cfg.pusch_num_symbols_per_slot
    fft_size = cfg.pusch_num_subcarriers

    # Create dummy inputs
    y = rand_cplx((batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size))
    h_hat = rand_cplx(
        (
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_streams_per_tx,
            num_ofdm_symbols,
            fft_size,
        )
    )
    err_var = tf.abs(
        tf.random.normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                fft_size,
            )
        )
    )
    no = tf.constant(0.1, tf.float32)

    # Custom constellation (random normalized points)
    num_points = 2**cfg.num_bits_per_symbol
    constellation = tf.complex(
        tf.random.normal((num_points,)), tf.random.normal((num_points,))
    )
    # Normalize
    constellation = constellation / tf.cast(
        tf.sqrt(tf.reduce_mean(tf.square(tf.abs(constellation)))), constellation.dtype
    )

    # Forward pass with constellation
    llr = detector(y, h_hat, err_var, no, constellation=constellation)

    print("\n[Neural Detector with Constellation]:")
    print(f"  Custom constellation shape: {constellation.shape}")
    print(f"  llr shape: {llr.shape}")

    assert llr.shape[0] == batch_size
    assert len(llr.shape) == 4


def test_neural_detector_trainable_variables():
    """Test that neural detector has trainable variables."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
    cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
    cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

    detector = PUSCHNeuralDetector(cfg)

    # Check that there are trainable variables
    trainable_vars = detector.trainable_variables
    assert len(trainable_vars) > 0

    print("\n[Neural Detector Trainable Vars]:")
    print(f"  Number of trainable variables: {len(trainable_vars)}")
    print(f"  First few variable shapes: {[v.shape for v in trainable_vars[:5]]}")
