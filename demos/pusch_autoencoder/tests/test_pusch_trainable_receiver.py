# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import pytest
import tensorflow as tf
from sionna.phy.nr import PUSCHConfig
from demos.pusch_autoencoder.src.config import Config
from demos.pusch_autoencoder.src.pusch_trainable_transmitter import (
    PUSCHTrainableTransmitter,
)
from demos.pusch_autoencoder.src.pusch_trainable_receiver import (
    PUSCHTrainableReceiver,
)
from demos.pusch_autoencoder.src.pusch_neural_detector import PUSCHNeuralDetector


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


def rand_cplx(shape, dtype=tf.float32):
    """Generate random complex tensor."""
    return tf.complex(
        tf.random.normal(shape, dtype=dtype), tf.random.normal(shape, dtype=dtype)
    )


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("perfect_csi", [True, False])
def test_trainable_receiver_initialization(training, perfect_csi):
    """Test PUSCHTrainableReceiver initialization."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    # Update config
    cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
    cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
    cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

    # Create transmitter
    tx = PUSCHTrainableTransmitter(
        [pusch_config], output_domain="freq", training=training
    )

    # Create detector
    detector = PUSCHNeuralDetector(cfg)

    # Create receiver
    receiver_kwargs = {
        "mimo_detector": detector,
        "input_domain": "freq",
        "pusch_transmitter": tx,
        "training": training,
    }

    if perfect_csi:
        receiver_kwargs["channel_estimator"] = "perfect"

    rx = PUSCHTrainableReceiver(**receiver_kwargs)

    # Check attributes
    assert rx._training == training
    assert rx._pusch_transmitter is tx

    print(f"\n[Trainable RX Init] Training={training}, Perfect CSI={perfect_csi}:")
    print(f"  Transmitter: {type(rx._pusch_transmitter).__name__}")
    print(f"  Detector: {type(rx._mimo_detector).__name__}")


@pytest.mark.skip(reason="Forward pass needs full system integration")
@pytest.mark.parametrize("training", [True, False])
def test_trainable_receiver_forward_perfect_csi(training):
    """Test PUSCHTrainableReceiver forward pass with perfect CSI."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    # Update config
    cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
    cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
    cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

    # Create transmitter
    tx = PUSCHTrainableTransmitter(
        [pusch_config], output_domain="freq", training=training
    )

    # Create detector
    detector = PUSCHNeuralDetector(cfg)

    # Create receiver with perfect CSI
    rx = PUSCHTrainableReceiver(
        mimo_detector=detector,
        input_domain="freq",
        pusch_transmitter=tx,
        training=training,
        channel_estimator="perfect",
    )

    batch_size = 4
    num_rx = cfg.num_bs
    num_rx_ant = cfg.num_bs_ant
    num_tx = cfg.num_ue
    num_streams_per_tx = cfg.num_layers
    num_ofdm_symbols = cfg.pusch_num_symbols_per_slot
    fft_size = cfg.pusch_num_subcarriers

    # Create dummy inputs
    y = rand_cplx((batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size))
    h = rand_cplx(
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
    no = tf.constant(0.1, tf.float32)

    # Forward pass
    output = rx(y, no, h)

    print(f"\n[Trainable RX Forward - Perfect CSI] Training={training}:")
    print(f"  y shape: {y.shape}")
    print(f"  h shape: {h.shape}")

    if training:
        # Training mode returns LLRs
        print(f"  LLR output shape: {output.shape}")
        assert len(output.shape) in [3, 4]  # Can be squeezed or not
    else:
        # Inference mode returns decoded bits
        print(f"  Decoded bits shape: {output.shape}")
        assert output.shape[0] == batch_size


@pytest.mark.skip(reason="Forward pass needs full system integration")
def test_trainable_receiver_forward_imperfect_csi():
    """Test PUSCHTrainableReceiver forward pass with imperfect CSI."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    # Update config
    cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
    cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
    cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

    # Create transmitter
    tx = PUSCHTrainableTransmitter([pusch_config], output_domain="freq", training=False)

    # Create detector
    detector = PUSCHNeuralDetector(cfg)

    # Create receiver without perfect CSI (uses channel estimator)
    rx = PUSCHTrainableReceiver(
        mimo_detector=detector,
        input_domain="freq",
        pusch_transmitter=tx,
        training=False,
    )

    batch_size = 4
    num_rx = cfg.num_bs
    num_rx_ant = cfg.num_bs_ant
    num_ofdm_symbols = cfg.pusch_num_symbols_per_slot
    fft_size = cfg.pusch_num_subcarriers

    # Create dummy input (no h needed for imperfect CSI)
    y = rand_cplx((batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size))
    no = tf.constant(0.1, tf.float32)

    # Forward pass (without h)
    output = rx(y, no)

    print("\n[Trainable RX Forward - Imperfect CSI]:")
    print(f"  y shape: {y.shape}")
    print(f"  Decoded bits shape: {output.shape}")

    assert output.shape[0] == batch_size


def test_trainable_receiver_get_normalized_constellation():
    """Test constellation retrieval from transmitter."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    # Update config
    cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
    cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
    cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

    # Create transmitter
    tx = PUSCHTrainableTransmitter([pusch_config], output_domain="freq", training=True)

    # Create detector
    detector = PUSCHNeuralDetector(cfg)

    # Create receiver
    rx = PUSCHTrainableReceiver(
        mimo_detector=detector,
        input_domain="freq",
        pusch_transmitter=tx,
        training=True,
        channel_estimator="perfect",
    )

    # Get constellation
    constellation = rx._get_normalized_constellation()

    print("\n[Trainable RX Constellation]:")
    print(f"  Constellation shape: {constellation.shape}")
    print(f"  Average energy: {tf.reduce_mean(tf.square(tf.abs(constellation))):.6f}")

    assert constellation is not None
    assert constellation.shape[0] == 2**cfg.num_bits_per_symbol


def test_trainable_receiver_trainable_variables():
    """Test that trainable variables from detector are exposed."""
    cfg = Config()
    pusch_config = get_pusch_config(cfg)

    # Update config
    cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
    cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
    cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

    # Create transmitter
    tx = PUSCHTrainableTransmitter([pusch_config], output_domain="freq", training=True)

    # Create detector
    detector = PUSCHNeuralDetector(cfg)

    # Create receiver
    rx = PUSCHTrainableReceiver(
        mimo_detector=detector,
        input_domain="freq",
        pusch_transmitter=tx,
        training=True,
        channel_estimator="perfect",
    )

    # Get trainable variables
    trainable_vars = rx.trainable_variables

    print("\n[Trainable RX Variables]:")
    print(f"  Number of trainable variables: {len(trainable_vars)}")

    # Should have detector's trainable variables
    assert len(trainable_vars) > 0
