# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

import pytest
import tensorflow as tf
from sionna.phy.utils import compute_ber
from demos.pusch_autoencoder.src.system import PUSCHLinkE2E


def create_dummy_cir(batch_size, num_rx_ant, num_tx_ant, num_paths=10):
    """Create dummy CIR (channel impulse response) for testing.

    Returns:
        tuple: (a, tau) where
            a: channel coefficients
            [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
            tau: delays
            [batch, num_tx, num_paths]
    """
    num_rx = 1
    num_tx = 4

    a_shape = (batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths)
    a = tf.complex(
        tf.random.normal(a_shape, dtype=tf.float32),
        tf.random.normal(a_shape, dtype=tf.float32),
    )

    tau_shape = (batch_size, num_tx, num_paths)
    # Delays should be positive and relatively small
    tau = tf.abs(tf.random.normal(tau_shape, dtype=tf.float32)) * 1e-6

    return a, tau


@pytest.mark.parametrize("perfect_csi", [True, False])
@pytest.mark.parametrize("use_autoencoder", [True, False])
def test_system_initialization(perfect_csi, use_autoencoder):
    """Test PUSCHLinkE2E initialization."""
    # Create dummy CIR data
    batch_size = 100
    a, tau = create_dummy_cir(batch_size, num_rx_ant=16, num_tx_ant=4)

    system = PUSCHLinkE2E(
        channel_model=(a, tau),
        perfect_csi=perfect_csi,
        use_autoencoder=use_autoencoder,
        training=False,
    )

    # Check basic attributes
    assert system._perfect_csi == perfect_csi
    assert system._use_autoencoder == use_autoencoder
    assert not system._training

    # Check components exist
    assert hasattr(system, "_pusch_transmitter")
    assert hasattr(system, "_pusch_receiver")
    assert hasattr(system, "_channel")

    print(f"\n[System Init] CSI={perfect_csi}, Autoencoder={use_autoencoder}:")
    print(f"  Transmitter: {type(system._pusch_transmitter).__name__}")
    print(f"  Receiver: {type(system._pusch_receiver).__name__}")
    print(f"  Detector: {type(system._detector).__name__}")


@pytest.mark.skip(reason="Requires proper channel model integration with real CIR data")
@pytest.mark.parametrize("perfect_csi", [True, False])
@pytest.mark.parametrize("use_autoencoder", [True, False])
def test_system_inference(perfect_csi, use_autoencoder):
    """Test PUSCHLinkE2E inference mode."""
    batch_size = 100
    a, tau = create_dummy_cir(batch_size, num_rx_ant=16, num_tx_ant=4)

    system = PUSCHLinkE2E(
        channel_model=(a, tau),
        perfect_csi=perfect_csi,
        use_autoencoder=use_autoencoder,
        training=False,
    )

    test_batch_size = tf.constant(8, tf.int32)  # Use tensor
    ebno_db = tf.constant(20.0, tf.float32)

    # Run inference
    b, b_hat = system(test_batch_size, ebno_db)

    # Compute BER
    ber = compute_ber(b, b_hat)

    print(
        "\n[System Inference] "
        f"CSI={perfect_csi}, "
        f"AE={use_autoencoder}, "
        f"Eb/N0={float(ebno_db)} dB:"
    )
    print(f"  b shape: {b.shape}")
    print(f"  b_hat shape: {b_hat.shape}")
    print(f"  BER: {float(ber):.6f}")

    # Check output shapes
    assert b.shape == b_hat.shape
    assert b.shape[0] == 8  # test_batch_size


@pytest.mark.skip(reason="Requires proper channel model integration with real CIR data")
@pytest.mark.parametrize("perfect_csi", [True, False])
def test_system_training_mode(perfect_csi):
    """Test PUSCHLinkE2E training mode (autoencoder only)."""
    batch_size = 100
    a, tau = create_dummy_cir(batch_size, num_rx_ant=16, num_tx_ant=4)

    system = PUSCHLinkE2E(
        channel_model=(a, tau),
        perfect_csi=perfect_csi,
        use_autoencoder=True,  # Training only works with autoencoder
        training=True,
    )

    test_batch_size = tf.constant(8, tf.int32)  # Use tensor
    ebno_db = tf.constant(15.0, tf.float32)

    # Run in training mode (returns loss)
    loss = system(test_batch_size, ebno_db)

    print(f"\n[System Training] CSI={perfect_csi}, Eb/N0={float(ebno_db)}dB:")
    print(f"  Loss: {float(loss):.6f}")

    # Check that loss is a scalar
    assert loss.shape == ()
    assert not tf.math.is_nan(loss)
    assert not tf.math.is_inf(loss)


@pytest.mark.skip(reason="Requires proper channel model, not dummy CIR tuple")
def test_system_trainable_variables_baseline():
    """Test trainable variables for baseline system (should be none)."""
    batch_size = 100
    a, tau = create_dummy_cir(batch_size, num_rx_ant=16, num_tx_ant=4)

    system = PUSCHLinkE2E(
        channel_model=(a, tau),
        perfect_csi=True,
        use_autoencoder=False,  # Baseline has no trainable vars
        training=False,
    )

    trainable_vars = system.trainable_variables

    print("\n[Baseline System Trainable Vars]:")
    print(f"  Number of trainable variables: {len(trainable_vars)}")

    # Baseline should have empty list (property returns empty list)
    assert len(trainable_vars) == 0


def test_system_trainable_variables_autoencoder():
    """Test trainable variables for autoencoder system."""
    batch_size = 100
    a, tau = create_dummy_cir(batch_size, num_rx_ant=16, num_tx_ant=4)

    system = PUSCHLinkE2E(
        channel_model=(a, tau),
        perfect_csi=True,
        use_autoencoder=True,
        training=True,
    )

    trainable_vars = system.trainable_variables

    print("\n[Autoencoder System Trainable Vars]:")
    print(f"  Number of trainable variables: {len(trainable_vars)}")

    # Autoencoder should have trainable variables
    # (constellation + neural detector)
    assert len(trainable_vars) > 0


def test_constellation_properties():
    """Test constellation property of the system."""
    batch_size = 100
    a, tau = create_dummy_cir(batch_size, num_rx_ant=16, num_tx_ant=4)

    system = PUSCHLinkE2E(
        channel_model=(a, tau),
        perfect_csi=True,
        use_autoencoder=True,
        training=True,
    )

    # Get constellation
    constellation = system.constellation

    print("\n[System Constellation]:")
    print(f"  Constellation shape: {constellation.shape}")
    print(f"  Constellation dtype: {constellation.dtype}")

    # Check unit power
    energy = tf.reduce_mean(tf.square(tf.abs(constellation)))
    print(f"  Average energy: {float(energy):.6f}")

    assert constellation.dtype == tf.complex64
    assert tf.abs(energy - 1.0) < 0.1


def test_constellation_min_distance():
    """Test constellation minimum distance calculation."""
    batch_size = 100
    a, tau = create_dummy_cir(batch_size, num_rx_ant=16, num_tx_ant=4)

    system = PUSCHLinkE2E(
        channel_model=(a, tau),
        perfect_csi=True,
        use_autoencoder=True,
        training=True,
    )

    # Get minimum distance
    min_dist = system.get_constellation_min_distance()

    print("\n[Constellation Min Distance]:")
    print(f"  Minimum distance: {float(min_dist):.6f}")

    assert min_dist > 0.0
    assert not tf.math.is_nan(min_dist)


@pytest.mark.skip(reason="Requires proper channel model integration with real CIR data")
@pytest.mark.parametrize("ebno_db", [5.0, 10.0, 20.0])
def test_system_different_snr(ebno_db):
    """Test system at different SNR levels."""
    batch_size = 100
    a, tau = create_dummy_cir(batch_size, num_rx_ant=16, num_tx_ant=4)

    system = PUSCHLinkE2E(
        channel_model=(a, tau), perfect_csi=True, use_autoencoder=False, training=False
    )

    test_batch_size = tf.constant(8, tf.int32)  # Use tensor
    ebno = tf.constant(ebno_db, tf.float32)

    # Run inference
    b, b_hat = system(test_batch_size, ebno)
    ber = compute_ber(b, b_hat)

    print(f"\n[System SNR Test] Eb/N0={ebno_db}dB:")
    print(f"  BER: {float(ber):.6f}")

    assert not tf.math.is_nan(ber)


@pytest.mark.skip(reason="Requires proper channel model integration with real CIR data")
def test_system_batch_size_variation():
    """Test system with different batch sizes."""
    cir_batch_size = 100
    a, tau = create_dummy_cir(cir_batch_size, num_rx_ant=16, num_tx_ant=4)

    system = PUSCHLinkE2E(
        channel_model=(a, tau), perfect_csi=True, use_autoencoder=True, training=False
    )

    ebno_db = tf.constant(15.0, tf.float32)

    for test_batch_size_val in [4, 8, 16]:
        test_batch_size = tf.constant(test_batch_size_val, tf.int32)  # Use tensor
        b, b_hat = system(test_batch_size, ebno_db)

        print(f"\n[Batch Size {test_batch_size_val}]:")
        print(f"  b shape: {b.shape}")
        print(f"  b_hat shape: {b.shape}")

        assert b.shape[0] == test_batch_size_val
        assert b_hat.shape[0] == test_batch_size_val
