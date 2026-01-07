# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
BER/BLER evaluation script for trained PUSCH autoencoder.

This script loads trained weights and evaluates the autoencoder's Block Error
Rate (BLER) and Bit Error Rate (BER) performance. Results can be compared
against the baseline (baseline.py) to quantify the improvement from neural
detection and learned constellation geometry.

The script runs inference for multiple BS antenna configurations (16 and 32).

Evaluation Pipeline
-------------------
1. Load trained weights from pickle file
2. Restore TX constellation and RX neural detector parameters
3. Run Monte Carlo simulation over Eb/N0 range
4. Save BER/BLER results for plotting

Weight Loading
--------------
The script expects weights saved by training.py in the format::

    results/PUSCH_autoencoder_weights_ant{num_bs_ant}

If weights are not found, the script continues with random initialization
(useful for debugging, but results will be meaningless).

Output
------
Results are saved to ``results/inference_results_ant{num_bs_ant}.npz`` containing:

- ``ebno_db``: Eb/N0 sweep values in dB
- ``ber``: Bit error rates
- ``bler``: Block error rates
- System configuration for reproducibility

Usage
-----
Run from the repository root after training::

    python -m demos.pusch_autoencoder.inference

Then use plots.py to compare against baseline results.
"""

import os
import sys
import tensorflow as tf

# get directory name of file
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# TensorFlow and GPU Configuration
# =============================================================================
# Must be done before importing other TF-dependent modules

# Default to GPU 0 if not specified (avoid multi-GPU complications)
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Suppress verbose TF logging during inference
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Enable dynamic GPU memory allocation
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # GPU context may already be initialized
        print(f"GPU configuration error: {e}", file=sys.stderr)

# Disable layout optimizer to prevent graph cycle errors with ConvLSTM layers.
# The Grappler layout optimizer can create invalid graph structures when
# optimizing recurrent layers, causing "Graph cycle detected" errors at runtime.
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

# Imports after TF configuration (noqa comments suppress linter warnings)
import pickle  # noqa: E402
import numpy as np  # noqa: E402
from sionna.phy.utils import PlotBER  # noqa: E402

from demos.pusch_autoencoder.src.config import Config  # noqa: E402
from demos.pusch_autoencoder.src.system import PUSCHLinkE2E  # noqa: E402
from demos.pusch_autoencoder.src.cir_manager import CIRManager  # noqa: E402


# =============================================================================
# Weight Loading Utility
# =============================================================================
def load_model_weights(
    model: tf.keras.Model, weights_path: str, batch_size: int
) -> bool:
    """
    Build model graph and restore trained weights from pickle file.

    This function performs three steps:

    1. **Build**: Run a forward pass to create all layer weights
    2. **Load**: Read weight arrays from pickle file
    3. **Assign**: Copy arrays to corresponding tf.Variable objects

    Parameters
    ----------
    model : PUSCHLinkE2E
        The end-to-end model to restore weights into.
    weights_path : str
        Path to pickle file containing weight dictionary.
    batch_size : int
        Batch size for the build forward pass.

    Returns
    -------
    bool
        ``True`` if weights were successfully loaded,
        ``False`` if weights file was not found (random init retained).

    Notes
    -----
    The weight dictionary structure (from training.py) is::

        {
            "tx_weights": [array, ...],  # Constellation real/imag
            "rx_weights": [array, ...],  # Scales + NN weights
            "tx_names": [str, ...],      # Variable names for debugging
            "rx_names": [str, ...],
            "normalized_constellation": array  # For reference
        }

    The constellation synchronization step ensures that the internal
    ``Constellation`` object used by the mapper/demapper has the same
    points as the trainable variables. Without this, the demapper would
    use stale points from construction time.
    """
    # Build model by running a forward pass (creates all tf.Variables)
    ebno_db_build = tf.fill([batch_size], 10.0)
    _ = model(tf.cast(batch_size, tf.int32), ebno_db_build)

    if not os.path.exists(weights_path):
        print(
            f"[WARN] Weights file not found at '{weights_path}'. "
            "Running inference with randomly initialized weights."
        )
        return False

    with open(weights_path, "rb") as f:
        weights_dict = pickle.load(f)

    # Restore transmitter weights (constellation real/imag parts)
    tx_vars = model._pusch_transmitter.trainable_variables
    for var, arr in zip(tx_vars, weights_dict["tx_weights"]):
        var.assign(arr)
    print(f"[INFO] Restored {len(tx_vars)} TX variables.")

    # Restore receiver weights (correction scales + NN weights)
    rx_vars = model._pusch_receiver.trainable_variables
    for var, arr in zip(rx_vars, weights_dict["rx_weights"]):
        var.assign(arr)
    print(f"[INFO] Restored {len(rx_vars)} RX variables.")

    # Synchronize constellation object with restored variable values.
    # The Mapper/Demapper use _constellation.points internally, which
    # was set at construction time and needs to be updated.
    tx = model._pusch_transmitter
    if hasattr(tx, "_points_r") and hasattr(tx, "_points_i"):
        pts = tf.complex(tx._points_r, tx._points_i)
        if hasattr(tx, "_constellation"):
            tx._constellation.points = pts
        print("[INFO] Synced trainable constellation points.")

    print(f"[INFO] Loaded weights from '{weights_path}'.")
    return True


# =============================================================================
# Parametrized BS Antenna Configurations
# =============================================================================
NUM_BS_ANT_VALUES = [16, 32]

for num_bs_ant in NUM_BS_ANT_VALUES:
    print(f"\n{'='*60}")
    print(f"Running inference with num_bs_ant = {num_bs_ant}")
    print(f"{'='*60}\n")

    # =========================================================================
    # System Configuration and Channel Model
    # =========================================================================
    _cfg = Config(num_bs_ant=num_bs_ant)
    batch_size = _cfg.batch_size
    num_ue = _cfg.num_ue
    num_ue_ant = _cfg.num_ue_ant
    num_time_steps = _cfg.num_time_steps

    cir_manager = CIRManager(config=_cfg)
    # Use same CIR loading as training.py for consistent channel distribution
    channel_model = cir_manager.load_from_tfrecord(group_for_mumimo=True)

    # =========================================================================
    # Model Instantiation
    # =========================================================================
    # Create model in inference mode (training=False returns decoded bits, not LLRs)
    e2e_model = PUSCHLinkE2E(
        channel_model,
        perfect_csi=False,
        use_autoencoder=True,
        training=False,  # Inference mode: full TB decoding for BER evaluation
        config=_cfg,
    )

    # Select weights file based on antenna configuration
    weights_filename = f"PUSCH_autoencoder_weights_ant{num_bs_ant}"
    weights_path = os.path.join(DEMO_DIR, "results", weights_filename)
    _ = load_model_weights(e2e_model, weights_path, batch_size)

    # =========================================================================
    # Quick Functional Check
    # =========================================================================
    # Verify the model runs correctly with loaded weights before long simulation
    ebno_db_test = tf.fill([batch_size], 10.0)  # Vector shape matches training
    b_test, b_hat_test = e2e_model(batch_size, ebno_db_test)
    print("Quick check shapes (autoencoder inference):", b_test.shape, b_hat_test.shape)

    # =========================================================================
    # PlotBER Adapter
    # =========================================================================
    def ae_model_for_ber(batch_size, ebno_db):
        """
        Adapter to match PlotBER's calling convention to PUSCHLinkE2E's interface.

        PlotBER.simulate() calls the model with scalar Eb/N0 values, but our
        PUSCHLinkE2E expects a vector of shape ``[batch_size]`` (matching
        training behavior where each sample can have different SNR).

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        ebno_db : tf.Tensor
            Eb/N0 value, either scalar or vector.

        Returns
        -------
        tuple of (b, b_hat)
            Ground-truth and decoded information bits.
        """
        ebno_db = tf.cast(ebno_db, tf.float32)

        # Expand scalar to vector if needed
        if ebno_db.shape.rank == 0:
            ebno_vec = tf.fill([batch_size], ebno_db)
        else:
            ebno_vec = ebno_db

        return e2e_model(batch_size, ebno_vec)

    # =========================================================================
    # BER/BLER Monte Carlo Simulation
    # =========================================================================
    # Same Eb/N0 range as baseline.py for direct comparison
    ebno_db = np.arange(-2, 10, 1)

    ber_plot = PlotBER(f"PUSCH Autoencoder Inference ({num_bs_ant} BS Antennas)")

    # Use wrapper function that handles scalar->vector Eb/N0 conversion
    ber, bler = ber_plot.simulate(
        ae_model_for_ber,
        ebno_dbs=ebno_db,
        max_mc_iter=500,
        num_target_block_errors=2000,
        batch_size=batch_size,
        soft_estimates=False,  # Hard-decision BER after LDPC decoding
        show_fig=False,
        add_bler=True,
    )

    # Convert to NumPy for storage
    if hasattr(ber, "numpy"):
        ber = ber.numpy()
    if hasattr(bler, "numpy"):
        bler = bler.numpy()

    # =========================================================================
    # Save Results
    # =========================================================================
    os.makedirs(os.path.join(DEMO_DIR, "results"), exist_ok=True)

    results_filename = f"inference_results_ant{num_bs_ant}.npz"
    out_path = os.path.join(DEMO_DIR, "results", results_filename)

    # Include all metadata for reproducibility and plotting
    np.savez(
        out_path,
        ebno_db=ebno_db,
        ber=ber,
        bler=bler,
        batch_size=batch_size,
        num_ue=num_ue,
        num_bs_ant=num_bs_ant,
        num_ue_ant=num_ue_ant,
        num_time_steps=num_time_steps,
        perfect_csi=False,
        use_autoencoder=True,
        training=False,
    )

    print(f"Saved autoencoder BER/BLER inference results to {out_path}")

print("\n" + "=" * 60)
print("Inference complete for all antenna configurations.")
print("=" * 60)
