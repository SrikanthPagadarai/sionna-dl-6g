# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Baseline BER/BLER simulation for classical LMMSE PUSCH receiver.

This script evaluates the Block Error Rate (BLER) and Bit Error Rate (BER)
performance of the standard LMMSE-based PUSCH receiver without neural network
enhancements. The results serve as a reference baseline for comparing against
the trained autoencoder system.

The simulation sweeps over a range of Eb/N0 values and measures performance
under two channel state information (CSI) conditions:

1. **Perfect CSI**: Ground-truth channel known at receiver (upper bound)
2. **Imperfect CSI**: LS channel estimation from DMRS (realistic scenario)

The gap between perfect and imperfect CSI performance indicates how much
potential improvement exists from better channel estimation or detection.

Output
------
Results are saved to ``results/baseline_results_ant{num_bs_ant}.npz`` containing:

- ``ebno_db``: Eb/N0 sweep values in dB
- ``ber``: Bit error rates, shape ``[2, num_snr_points]`` (perfect, imperfect)
- ``bler``: Block error rates, shape ``[2, num_snr_points]``
- System configuration parameters for reproducibility

Usage
-----
Run from the repository root::

    python -m demos.pusch_autoencoder.baseline
    python -m demos.pusch_autoencoder.baseline --num_bs_ant 32
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sionna.phy.utils import PlotBER

from demos.pusch_autoencoder.src.config import Config
from demos.pusch_autoencoder.src.system import PUSCHLinkE2E
from demos.pusch_autoencoder.src.cir_manager import CIRManager

# get directory name of file
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# TensorFlow and GPU Configuration
# =============================================================================
# Default to GPU 0 if not specified (avoid multi-GPU complications)
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Suppress verbose TF logging during simulation runs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Enable dynamic GPU memory allocation to avoid OOM with varying batch sizes
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # GPU context may already be initialized
        print(f"GPU configuration error: {e}", file=sys.stderr)


# =============================================================================
# Command-Line Argument Parsing
# =============================================================================
parser = argparse.ArgumentParser(
    description="Baseline BER/BLER simulation for classical LMMSE PUSCH receiver."
)
parser.add_argument(
    "--num_bs_ant",
    type=int,
    default=16,
    choices=[16, 32],
    help="Number of BS antennas (default: 16)",
)

args = parser.parse_args()
num_bs_ant = args.num_bs_ant

print(f"\n{'='*60}")
print(f"Running baseline simulation with num_bs_ant = {num_bs_ant}")
print(f"{'='*60}\n")

# =============================================================================
# System Configuration
# =============================================================================
_cfg = Config(num_bs_ant=num_bs_ant)
batch_size = _cfg.batch_size
num_ue = _cfg.num_ue
num_ue_ant = _cfg.num_ue_ant
num_time_steps = _cfg.num_time_steps  # Stored for reproducibility in results

# =============================================================================
# Channel Model Setup
# =============================================================================
# Load pre-generated CIR data from TFRecords and wrap in CIRDataset
# This provides realistic ray-traced channels from the Munich urban scenario
cir_manager = CIRManager(config=_cfg)
channel_model = cir_manager.build_channel_model()

# =============================================================================
# Quick Functional Check
# =============================================================================
# Verify the E2E model runs correctly before starting long simulation
# This catches configuration errors early (e.g., missing CIR files)
ebno_db_test = 10.0
e2e_model_test = PUSCHLinkE2E(
    channel_model,
    perfect_csi=False,
    use_autoencoder=False,
    config=_cfg,
)
b, b_hat = e2e_model_test(batch_size, ebno_db_test)
print("Quick check shapes:", b.shape, b_hat.shape)

# =============================================================================
# BER/BLER Monte Carlo Simulation
# =============================================================================
# Eb/N0 range: -2 to 9 dB covers the waterfall region for typical LDPC codes
# Lower values show error floor, higher values approach error-free operation
ebno_db = np.arange(-2, 10, 1)

# PlotBER handles Monte Carlo iteration, early stopping, and result aggregation
ber_plot = PlotBER(f"Site-Specific MU-MIMO 5G NR PUSCH ({num_bs_ant} BS Antennas)")

ber_list, bler_list = [], []
for perf_csi in [True, False]:
    # Create fresh E2E model for each CSI condition
    # (avoids any state leakage between simulations)
    e2e_model = PUSCHLinkE2E(
        channel_model,
        perfect_csi=perf_csi,
        use_autoencoder=False,  # Baseline uses classical LMMSE, no neural detector
        config=_cfg,
    )

    # Run Monte Carlo simulation with early stopping on target block errors
    # max_mc_iter=500 limits runtime; num_target_block_errors=2000 ensures
    # statistical confidence
    ber_i, bler_i = ber_plot.simulate(
        e2e_model,
        ebno_dbs=ebno_db,
        max_mc_iter=500,
        num_target_block_errors=2000,
        batch_size=batch_size,
        soft_estimates=False,  # Use hard-decision BER (after LDPC decoding)
        show_fig=False,  # Don't display plot (save results only)
        add_bler=True,  # Compute both BER and BLER
    )

    # Convert to NumPy arrays for consistent storage format
    ber_list.append(ber_i.numpy() if hasattr(ber_i, "numpy") else np.asarray(ber_i))
    bler_list.append(bler_i.numpy() if hasattr(bler_i, "numpy") else np.asarray(bler_i))

# Stack results: row 0 = perfect CSI, row 1 = imperfect CSI
ber = np.stack(ber_list, axis=0)
bler = np.stack(bler_list, axis=0)

# =============================================================================
# Save Results
# =============================================================================
# Store results with all parameters needed for reproducibility and plotting
os.makedirs(os.path.join(DEMO_DIR, "results"), exist_ok=True)
out_path = os.path.join(DEMO_DIR, "results", f"baseline_results_ant{num_bs_ant}.npz")
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
)
print(f"Saved BER/BLER results to {out_path}")

print("\n" + "=" * 60)
print("Baseline simulation complete.")
print("=" * 60)
