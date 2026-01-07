#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Least-squares DPD training script.

This script trains a Memory Polynomial predistorter using
closed-form least-squares coefficient estimation. Typically
converges in 4-6 iterations.

Parameters
----------
The script accepts command-line arguments:

--iterations : int
    Number of indirect learning iterations. Default: 3.
--batch_size : int
    Number of input waveform batches for coefficient estimation. Default: 16.
--order : int
    Maximum polynomial order (must be odd). Default: 7.
--memory_depth : int
    Number of memory taps per polynomial branch. Default: 4.
--learning_rate : float
    Coefficient update rate (0-1). Default: 0.75.
--learning_method : str
    Update method: 'newton' or 'ema'. Default: 'newton'.

Output Files
------------
- ``results/ls-dpd-weights`` : Pickled model weights for inference
- ``results/ls-dpd-coeff-history.npy`` : Coefficient evolution for analysis

Usage
-----
::

    # Default training (order=7, memory=4, 3 iterations)
    python training_ls.py

    # Custom configuration
    python training_ls.py --order 9 --memory_depth 6 --iterations 5

    # After training, run inference
    python inference.py --dpd_method ls

Notes
-----
**Choosing Polynomial Order:**

Higher orders capture more nonlinearity but risk overfitting and numerical
instability. Order 7 is typical for moderate PA compression.

**Choosing Memory Depth:**

Memory depth should match the PA's memory time constant divided by the
sample period. For wideband signals (high sample rate), more memory taps
are needed. 4-6 taps is typical for 100+ MHz sample rates.

**Newton vs EMA:**

Newton method is more stable and recommended for most cases. EMA may
converge faster but can overshoot with high learning rates.

See Also
--------
training_nn.py : Neural network DPD training (alternative approach).
inference.py : Evaluation script for trained models.
plots_ls.py : Visualization of LS-DPD results.
"""

import os

# Suppress TensorFlow C++ logging (must be set before TF import).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings  # noqa: E402

# Suppress expected Sionna complex->float cast warnings.
warnings.filterwarnings("ignore", message=".*complex64.*float32.*")

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")

# Configure GPU memory growth before any TF operations.
# Prevents OOM errors on shared GPU systems.
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", tf.config.list_logical_devices("GPU"))

import numpy as np  # noqa: E402
import pickle  # noqa: E402
import argparse  # noqa: E402

from demos.dpd.src.config import Config  # noqa: E402
from demos.dpd.src.ls_dpd_system import LS_DPDSystem  # noqa: E402


# =============================================================================
# Command-line argument parsing
# =============================================================================
parser = argparse.ArgumentParser(
    description="Train Least-Squares DPD.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python training_ls.py                          # Default settings
  python training_ls.py --order 9 --iterations 5 # Higher order, more iterations
  python training_ls.py --learning_method ema    # Use EMA update method
    """,
)
parser.add_argument(
    "--iterations",
    type=int,
    default=3,
    help="Number of LS iterations (default: 3)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Batch size for signal generation (default: 16)",
)
parser.add_argument(
    "--order",
    type=int,
    default=7,
    help="DPD polynomial order (default: 7)",
)
parser.add_argument(
    "--memory_depth",
    type=int,
    default=4,
    help="DPD memory depth (default: 4)",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.75,
    help="LS learning rate (default: 0.75)",
)
parser.add_argument(
    "--learning_method",
    type=str,
    default="newton",
    choices=["newton", "ema"],
    help="LS learning method (default: newton)",
)
args = parser.parse_args()

# =============================================================================
# Setup
# =============================================================================
# Create output directory for results.
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(DEMO_DIR, "results"), exist_ok=True)

# Create OFDM configuration with specified batch size.
config = Config(batch_size=args.batch_size)

# =============================================================================
# Build LS-DPD System
# =============================================================================
print("Building LS-DPD System...")
system = LS_DPDSystem(
    training=True,
    config=config,
    dpd_order=args.order,
    dpd_memory_depth=args.memory_depth,
    ls_nIterations=args.iterations,
    ls_learning_rate=args.learning_rate,
    ls_learning_method=args.learning_method,
    rms_input_dbm=0.5,  # Target PA input power.
    pa_sample_rate=122.88e6,  # PA operating sample rate.
)

# Warm-up pass to trigger lazy layer initialization.
# Uses inference mode to build layers without training.
print("Warming up model...")
x_warmup = system.generate_signal(args.batch_size)
_ = system(x_warmup, training=False)
print(f"Number of DPD coefficients: {system.dpd.n_coeffs}")

# =============================================================================
# Estimate PA Gain
# =============================================================================
# PA gain estimation is required before training for proper normalization
# in the indirect learning architecture.
pa_gain = system.estimate_pa_gain()
print(f"Estimated PA gain: {pa_gain:.4f} ({20*np.log10(pa_gain):.2f} dB)")

# =============================================================================
# Perform LS Learning
# =============================================================================
print("\nStarting LS-DPD learning...")
print(f"  Order: {args.order}")
print(f"  Memory depth: {args.memory_depth}")
print(f"  Iterations: {args.iterations}")
print(f"  Learning rate: {args.learning_rate}")
print(f"  Learning method: {args.learning_method}")
print(f"  Batch size: {args.batch_size}")
print()

result = system.perform_ls_learning(
    batch_size=args.batch_size,
    nIterations=args.iterations,
    verbose=True,
)

print("\nLS-DPD learning complete.")

# =============================================================================
# Save Results
# =============================================================================
# Save model weights (includes DPD coefficients and any other state).
weights_file = os.path.join(DEMO_DIR, "results", "ls-dpd-weights")
with open(weights_file, "wb") as f:
    pickle.dump(system.get_weights(), f)
print(f"Saved weights to {weights_file}")

# Save coefficient history for convergence analysis and plotting.
coeff_history_file = os.path.join(DEMO_DIR, "results", "ls-dpd-coeff-history.npy")
np.save(coeff_history_file, result["coeff_history"])
print(f"Saved coefficient history to {coeff_history_file}")
print("Run 'python plots_ls.py' to generate plots.")
