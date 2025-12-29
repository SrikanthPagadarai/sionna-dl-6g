#!/usr/bin/env python3
"""
Training script for Least-Squares DPD using Indirect Learning Architecture.

Unlike NN-DPD which uses gradient descent, LS-DPD uses closed-form least-squares
estimation to find optimal DPD coefficients.

Usage:
    python training_ls.py
    python training_ls.py --iterations 5 --batch_size 32
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*complex64.*float32.*")

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")

# GPU setup - must happen before any TensorFlow operations
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", tf.config.list_logical_devices("GPU"))

import numpy as np  # noqa: E402
import pickle  # noqa: E402
import argparse  # noqa: E402

from demos.dpd.src.config import Config  # noqa: E402
from demos.dpd.src.ls_dpd_system import LS_DPDSystem  # noqa: E402


# CLI
parser = argparse.ArgumentParser(description="Train Least-Squares DPD.")
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

# Filesystem
os.makedirs("results", exist_ok=True)

# Create config
config = Config(batch_size=args.batch_size)

# System
print("Building LS-DPD System...")
system = LS_DPDSystem(
    training=True,
    config=config,
    dpd_order=args.order,
    dpd_memory_depth=args.memory_depth,
    ls_nIterations=args.iterations,
    ls_learning_rate=args.learning_rate,
    ls_learning_method=args.learning_method,
    rms_input_dbm=0.5,
    pa_sample_rate=122.88e6,
)

# Warm-up (variable creation) - use inference mode to build layers
print("Warming up model...")
x_warmup = system.generate_signal(args.batch_size)
_ = system(x_warmup, training=False)
print(f"Number of DPD coefficients: {system.dpd.n_coeffs}")

# Estimate PA gain (required for proper indirect learning)
pa_gain = system.estimate_pa_gain()
print(f"Estimated PA gain: {pa_gain:.4f} ({20*np.log10(pa_gain):.2f} dB)")

# Perform LS learning
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

# Save weights
weights_file = os.path.join("results", "ls-dpd-weights")
with open(weights_file, "wb") as f:
    pickle.dump(system.get_weights(), f)
print(f"Saved weights to {weights_file}")

# Save coefficient history
coeff_history_file = os.path.join("results", "ls-dpd-coeff-history.npy")
np.save(coeff_history_file, result["coeff_history"])
print(f"Saved coefficient history to {coeff_history_file}")
print("Run 'python plots_ls.py' to generate plots.")

# Print final coefficients summary
final_coeffs = result["coeffs"]
print("\nFinal DPD coefficients summary:")
print(f"  Number of coefficients: {len(final_coeffs)}")
print(f"  Max magnitude: {np.max(np.abs(final_coeffs)):.6f}")
print(f"  First coefficient (linear term): {final_coeffs[0, 0]:.6f}")
