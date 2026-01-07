#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Neural network DPD training script.

This script trains a feedforward neural network predistorter using
gradient-based optimization with the indirect learning architecture.
Training uses TensorFlow's ``tf.function`` for GPU-accelerated graph
execution.

Training Features
-----------------
**Gradient Accumulation:**
    Gradients are accumulated over multiple mini-batches before applying
    updates. This effectively increases batch size without requiring more
    GPU memory. Default: 4 accumulation steps.

**Checkpointing:**
    Training state (model weights, optimizer state, iteration count) is
    saved to ``checkpoints/`` directory. Training can be resumed from the
    last checkpoint. Use ``--fresh`` to ignore checkpoints.

**Loss Scaling:**
    The MSE loss is scaled by 1000x for readable values during monitoring.
    This doesn't affect optimization (uniform scaling of gradients).

Parameters
----------
The script accepts command-line arguments:

--iterations : int
    Number of training iterations to run. Default: 10000.
--batch_size : int
    Mini-batch size for each forward pass. Default: 16.
--fresh : flag
    If set, ignore existing checkpoint and start fresh.

Output Files
------------
- ``checkpoints/ckpt-*`` : TensorFlow checkpoint files
- ``checkpoints/iter.txt`` : Current iteration count
- ``checkpoints/loss.npy`` : Loss history for resumption
- ``results/loss.npy`` : Loss history for plotting
- ``results/nn-dpd-weights`` : Pickled weights for inference

Usage
-----
::

    # Train for 10000 iterations (default)
    python training_nn.py

    # Train for more iterations
    python training_nn.py --iterations 50000

    # Start fresh, ignoring any checkpoint
    python training_nn.py --iterations 10000 --fresh

    # Resume training for 5000 more iterations
    python training_nn.py --iterations 5000

    # After training, run inference
    python inference.py --dpd_method nn

Notes
-----
**Convergence:**

NN-DPD typically requires 5000-20000 iterations for good convergence.

**Memory Usage:**

Gradient accumulation allows larger effective batch sizes without
increasing GPU memory. If OOM errors occur, reduce ``--batch_size``
and increase ``ACCUMULATION_STEPS`` to compensate.

**Comparison to LS-DPD:**

NN-DPD is more flexible but slower to train. However, NN-DPD
can potentially achieve better performance for highly nonlinear PAs.

See Also
--------
training_ls.py : Least-squares DPD training.
inference.py : Evaluation script for trained models.
plots_nn.py : Visualization of NN-DPD training and results.
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
from demos.dpd.src.nn_dpd_system import NN_DPDSystem  # noqa: E402


# =============================================================================
# Command-line argument parsing
# =============================================================================
parser = argparse.ArgumentParser(
    description="Train Neural Network DPD.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python training_nn.py                      # Default: 10000 iterations
  python training_nn.py --iterations 50000   # Longer training
  python training_nn.py --fresh              # Ignore checkpoint, start fresh
  python training_nn.py --iterations 5000    # Resume for 5000 more iterations
    """,
)
parser.add_argument(
    "--iterations", type=int, default=10000, help="Train for N more iterations"
)
parser.add_argument(
    "--fresh", action="store_true", help="Start fresh (ignore checkpoint)"
)
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
args = parser.parse_args()

# =============================================================================
# Training configuration
# =============================================================================
BATCH_SIZE = args.batch_size
# Gradient accumulation: apply optimizer every N steps.
# Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS.
ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-4  # Adam default works well for most cases.

# =============================================================================
# Setup directories
# =============================================================================
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(DEMO_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(DEMO_DIR, "checkpoints"), exist_ok=True)
ckpt_dir = os.path.join(DEMO_DIR, "checkpoints")

# Create OFDM configuration.
config = Config(batch_size=BATCH_SIZE)

# =============================================================================
# Build NN-DPD System
# =============================================================================
print("Building NN-DPD System...")
system = NN_DPDSystem(
    training=True,
    config=config,
    dpd_memory_depth=4,  # Sliding window size for memory effects.
    dpd_num_filters=64,  # Hidden layer width.
    dpd_num_layers_per_block=2,  # Layers per residual block.
    dpd_num_res_blocks=3,  # Number of residual blocks.
    rms_input_dbm=0.5,  # Target PA input power.
    pa_sample_rate=122.88e6,  # PA operating sample rate.
)

# Warm-up pass to trigger lazy variable creation.
print("Warming up model...")
_ = system(BATCH_SIZE, training=True)
print(f"Number of trainable variables: {len(system.trainable_variables)}")
print(
    "Total trainable parameters: ",
    f"{sum(tf.size(v).numpy() for v in system.trainable_variables)}",
)

# =============================================================================
# Estimate PA Gain
# =============================================================================
# Required for indirect learning normalization.
pa_gain = system.estimate_pa_gain()
print(f"Estimated PA gain: {pa_gain:.4f} ({20*np.log10(pa_gain):.2f} dB)")

# =============================================================================
# Setup optimizer and checkpointing
# =============================================================================
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Checkpoint includes model and optimizer state for full resumption.
checkpoint = tf.train.Checkpoint(
    model=system,
    optimizer=optimizer,
)

# Attempt to restore from checkpoint unless --fresh flag is set.
start_iteration = 0
loss_history = []
latest = tf.train.latest_checkpoint(ckpt_dir)
if not args.fresh and latest:
    checkpoint.restore(latest)
    # Load iteration count and loss history from separate files.
    iter_file = os.path.join(ckpt_dir, "iter.txt")
    loss_file = os.path.join(ckpt_dir, "loss.npy")
    if os.path.exists(iter_file):
        start_iteration = int(open(iter_file).read())
    if os.path.exists(loss_file):
        loss_history = np.load(loss_file).tolist()
    print(f"Resumed from iteration {start_iteration}")

target_iteration = start_iteration + args.iterations
print(f"Training from {start_iteration} to {target_iteration}")


# =============================================================================
# Define training step (graph-compiled for performance)
# =============================================================================
@tf.function(reduce_retracing=True)
def train_step(batch_size):
    """
    Execute one training step with gradient computation.

    This function is compiled to a TensorFlow graph for GPU acceleration.
    The entire forward pass, loss computation, and gradient calculation
    happen on GPU without Python overhead.

    Parameters
    ----------
    batch_size : tf.Tensor
        Batch size as a TensorFlow constant (avoids retracing).

    Returns
    -------
    loss : tf.Tensor
        Scalar loss value.
    grads : list of tf.Tensor
        Gradients for all trainable variables.
    """
    with tf.GradientTape() as tape:
        loss = system(batch_size, training=True)
    grads = tape.gradient(loss, system.trainable_variables)
    # Replace None gradients with zeros (for variables not in compute path).
    grads = [
        g if g is not None else tf.zeros_like(w)
        for g, w in zip(grads, system.trainable_variables)
    ]
    return loss, grads


# =============================================================================
# Align iterations to accumulation steps
# =============================================================================
# Ensure clean gradient accumulation boundaries.
if start_iteration % ACCUMULATION_STEPS != 0:
    start_iteration = (start_iteration // ACCUMULATION_STEPS) * ACCUMULATION_STEPS
    print(f"Adjusted start_iteration to {start_iteration} for accumulation alignment")

if target_iteration % ACCUMULATION_STEPS != 0:
    target_iteration = (
        (target_iteration // ACCUMULATION_STEPS) + 1
    ) * ACCUMULATION_STEPS
    print(f"Adjusted target_iteration to {target_iteration} for accumulation alignment")

# =============================================================================
# Training loop
# =============================================================================
print("\nStarting training...")

# Pre-allocate gradient accumulators as non-trainable Variables.
# This avoids memory allocation during the training loop.
accumulated_grads = [
    tf.Variable(tf.zeros_like(v), trainable=False) for v in system.trainable_variables
]

# Use tensor for batch_size to avoid tf.function retracing.
batch_size_tensor = tf.constant(BATCH_SIZE, dtype=tf.int32)

for i in range(start_iteration, target_iteration):
    # Forward pass and gradient computation.
    loss, grads = train_step(batch_size_tensor)

    # Accumulate gradients.
    for acc_g, g in zip(accumulated_grads, grads):
        acc_g.assign_add(g)

    # Apply accumulated gradients every ACCUMULATION_STEPS.
    if (i + 1) % ACCUMULATION_STEPS == 0:
        # Average gradients over accumulation window.
        avg_grads = [g / ACCUMULATION_STEPS for g in accumulated_grads]
        optimizer.apply_gradients(zip(avg_grads, system.trainable_variables))
        # Reset accumulators for next window.
        for acc_g in accumulated_grads:
            acc_g.assign(tf.zeros_like(acc_g))

    loss_value = float(loss.numpy())
    loss_history.append(loss_value)

    # Progress logging (overwrite same line).
    print(
        f"\rStep {i + 1}/{target_iteration}  Loss: {loss_value:.6f}",
        end="",
        flush=True,
    )

print("\n\nTraining complete.")

# =============================================================================
# Save training state and results
# =============================================================================
# Save TensorFlow checkpoint (model + optimizer state).
checkpoint.save(os.path.join(ckpt_dir, "ckpt"))

# Save iteration count for resumption.
open(os.path.join(ckpt_dir, "iter.txt"), "w").write(str(target_iteration))

# Save loss history (both in checkpoint dir for resumption and results for plotting).
np.save(os.path.join(ckpt_dir, "loss.npy"), loss_history)
np.save(
    os.path.join(DEMO_DIR, "results", "loss.npy"),
    np.array(loss_history, dtype=np.float32),
)

# Save pickled weights for inference script compatibility.
with open(os.path.join(DEMO_DIR, "results", "nn-dpd-weights"), "wb") as f:
    pickle.dump(system.get_weights(), f)

print("Saved checkpoints, loss history, and weights.")
print("Run 'python plots_nn.py' to generate plots.")
