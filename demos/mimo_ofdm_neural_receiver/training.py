# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Training script for the MIMO-OFDM neural receiver.

This script implements the training loop for the neural receiver, including:

- Gradient accumulation for effective batch size scaling
- Checkpoint/resume functionality for long training runs
- Random Eb/N0 sampling to train across SNR operating range
- Loss history tracking for convergence analysis

Usage
-----
Train for 10,000 iterations from scratch::

    python training.py --iterations 10000 --fresh

Resume training for 5,000 more iterations::

    python training.py --iterations 5000

Training Configuration
----------------------
The following hyperparameters are configured in this script:

- ``BATCH_SIZE``: Physical batch size per forward pass (32)
- ``ACCUMULATION_STEPS``: Gradient accumulation factor (4)
- Effective batch size = BATCH_SIZE x ACCUMULATION_STEPS = 128
- ``EBN0_DB_MIN/MAX``: SNR range for training (-3 to 7 dB)
- Neural Rx architecture: 512 filters, 12 residual blocks

Output Files
------------
Checkpoints (``checkpoints/`` directory):
    - ``ckpt-*``: TensorFlow checkpoint files (model + optimizer state)
    - ``iter.txt``: Current iteration count
    - ``loss.npy``: Complete loss history

Results (``results/`` directory):
    - ``loss.npy``: Loss history (copy for analysis)
    - ``mimo-ofdm-neuralrx-weights``: Pickled model weights

Note
----
Training uses random Eb/N0 values within each batch, exposing the network
to varying SNR conditions. This produces a receiver that generalizes across
the operating range rather than overfitting to a single SNR point.

The checkpoint system saves model weights, optimizer state, AND the random
number generator state, ensuring perfect reproducibility when resuming.
"""

import tensorflow as tf
import numpy as np
import pickle
import os
import argparse

from demos.mimo_ofdm_neural_receiver.src.system import System


# =============================================================================
# Command Line Interface
# =============================================================================
parser = argparse.ArgumentParser(description="Train NeuralRx.")
parser.add_argument(
    "--iterations", type=int, default=10000, help="Train for N more iterations"
)
parser.add_argument(
    "--fresh", action="store_true", help="Start fresh (ignore checkpoint)"
)
args = parser.parse_args()

# =============================================================================
# GPU Configuration
# Memory growth prevents TensorFlow from allocating all GPU memory at startup
# =============================================================================
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", tf.config.list_logical_devices("GPU"))

# =============================================================================
# Training Hyperparameters
# =============================================================================
# Physical batch size per forward pass
BATCH_SIZE = 32

# SNR range for training: network learns to operate across this range
# -3 to 7 dB covers typical operating points for QPSK with rate-1/2 LDPC
EBN0_DB_MIN = -3.0
EBN0_DB_MAX = 7.0

# Gradient accumulation: effective_batch = BATCH_SIZE x ACCUMULATION_STEPS
# Larger effective batch provides more stable gradients without OOM
ACCUMULATION_STEPS = 4

# =============================================================================
# Directory Setup
# =============================================================================
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(DEMO_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(DEMO_DIR, "checkpoints"), exist_ok=True)
ckpt_dir = os.path.join(DEMO_DIR, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

# =============================================================================
# Model Instantiation
# Architecture: 256 filters x 4 residual blocks
# =============================================================================
system = System(
    training=True,
    use_neural_rx=True,
    num_conv2d_filters=256,
    num_res_blocks=4,
)

# Warm-up call to trigger variable creation before checkpoint restore
# Without this, variables wouldn't exist when restore() is called
_ = system(tf.constant(BATCH_SIZE, tf.int32), tf.fill([BATCH_SIZE], 10.0))
print("num trainables:", len(system.trainable_variables))

# =============================================================================
# Optimizer and Checkpoint Setup
# Checkpoint includes: model weights, optimizer state, RNG state
# =============================================================================
optimizer = tf.keras.optimizers.Adam()

# Deterministic RNG for reproducibility; state is checkpointed
rng = tf.random.Generator.from_seed(42)

checkpoint = tf.train.Checkpoint(
    model=system,
    optimizer=optimizer,
    rng=rng,
)

# =============================================================================
# Resume Logic
# Load checkpoint if exists and --fresh not specified
# =============================================================================
start_iteration = 0
loss_history = []
latest = tf.train.latest_checkpoint(ckpt_dir)
if not args.fresh and latest:
    checkpoint.restore(latest)
    start_iteration = int(open(os.path.join(ckpt_dir, "iter.txt")).read())
    loss_history = np.load(os.path.join(ckpt_dir, "loss.npy")).tolist()
    print(f"Resumed from iteration {start_iteration}")

target_iteration = start_iteration + args.iterations
print(f"Training from {start_iteration} to {target_iteration}")


# =============================================================================
# Training Step Function
# Returns loss and gradients; accumulation handled in main loop
# =============================================================================
@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([None], tf.float32),
    ],
)
def train_step(batch_size, ebno_vec):
    """
    Execute single forward/backward pass.

    Parameters
    ----------
    batch_size : tf.Tensor, int32, scalar
        Number of samples in batch.

    ebno_vec : tf.Tensor, float32, [batch_size]
        Per-sample Eb/N0 in dB.

    Returns
    -------
    loss : tf.Tensor, float32, scalar
        BCE loss for this batch.

    grads : List[tf.Tensor]
        Gradients for each trainable variable. None gradients are
        replaced with zeros to avoid issues in accumulation.

    Note
    ----
    The @tf.function decorator with input_signature ensures the function
    is traced once and reused, avoiding retracing overhead each iteration.
    """
    with tf.GradientTape() as tape:
        loss = system(batch_size, ebno_vec)
    grads = tape.gradient(loss, system.trainable_variables)
    # Replace None gradients with zeros (occurs for unused variables)
    grads = [
        g if g is not None else tf.zeros_like(w)
        for g, w in zip(grads, system.trainable_variables)
    ]
    return loss, grads


# =============================================================================
# Validation: Accumulation Alignment
# Start/target must align with accumulation steps for correct averaging
# =============================================================================
if start_iteration % ACCUMULATION_STEPS != 0:
    raise ValueError("start_iteration must be a multiple of ACCUMULATION_STEPS")

if target_iteration % ACCUMULATION_STEPS != 0:
    raise ValueError("target_iteration must be a multiple of ACCUMULATION_STEPS")

# [training-loop-start]
# =============================================================================
# Training Loop
# Gradient accumulation: sum gradients over ACCUMULATION_STEPS, then apply
# =============================================================================
accumulated_grads = None
for i in range(start_iteration, target_iteration):
    # Sample random Eb/N0 for each batch element
    # Training across SNR range improves generalization
    ebno_db = rng.uniform([BATCH_SIZE], EBN0_DB_MIN, EBN0_DB_MAX, tf.float32)
    loss, grads = train_step(tf.constant(BATCH_SIZE, tf.int32), ebno_db)

    # Accumulate gradients
    if accumulated_grads is None:
        accumulated_grads = [tf.Variable(g, trainable=False) for g in grads]
    else:
        for acc_g, g in zip(accumulated_grads, grads):
            acc_g.assign_add(g)

    # Apply accumulated gradients every ACCUMULATION_STEPS iterations
    if (i + 1) % ACCUMULATION_STEPS == 0:
        avg_grads = [g / ACCUMULATION_STEPS for g in accumulated_grads]
        optimizer.apply_gradients(zip(avg_grads, system.trainable_variables))
        accumulated_grads = None

    loss_value = float(loss.numpy())
    loss_history.append(loss_value)

    print(
        f"\rStep {i}/{target_iteration}  Loss: {loss_value:.4f}",
        end="",
        flush=True,
    )
print("\n\nTraining complete.")
# [training-loop-end]

# =============================================================================
# Save Checkpoint and Results
# =============================================================================
# TensorFlow checkpoint (model + optimizer + RNG state)
checkpoint.save(os.path.join(ckpt_dir, "ckpt"))

# Iteration counter for resume
open(os.path.join(ckpt_dir, "iter.txt"), "w").write(str(target_iteration))

# Loss history (both in checkpoint dir and results dir)
np.save(os.path.join(ckpt_dir, "loss.npy"), loss_history)
np.save(
    os.path.join(DEMO_DIR, "results", "loss.npy"),
    np.array(loss_history, dtype=np.float32),
)

# Pickled weights for easy loading without full checkpoint restore
with open(os.path.join(DEMO_DIR, "results", "mimo-ofdm-neuralrx-weights"), "wb") as f:
    pickle.dump(system.get_weights(), f)
print("Saved checkpoints, loss history, and weights.")
