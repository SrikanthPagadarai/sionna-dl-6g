# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
End-to-end training script for PUSCH autoencoder with neural detection.

This script trains the joint transmitter-receiver autoencoder system,
optimizing both the constellation geometry and the neural MIMO detector
to minimize Binary Cross-Entropy (BCE) loss on coded bits.

Architecture
------------
The autoencoder consists of:

- **Transmitter**: Trainable 16-QAM constellation (16 complex points)
- **Receiver**: Neural MIMO detector with:
  - Shared backbone (4 ResBlocks)
  - Channel estimation refinement head
  - Detection continuation network (6 ResBlocks)
  - Three trainable correction scales (h, err_var, LLR)

Training Strategy
-----------------
- **Gradient accumulation**: 16 micro-batches averaged before update
  to reduce gradient variance and enable larger effective batch sizes.
- **Separate optimizers**: TX, RX scales, and RX NN weights use different
  learning rates (1e-2, 1e-2, 1e-4 respectively with cosine decay).
- **SNR curriculum**: Random Eb/N0 in [-2, 10] dB per batch for robustness
  across operating conditions.

Output
------
Results are saved to ``results/`` directory with antenna suffix:

- ``PUSCH_autoencoder_weights_ant{num_bs_ant}``: Final TX/RX weights (pickle)
- ``training_loss_ant{num_bs_ant}.npy``: Loss history array
- ``training_loss_ant{num_bs_ant}.png``: Loss curve plot
- ``constellations_overlaid_ant{num_bs_ant}.png``: Before/after constellation comparison
- Checkpoints every 1000 iterations

Usage
-----
Run from the repository root::

    python -m demos.pusch_autoencoder.training
    python -m demos.pusch_autoencoder.training --num_bs_ant 32
"""

import os
import pickle
import argparse
import tensorflow as tf
from demos.pusch_autoencoder.src.cir_manager import CIRManager
from demos.pusch_autoencoder.src.system import PUSCHLinkE2E
from demos.pusch_autoencoder.src.config import Config
import matplotlib.pyplot as plt
import numpy as np

import time

start = time.time()

# get directory name of file
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Command-Line Argument Parsing
# =============================================================================
parser = argparse.ArgumentParser(
    description="Train PUSCH autoencoder with neural detection."
)
parser.add_argument(
    "--num_bs_ant",
    type=int,
    default=16,
    help="Number of BS antennas (default: 16)",
)

args = parser.parse_args()
num_bs_ant = args.num_bs_ant

print(f"Number of BS antennas: {num_bs_ant}")


# =============================================================================
# System Configuration and Channel Model
# =============================================================================
_cfg = Config(num_bs_ant=num_bs_ant)
batch_size = _cfg.batch_size

# Load MU-MIMO grouped CIR data (4 UEs per sample)
# This is different from baseline.py which uses CIRDataset for on-demand sampling
cir_manager = CIRManager(config=_cfg)
channel_model = cir_manager.load_from_tfrecord(group_for_mumimo=True)


# =============================================================================
# Model Instantiation and Initial Verification
# =============================================================================
# Create E2E model in training mode (returns BCE loss, not decoded bits)
ebno_db_test = tf.fill([batch_size], 10.0)
model = PUSCHLinkE2E(
    channel_model, perfect_csi=False, use_autoencoder=True, training=True, config=_cfg
)

# Verify forward pass works and inspect model structure
loss = model(batch_size, ebno_db_test)
print("  Initial forward-pass loss:", loss.numpy())
print("  Trainable variable count:", len(model.trainable_variables))
for v in model.trainable_variables[:5]:
    print("   ", v.name, v.shape)

# Snapshot initial constellation for before/after comparison plot
# Using tf.identity to create a copy that won't be modified during training
init_const_real = tf.identity(model._pusch_transmitter._points_r)
init_const_imag = tf.identity(model._pusch_transmitter._points_i)


# =============================================================================
# Variable Grouping for Separate Learning Rates
# =============================================================================
# Different components benefit from different learning rates:
# - TX constellation: slow updates to allow RX to adapt
# - RX correction scales: fast updates to find good operating points
# - RX NN weights: moderate updates for stable convergence

tx_vars = model._pusch_transmitter.trainable_variables
rx_vars_all = model._pusch_receiver.trainable_variables

# Neural detector returns variables in specific order (see PUSCHNeuralDetector):
# [h_correction_scale, err_var_correction_scale_raw, llr_correction_scale, ...conv weights...]
rx_scale_vars = rx_vars_all[:3]
nn_rx_vars = rx_vars_all[3:]

print("\n=== Variable groups ===")
print(f"TX vars: {len(tx_vars)}")
for v in tx_vars:
    print(f"  {v.name}: {v.shape}")

print(f"\nRX Scale vars: {len(rx_scale_vars)}")
for v in rx_scale_vars:
    print(f"  {v.name}: {v.shape}")

print(f"\nNN RX vars: {len(nn_rx_vars)} (showing first 5)")
for v in nn_rx_vars[:5]:
    print(f"  {v.name}: {v.shape}")
print("=== End variable groups ===\n")

# Combined list for gradient computation (order matters for slicing)
all_vars = tx_vars + rx_scale_vars + nn_rx_vars


# =============================================================================
# Gradient Sanity Check
# =============================================================================
# Verify gradients flow to all variable groups before starting long training.
# This catches issues like disconnected computation graphs or None gradients.
print("\n=== Single-step gradient sanity check ===")

dbg_batch_size = 4
dbg_ebno = tf.fill([dbg_batch_size], 10.0)

with tf.GradientTape() as tape:
    loss_dbg = model(dbg_batch_size, dbg_ebno)

all_grads = tape.gradient(loss_dbg, all_vars)

# Slice gradients to match variable groups
n_tx = len(tx_vars)
n_scales = len(rx_scale_vars)

grads_tx = all_grads[:n_tx]
grads_scales = all_grads[n_tx : n_tx + n_scales]
grads_rx_nn = all_grads[n_tx + n_scales :]

# Print gradient norms to verify flow (None = disconnected, 0 = vanishing)
print("\nTransmitter gradients:")
for v, g in zip(tx_vars, grads_tx):
    g_norm = 0.0 if g is None else float(tf.norm(g).numpy())
    print(f"  {v.name:40s} grad_norm = {g_norm:.3e}")

print("\nReceiver correction scale gradients:")
for v, g in zip(rx_scale_vars, grads_scales):
    g_norm = 0.0 if g is None else float(tf.norm(g).numpy())
    print(f"  {v.name:40s} grad_norm = {g_norm:.3e}")

print("\nReceiver NN gradients (first 5):")
for v, g in zip(nn_rx_vars[:5], grads_rx_nn[:5]):
    g_norm = 0.0 if g is None else float(tf.norm(g).numpy())
    print(f"  {v.name:40s} grad_norm = {g_norm:.3e}")

print("=== End gradient sanity check ===\n")


# =============================================================================
# Training Hyperparameters
# =============================================================================
ebno_db_min = -2.0  # Low SNR for learning robustness
ebno_db_max = 10.0  # High SNR for fine-tuning constellation geometry
training_batch_size = batch_size
num_training_iterations = 5000

# Cosine decay schedules: start high, decay to 1% of initial LR
# TX and scales get higher LR (1e-2) for faster initial adaptation
# NN weights get lower LR (1e-4) to avoid disrupting pretrained-like behavior
lr_schedule_tx = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-2,
    decay_steps=num_training_iterations,
    alpha=0.01,
)
lr_schedule_rx_scales = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-2,
    decay_steps=num_training_iterations,
    alpha=0.01,
)
lr_schedule_rx_nn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=num_training_iterations,
    alpha=0.01,
)

optimizer_tx = tf.keras.optimizers.Adam(learning_rate=lr_schedule_tx)
optimizer_scales = tf.keras.optimizers.Adam(learning_rate=lr_schedule_rx_scales)
optimizer_rx = tf.keras.optimizers.Adam(learning_rate=lr_schedule_rx_nn)


# =============================================================================
# Gradient Computation Functions
# =============================================================================
# Gradient accumulation: 16 micro-batches averaged to reduce variance
# This simulates larger batch training without memory overhead
accumulation_steps = 16


@tf.function
def compute_grads_single():
    """
    Compute gradients for a single random SNR micro-batch.

    Returns
    -------
    loss : tf.Tensor
        BCE loss for this micro-batch.
    grads : list of tf.Tensor
        Gradients for all trainable variables.
    """
    ebno_db = tf.random.uniform(
        [training_batch_size], minval=ebno_db_min, maxval=ebno_db_max
    )
    with tf.GradientTape() as tape:
        loss = model(training_batch_size, ebno_db)
    grads = tape.gradient(loss, all_vars)
    return loss, grads


def compute_accumulated_grads():
    """
    Compute gradients accumulated over multiple micro-batches.

    Averages gradients from accumulation_steps forward passes to reduce
    variance. This simulates larger batch training without memory overhead.

    Returns
    -------
    avg_loss : tf.Tensor
        Mean BCE loss over all micro-batches.
    grads_tx : list of tf.Tensor
        Averaged gradients for transmitter variables.
    grads_scales : list of tf.Tensor
        Averaged gradients for correction scale variables.
    grads_rx_nn : list of tf.Tensor
        Averaged gradients for neural network weight variables.
    """
    accumulated_grads = [tf.zeros_like(v) for v in all_vars]
    total_loss = 0.0

    for _ in range(accumulation_steps):
        loss, grads = compute_grads_single()
        accumulated_grads = [ag + g for ag, g in zip(accumulated_grads, grads)]
        total_loss += loss

    # Average over accumulation steps
    accumulated_grads = [g / accumulation_steps for g in accumulated_grads]
    avg_loss = total_loss / accumulation_steps

    # Split into variable groups for separate optimizer application
    grads_tx = accumulated_grads[:n_tx]
    grads_scales = accumulated_grads[n_tx : n_tx + n_scales]
    grads_rx_nn = accumulated_grads[n_tx + n_scales :]

    return avg_loss, grads_tx, grads_scales, grads_rx_nn


# =============================================================================
# Main Training Loop
# =============================================================================
loss_history = []

# Suffix for all output files
ant_suffix = f"_ant{num_bs_ant}"

print(f"Starting training for {num_training_iterations} iterations...")
print("  TX LR: 1e-2, RX Scales LR: 1e-2, RX NN LR: 1e-4")
print(f"  Output files will have suffix: {ant_suffix}")

for i in range(num_training_iterations):
    avg_loss, grads_tx, grads_scales, grads_rx_nn = compute_accumulated_grads()
    loss_value = float(avg_loss.numpy())
    loss_history.append(loss_value)

    # Simultaneous update: all variable groups updated together
    optimizer_tx.apply_gradients(zip(grads_tx, tx_vars))
    optimizer_scales.apply_gradients(zip(grads_scales, rx_scale_vars))
    optimizer_rx.apply_gradients(zip(grads_rx_nn, nn_rx_vars))

    # Progress display (overwrite same line)
    print(
        "Iteration {}/{}  BCE: {:.4f}".format(
            i + 1, num_training_iterations, loss_value
        ),
        end="\r",
        flush=True,
    )

    # Periodic checkpointing to resume from crashes
    if (i + 1) % 1000 == 0:
        os.makedirs(os.path.join(DEMO_DIR, "results"), exist_ok=True)
        save_path = os.path.join(
            DEMO_DIR, "results", f"PUSCH_autoencoder_weights_iter_{i + 1}{ant_suffix}"
        )

        # Store both raw variables and normalized constellation
        normalized_const = (
            model._pusch_transmitter.get_normalized_constellation().numpy()
        )
        weights_dict = {
            "tx_weights": [
                v.numpy() for v in model._pusch_transmitter.trainable_variables
            ],
            "rx_weights": [
                v.numpy() for v in model._pusch_receiver.trainable_variables
            ],
            "tx_names": [v.name for v in model._pusch_transmitter.trainable_variables],
            "rx_names": [v.name for v in model._pusch_receiver.trainable_variables],
            "normalized_constellation": normalized_const,
        }
        with open(save_path, "wb") as f:
            pickle.dump(weights_dict, f)
        print(f"[Checkpoint] Saved weights at iteration {i + 1} -> {save_path}")

print()  # Newline after progress display


# =============================================================================
# Save Final Results
# =============================================================================
os.makedirs(os.path.join(DEMO_DIR, "results"), exist_ok=True)

# Save loss history for analysis
loss_path = os.path.join(DEMO_DIR, "results", f"training_loss{ant_suffix}.npy")
np.save(loss_path, np.array(loss_history))

# Save final weights
weights_path = os.path.join(
    DEMO_DIR, "results", f"PUSCH_autoencoder_weights{ant_suffix}"
)

normalized_const = model._pusch_transmitter.get_normalized_constellation().numpy()
weights_dict = {
    "tx_weights": [v.numpy() for v in model._pusch_transmitter.trainable_variables],
    "rx_weights": [v.numpy() for v in model._pusch_receiver.trainable_variables],
    "tx_names": [v.name for v in model._pusch_transmitter.trainable_variables],
    "rx_names": [v.name for v in model._pusch_receiver.trainable_variables],
    "normalized_constellation": normalized_const,
}
with open(weights_path, "wb") as f:
    pickle.dump(weights_dict, f)

print(
    f"Saved {len(weights_dict['tx_weights'])} TX and "
    f"{len(weights_dict['rx_weights'])} RX weight arrays"
)

# Print final correction scale values for analysis
# These indicate how much the neural network deviates from classical LMMSE
print("\nFinal correction scales:")
h_scale = float(rx_scale_vars[0].numpy())
err_var_scale_raw = float(rx_scale_vars[1].numpy())
err_var_scale = float(np.log(1 + np.exp(err_var_scale_raw)))  # softplus
llr_scale = float(rx_scale_vars[2].numpy())
print(f"  h_correction_scale: {h_scale:.6f}")
print(f"  err_var_correction_scale (softplus): {err_var_scale:.6f}")
print(f"  llr_correction_scale: {llr_scale:.6f}")


# =============================================================================
# Visualization: Training Loss Curve
# =============================================================================
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("BCE loss")
plt.title(f"Training loss vs. iteration ({num_bs_ant} BS ant)")
plt.grid(True, linestyle="--", linewidth=0.5)

loss_fig_path = os.path.join(DEMO_DIR, "results", f"training_loss{ant_suffix}.png")
plt.savefig(loss_fig_path, dpi=150)
plt.close()

print(f"Saved training loss plot to: {loss_fig_path}")


# =============================================================================
# Visualization: Constellation Before vs After
# =============================================================================
# Compare initial QAM geometry to trained constellation
# Significant movement indicates the optimizer found channel-adapted points
trained_const_real = model._pusch_transmitter._points_r
trained_const_imag = model._pusch_transmitter._points_i

const_init = tf.complex(init_const_real, init_const_imag)
const_trained = tf.complex(trained_const_real, trained_const_imag)

os.makedirs(os.path.join(DEMO_DIR, "results"), exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 5))

pts_init = const_init.numpy()
pts_trained = const_trained.numpy()

ax.scatter(pts_init.real, pts_init.imag, s=25, marker="o", label="Initial")
ax.scatter(pts_trained.real, pts_trained.imag, s=25, marker="x", label="Trained")

ax.axhline(0.0, linewidth=0.5)
ax.axvline(0.0, linewidth=0.5)
ax.set_aspect("equal", "box")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.set_title(f"Constellation: initial vs trained ({num_bs_ant} BS ant)")
ax.set_xlabel("In-phase")
ax.set_ylabel("Quadrature")
ax.legend()

fig.tight_layout()
fig_path = os.path.join(DEMO_DIR, "results", f"constellations_overlaid{ant_suffix}.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)

print(f"Saved constellation comparison plot to: {fig_path}")

print("Total time:", time.time() - start, "seconds")
