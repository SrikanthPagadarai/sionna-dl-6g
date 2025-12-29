import os
import sys
import pickle
import tensorflow as tf
from src.cir_manager import CIRManager
from src.system import PUSCHLinkE2E
from src.config import Config
import matplotlib.pyplot as plt
import numpy as np

import time

start = time.time()

# ----------------------------------------
# Parse command-line argument
# ----------------------------------------
if len(sys.argv) != 2 or sys.argv[1] not in ("conventional", "two_phase"):
    print("Usage: python training.py [conventional|two_phase]")
    sys.exit(1)

training_mode = sys.argv[1]
print(f"Training mode: {training_mode}")

# Get configuration
_cfg = Config()
batch_size = _cfg.batch_size

# Build channel model
cir_manager = CIRManager()
channel_model = cir_manager.load_from_tfrecord(group_for_mumimo=True)

# Instantiate and train the end-to-end system
ebno_db_test = tf.fill([batch_size], 10.0)
model = PUSCHLinkE2E(
    channel_model, perfect_csi=False, use_autoencoder=True, training=True
)
loss = model(batch_size, ebno_db_test)
print("  Initial forward-pass loss:", loss.numpy())
print("  Trainable variable count:", len(model.trainable_variables))
for v in model.trainable_variables[:5]:
    print("   ", v.name, v.shape)

# Snapshot initial constellation (before training)
init_const_real = tf.identity(model._pusch_transmitter._points_r)
init_const_imag = tf.identity(model._pusch_transmitter._points_i)

# ----------------------------------------
# Split variables into groups with different learning rates
# ----------------------------------------
tx_vars = model._pusch_transmitter.trainable_variables
rx_vars_all = model._pusch_receiver.trainable_variables

# Split RX variables: first 3 are correction scales, rest are NN weights
# Order in trainable_variables:
# _h_correction_scale,
# _err_var_correction_scale_raw,
# _llr_correction_scale,
# then conv weights
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

# All variables for gradient computation
all_vars = tx_vars + rx_scale_vars + nn_rx_vars

# ---- Gradient sanity check: one step in eager mode ----
print("\n=== Single-step gradient sanity check ===")

dbg_batch_size = 4
dbg_ebno = tf.fill([dbg_batch_size], 10.0)

with tf.GradientTape() as tape:
    loss_dbg = model(dbg_batch_size, dbg_ebno)

all_grads = tape.gradient(loss_dbg, all_vars)

n_tx = len(tx_vars)
n_scales = len(rx_scale_vars)

grads_tx = all_grads[:n_tx]
grads_scales = all_grads[n_tx : n_tx + n_scales]
grads_rx_nn = all_grads[n_tx + n_scales :]

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

# ----------------------------------------
# Training loop
# ----------------------------------------
ebno_db_min = -2.0
ebno_db_max = 10.0
training_batch_size = batch_size
num_training_iterations = 5000

# Learning rate schedules - scales get 100x higher LR
lr_schedule_tx = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-2, decay_steps=num_training_iterations, alpha=0.01
)
lr_schedule_scales = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-2, decay_steps=num_training_iterations, alpha=0.01
)
lr_schedule_rx = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4, decay_steps=num_training_iterations, alpha=0.01
)

optimizer_tx = tf.keras.optimizers.Adam(learning_rate=lr_schedule_tx)
optimizer_scales = tf.keras.optimizers.Adam(learning_rate=lr_schedule_scales)
optimizer_rx = tf.keras.optimizers.Adam(learning_rate=lr_schedule_rx)

accumulation_steps = 16


@tf.function(jit_compile=False)
def compute_grads_single():
    ebno_db = tf.random.uniform(
        shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max
    )
    with tf.GradientTape() as tape:
        loss = model(training_batch_size, ebno_db)
    grads = tape.gradient(loss, all_vars)
    return loss, grads


def compute_accumulated_grads():
    """Compute accumulated gradients without applying them."""
    accumulated_grads = [tf.zeros_like(v) for v in all_vars]
    total_loss = 0.0

    for _ in range(accumulation_steps):
        loss, grads = compute_grads_single()
        accumulated_grads = [ag + g for ag, g in zip(accumulated_grads, grads)]
        total_loss += loss

    # Average
    accumulated_grads = [g / accumulation_steps for g in accumulated_grads]
    avg_loss = total_loss / accumulation_steps

    grads_tx = accumulated_grads[:n_tx]
    grads_scales = accumulated_grads[n_tx : n_tx + n_scales]
    grads_rx_nn = accumulated_grads[n_tx + n_scales :]

    return avg_loss, grads_tx, grads_scales, grads_rx_nn


# store loss values for plotting
loss_history = []

print(f"Starting {training_mode} training for {num_training_iterations} iterations...")
print("  TX LR: 1e-2, RX Scales LR: 1e-2, RX NN LR: 1e-4")

for i in range(num_training_iterations):
    avg_loss, grads_tx, grads_scales, grads_rx_nn = compute_accumulated_grads()
    loss_value = float(avg_loss.numpy())
    loss_history.append(loss_value)

    if training_mode == "conventional":
        # Update all three groups every iteration
        optimizer_tx.apply_gradients(zip(grads_tx, tx_vars))
        optimizer_scales.apply_gradients(zip(grads_scales, rx_scale_vars))
        optimizer_rx.apply_gradients(zip(grads_rx_nn, nn_rx_vars))

    elif training_mode == "two_phase":
        # 10 RX updates (with fresh gradients each time)
        for _ in range(10):
            _, _, grads_scales_fresh, grads_rx_nn_fresh = compute_accumulated_grads()
            optimizer_scales.apply_gradients(zip(grads_scales_fresh, rx_scale_vars))
            optimizer_rx.apply_gradients(zip(grads_rx_nn_fresh, nn_rx_vars))

        # 1 TX update (with fresh gradient after RX has adapted)
        _, grads_tx_fresh, _, _ = compute_accumulated_grads()
        optimizer_tx.apply_gradients(zip(grads_tx_fresh, tx_vars))

    print(
        "Iteration {}/{}  BCE: {:.4f}".format(
            i + 1, num_training_iterations, loss_value
        ),
        end="\r",
        flush=True,
    )

    # Save weights intermittently
    if (i + 1) % 1000 == 0:
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join(
            "results", f"PUSCH_autoencoder_weights_{training_mode}_iter_{i + 1}"
        )

        # Get normalized constellation
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

print()  # newline after the loop

# Save training loss
os.makedirs("results", exist_ok=True)
loss_path = os.path.join("results", f"{training_mode}_training_loss.npy")

# Save weights
np.save(loss_path, np.array(loss_history))
weights_path = os.path.join(
    "results", f"PUSCH_autoencoder_weights_{training_mode}_training"
)

# Get normalized constellation
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

# Print final scale values
print("\nFinal correction scales:")
h_scale = float(rx_scale_vars[0].numpy())
err_var_scale_raw = float(rx_scale_vars[1].numpy())
err_var_scale = float(np.log(1 + np.exp(err_var_scale_raw)))
llr_scale = float(rx_scale_vars[2].numpy())
print(f"  h_correction_scale: {h_scale:.6f}")
print(f"  err_var_correction_scale (softplus): {err_var_scale:.6f}")
print(f"  llr_correction_scale: {llr_scale:.6f}")

# ----------------------------------------
# Plot training loss vs iteration
# ----------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("BCE loss")
plt.title(f"Training loss vs. iteration ({training_mode})")
plt.grid(True, linestyle="--", linewidth=0.5)

loss_fig_path = os.path.join("results", f"{training_mode}_training_loss.png")
plt.savefig(loss_fig_path, dpi=150)
plt.close()

print(f"Saved training loss plot to: {loss_fig_path}")

# ----------------------------------------
# Constellation before vs after training (overlaid)
# ----------------------------------------
trained_const_real = model._pusch_transmitter._points_r
trained_const_imag = model._pusch_transmitter._points_i

const_init = tf.complex(init_const_real, init_const_imag)
const_trained = tf.complex(trained_const_real, trained_const_imag)

# Make sure results directory exists
os.makedirs("results", exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 5))

pts_init = const_init.numpy()
pts_trained = const_trained.numpy()

ax.scatter(pts_init.real, pts_init.imag, s=25, marker="o", label="Initial")
ax.scatter(pts_trained.real, pts_trained.imag, s=25, marker="x", label="Trained")

ax.axhline(0.0, linewidth=0.5)
ax.axvline(0.0, linewidth=0.5)
ax.set_aspect("equal", "box")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.set_title(f"Constellation: initial vs trained ({training_mode})")
ax.set_xlabel("In-phase")
ax.set_ylabel("Quadrature")
ax.legend()

fig.tight_layout()
fig_path = os.path.join("results", f"constellations_overlaid_{training_mode}.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)

print(f"Saved constellation comparison plot to: {fig_path}")

print("Total time:", time.time() - start, "seconds")
