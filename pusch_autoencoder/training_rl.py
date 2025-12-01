import os, sys
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.cir_manager import CIRManager
from src.system import PUSCHLinkE2E
from src.config import Config

# --------------------------------------------------
# Configuration & channel model
# --------------------------------------------------
_cfg = Config()
batch_size = _cfg.batch_size  # same as in training.py

cir_manager = CIRManager()
channel_model = cir_manager.load_from_tfrecord(group_for_mumimo=True)

# --------------------------------------------------
# RL training configuration (analogous to cell #9)
# --------------------------------------------------
# SNR range used for training
ebno_db_min = -2.0
ebno_db_max = 10.0

# Training batch size (reuse system batch_size)
training_batch_size = batch_size

# Number of alternating RL iterations and RX fine-tuning iterations
num_training_iterations_rl_alt = 10000         # you can increase to 7000 later

# --------------------------------------------------
# Build RL-based end-to-end model
# --------------------------------------------------
model = PUSCHLinkE2E(
    channel_model=channel_model,
    perfect_csi=False,         # same choice as in training.py
    use_autoencoder=True,
    training=True,
    training_mode="rl",        # <--- RL mode
)

# Quick sanity check forward pass (also initializes variables)
ebno_db_test = tf.fill([batch_size], 10.0)
tx_loss_test, rx_loss_test = model(
    batch_size,
    ebno_db_test
)
print("Quick sanity check ===")
print(f"  Initial TX loss (RL): {tx_loss_test.numpy():.4f}")
print(f"  Initial RX loss (BCE): {rx_loss_test.numpy():.4f}")
print(f"  Trainable TX vars: {len(model._pusch_transmitter.trainable_variables)}")
print(f"  Trainable RX vars: {len(model._pusch_receiver.trainable_variables)}")

# Snapshot initial constellation (before RL training)
init_const_real = tf.identity(model._pusch_transmitter._points_r)
init_const_imag = tf.identity(model._pusch_transmitter._points_i)

# ---- Gradient sanity check: one step in eager mode ----
print("\n=== Single-step gradient sanity check ===")

dbg_batch_size = 4
dbg_ebno = tf.fill([dbg_batch_size], 10.0)  # arbitrary Eb/N0

tx_vars = model._pusch_transmitter.trainable_variables
rx_vars = model._pusch_receiver.trainable_variables
all_vars = tx_vars + rx_vars

with tf.GradientTape() as tape:
    loss_dbg = model(dbg_batch_size, dbg_ebno)

all_grads = tape.gradient(loss_dbg, all_vars)

# Split gradients back into TX and RX parts
grads_tx = all_grads[:len(tx_vars)]
grads_rx = all_grads[len(tx_vars):]

print("\nTransmitter gradients:")
for v, g in zip(tx_vars, grads_tx):
    g_norm = 0.0 if g is None else float(tf.norm(g).numpy())
    print(f"  {v.name:40s} grad_norm = {g_norm:.3e}")

print("\nReceiver gradients:")
for v, g in zip(rx_vars, grads_rx):
    g_norm = 0.0 if g is None else float(tf.norm(g).numpy())
    print(f"  {v.name:40s} grad_norm = {g_norm:.3e}")

print("=== End gradient sanity check ===\n")

# ----------
# Optimizers
# ----------
optimizer_tx = tf.keras.optimizers.Adam(learning_rate=5e-4)
optimizer_rx = tf.keras.optimizers.Adam(learning_rate=1e-3)

# --------------------------------------------------
# One TX training step (RL)
# --------------------------------------------------
@tf.function(jit_compile=False)
def train_tx():
    # Sample a batch of SNRs
    ebno_db = tf.random.uniform(
        shape=[training_batch_size],
        minval=ebno_db_min,
        maxval=ebno_db_max
    )

    with tf.GradientTape() as tape:
        # Keep only the TX loss; perturbation variance enables RL exploration
        tx_loss, _ = model(
            training_batch_size,
            ebno_db,
        )

    tx_vars = model._pusch_transmitter.trainable_variables
    grads = tape.gradient(
        tx_loss,
        tx_vars,
        unconnected_gradients=tf.UnconnectedGradients.ZERO
    )
    optimizer_tx.apply_gradients(zip(grads, tx_vars))
    return tx_loss

# --------------------------------------------------
# One RX training step (conventional BCE)
# --------------------------------------------------
@tf.function(jit_compile=False)
def train_rx():
    # Sample a batch of SNRs
    ebno_db = tf.random.uniform(
        shape=[training_batch_size],
        minval=ebno_db_min,
        maxval=ebno_db_max
    )

    with tf.GradientTape() as tape:
        # Keep only the RX loss; no perturbation is added here
        _, rx_loss = model(
            training_batch_size,
            ebno_db,
        )

    rx_vars = model._pusch_receiver.trainable_variables
    grads = tape.gradient(
        rx_loss,
        rx_vars,
        unconnected_gradients=tf.UnconnectedGradients.ZERO
    )
    optimizer_rx.apply_gradients(zip(grads, rx_vars))
    return rx_loss

# --------------------------------------------------
# Alternating RL training loop (TX/RX)
# --------------------------------------------------
rx_loss_history_alt = []
tx_loss_history_alt = []

print("Starting alternating RL training...")
for i in range(num_training_iterations_rl_alt):
    # Keep the receiver "ahead" of the transmitter:
    # perform several RX steps per TX step
    rx_loss_val = tf.constant(0.0, dtype=tf.float32)
    for _ in range(5):
        rx_loss_val = train_rx()
    tx_loss_val = train_tx()

    # Log
    rx_loss_history_alt.append(rx_loss_val.numpy())
    tx_loss_history_alt.append(tx_loss_val.numpy())

    print("Alt Iter {}/{}  RX_BCE {:.4f}  TX_loss {:.4f}".format(i, num_training_iterations_rl_alt,rx_loss_val.numpy(),tx_loss_val.numpy()),end="\r",flush=True)

    # Save weights intermittently
    if (i % 1000) == 0:
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join(
            "results",
            f"PUSCH_autoencoder_weights_rl_iter_{i}"
        )
        weights_dict = {
            'tx_weights': [v.numpy() for v in model._pusch_transmitter.trainable_variables],
            'rx_weights': [v.numpy() for v in model._pusch_receiver.trainable_variables],
            'tx_names': [v.name for v in model._pusch_transmitter.trainable_variables],
            'rx_names': [v.name for v in model._pusch_receiver.trainable_variables],
        }
        with open(save_path, 'wb') as f:
            pickle.dump(weights_dict, f)
        print(f"\n[Checkpoint] Saved weights at iteration {i} -> {save_path}")
print()  # newline after alternating phase

# --------------------------------------------------
# Save weights and training curves
# --------------------------------------------------
os.makedirs("results", exist_ok=True)
weights_path = os.path.join("results", "PUSCH_autoencoder_weights_rl_training")

weights_dict = {
    'tx_weights': [v.numpy() for v in model._pusch_transmitter.trainable_variables],
    'rx_weights': [v.numpy() for v in model._pusch_receiver.trainable_variables],
    'tx_names': [v.name for v in model._pusch_transmitter.trainable_variables],
    'rx_names': [v.name for v in model._pusch_receiver.trainable_variables],
}
with open(weights_path, 'wb') as f:
    pickle.dump(weights_dict, f)

print(f"Saved RL-trained weights to: {weights_path}")

# Save loss histories as .npy for later analysis
np.save(
    os.path.join("results", "rl_rx_loss_alt.npy"),
    np.array(rx_loss_history_alt, dtype=np.float32)
)
np.save(
    os.path.join("results", "rl_tx_loss_alt.npy"),
    np.array(tx_loss_history_alt, dtype=np.float32)
)

# --------------------------------------------------
# Optional: simple plot of RX loss vs iteration
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(rx_loss_history_alt, label="RX BCE (alt phase)")
plt.xlabel("Logged iteration")
plt.ylabel("RX BCE loss")
plt.title("RL Training: Receiver Loss")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()

loss_fig_path = os.path.join("results", "rl_training_rx_loss.png")
plt.savefig(loss_fig_path, dpi=150)
plt.close()
print(f"Saved RL RX-loss plot to: {loss_fig_path}")

# --------------------------------------------------
# Plot TX loss vs iteration (separate figure)
# --------------------------------------------------
plt.figure(figsize=(6, 4))

if tx_loss_history_alt:
    plt.plot(tx_loss_history_alt,label="TX Loss (alt phase)")
plt.xlabel("Logged iteration")
plt.ylabel("TX Loss")
plt.title("RL Training: Transmitter Loss")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()

tx_loss_fig_path = os.path.join("results", "rl_training_tx_loss.png")
plt.savefig(tx_loss_fig_path, dpi=150)
plt.close()

print(f"Saved RL TX-loss plot to: {tx_loss_fig_path}")

# ----------------------------------------
# Constellation before vs after RL training (overlaid)
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

ax.scatter(
    pts_init.real,
    pts_init.imag,
    s=25,
    marker='o',
    label='Initial'
)
ax.scatter(
    pts_trained.real,
    pts_trained.imag,
    s=25,
    marker='x',
    label='RL-trained'
)

ax.axhline(0.0, linewidth=0.5)
ax.axvline(0.0, linewidth=0.5)
ax.set_aspect("equal", "box")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.set_title("Constellation: initial vs RL-trained")
ax.set_xlabel("In-phase")
ax.set_ylabel("Quadrature")
ax.legend()

fig.tight_layout()
fig_path = os.path.join("results", "constellations_overlaid_rl.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)

print(f"Saved RL constellation comparison plot to: {fig_path}")

