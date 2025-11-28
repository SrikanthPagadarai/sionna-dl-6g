import os
import pickle
import tensorflow as tf
from src.cir_manager import CIRManager
from src.system import PUSCHLinkE2E
from src.config import Config
import matplotlib.pyplot as plt

import time
start = time.time()

# Get configuration
_cfg = Config()
batch_size = _cfg.batch_size

# Build channel model
cir_manager = CIRManager()
channel_model = cir_manager.load_from_tfrecord()

# Instantiate and train the end-to-end system
ebno_db_test = tf.fill([batch_size], 10.0)
model = PUSCHLinkE2E(
    channel_model,
    perfect_csi=False,
    use_autoencoder=True,
    training=True
)
loss = model(batch_size, ebno_db_test)
print("  Initial forward-pass loss:", loss.numpy())
print("  Trainable variable count:", len(model.trainable_variables))
for v in model.trainable_variables[:5]:
    print("   ", v.name, v.shape)

# Snapshot initial constellation (before training)
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

# ----------------------------------------
# Training loop
# ----------------------------------------
ebno_db_min = -2.0
ebno_db_max = 10.0
training_batch_size = batch_size
num_training_iterations = 1000

optimizer_tx = tf.keras.optimizers.Adam(learning_rate=1e-3)   # 10x higher
optimizer_rx = tf.keras.optimizers.Adam(learning_rate=1e-4)

@tf.function(jit_compile=False)
def train_step():
    ebno_db = tf.random.uniform(
        shape=[training_batch_size],
        minval=ebno_db_min,
        maxval=ebno_db_max
    )
    with tf.GradientTape() as tape:
        loss = model(training_batch_size, ebno_db)
    
    tx_vars = model._pusch_transmitter.trainable_variables
    rx_vars = model._pusch_receiver.trainable_variables
    
    grads = tape.gradient(loss, tx_vars + rx_vars)
    grads_tx = grads[:len(tx_vars)]
    grads_rx = grads[len(tx_vars):]
    
    optimizer_tx.apply_gradients(zip(grads_tx, tx_vars))
    optimizer_rx.apply_gradients(zip(grads_rx, rx_vars))
    
    return loss

# store loss values for plotting
loss_history = []


for i in range(num_training_iterations):
    loss = train_step()
    loss_value = float(loss.numpy())
    loss_history.append(loss_value)
    print(
        'Iteration {}/{}  BCE: {:.4f}'.format(
            i + 1, num_training_iterations, loss_value
        ),
        end='\r',
        flush=True
    )

    # Save weights intermittently
    if (i % 200) == 0:
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join(
            "results",
            f"PUSCH_autoencoder_weights_iter_{i}"
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

print()  # newline after the loop


# Save weights
os.makedirs("results", exist_ok=True)
weights_path = os.path.join("results", "PUSCH_autoencoder_weights_conventional_training")

weights_dict = {
    'tx_weights': [v.numpy() for v in model._pusch_transmitter.trainable_variables],
    'rx_weights': [v.numpy() for v in model._pusch_receiver.trainable_variables],
    'tx_names': [v.name for v in model._pusch_transmitter.trainable_variables],
    'rx_names': [v.name for v in model._pusch_receiver.trainable_variables],
}
with open(weights_path, 'wb') as f:
    pickle.dump(weights_dict, f)

print(f"Saved {len(weights_dict['tx_weights'])} TX and {len(weights_dict['rx_weights'])} RX weight arrays")

# ----------------------------------------
# Plot training loss vs iteration
# ----------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("BCE loss")
plt.title("Training loss vs. iteration")
plt.grid(True, linestyle="--", linewidth=0.5)

loss_fig_path = os.path.join("results", "training_loss.png")
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
    label='Trained'
)

ax.axhline(0.0, linewidth=0.5)
ax.axvline(0.0, linewidth=0.5)
ax.set_aspect("equal", "box")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.set_title("Constellation: initial vs trained")
ax.set_xlabel("In-phase")
ax.set_ylabel("Quadrature")
ax.legend()

fig.tight_layout()
fig_path = os.path.join("results", "constellations_overlaid.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)

print(f"Saved constellation comparison plot to: {fig_path}")

print("Total time:", time.time() - start, "seconds")
