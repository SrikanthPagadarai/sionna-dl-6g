# train.py
import tensorflow as tf
import numpy as np
import sionna as sn
import pickle
from src.system import System
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
from pathlib import Path

# Training config
BATCH_SIZE = 64
EBN0_DB_MIN = -3
EBN0_DB_MAX = 5
NUM_TRAINING_ITERATIONS = 50

system = System(training=True, use_neural_rx=True, direction="uplink", perfect_csi=True)
print(type(system), getattr(system, "__class__", None))

# Warm-up call on the SAME instance that we'll train (creates variables)
_ = system(tf.constant(BATCH_SIZE, tf.int32), tf.fill([BATCH_SIZE], 10.0))
print("num trainables:", len(system.trainable_variables))

optimizer = tf.keras.optimizers.Adam()

# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    ebno_db = tf.random.uniform(shape=[BATCH_SIZE], minval=EBN0_DB_MIN, maxval=EBN0_DB_MAX)

    with tf.GradientTape() as tape:
        loss = system(tf.constant(BATCH_SIZE, tf.int32), ebno_db)

    # fetch variables after forward (they’re already created; warmup call did that)
    weights = system.trainable_variables

    # compute grads
    grads = tape.gradient(loss, weights)

    # filter out any None grads to avoid optimizer error if some vars weren’t used
    grads_and_vars = [(g, w) for g, w in zip(grads, weights) if g is not None]
    if not grads_and_vars:
        raise RuntimeError(
            "No gradients to apply. The loss is likely disconnected from all trainable variables."
        )

    optimizer.apply_gradients(grads_and_vars)

    # Progress
    print(f"Step {i}/{NUM_TRAINING_ITERATIONS}  Loss: {float(loss.numpy()):.4f}")

print("\nTraining complete.")

# Save trained weights
weights = system.get_weights()
with open('weights-ofdm-neuralrx', 'wb') as f:
    pickle.dump(weights, f)

# Evaluation: instantiate fresh inference model, load weights, plot BER
eval_system = System(training=False, use_neural_rx=True)

# Build eval model & load weights
_ = eval_system(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))
with open('weights-ofdm-neuralrx', 'rb') as f:
    weights = pickle.load(f)
    eval_system.set_weights(weights)

# Compute and plot BER
outdir = Path("sim/results")
outdir.mkdir(parents=True, exist_ok=True)

# Start a fresh figure so we know what we're saving
plt.close("all")
plt.figure(figsize=(6, 4))

ber_plots = sn.phy.utils.PlotBER("Advanced neural receiver")

ber_plots.simulate(
    eval_system,
    ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
    batch_size=BATCH_SIZE,
    num_target_block_errors=100,
    legend="Neural Receiver",
    soft_estimates=True,
    max_mc_iter=2,
    show_fig=True,
)

# Make sure matplotlib renders before saving (Agg backend)
plt.tight_layout()
plt.draw()
png_path = outdir / "ber_neural.png"
plt.savefig(png_path, dpi=180)
plt.close()
print(f"Saved BER plot to: {png_path}")
