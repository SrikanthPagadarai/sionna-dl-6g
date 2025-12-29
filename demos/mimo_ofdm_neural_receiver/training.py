import tensorflow as tf
import numpy as np
import pickle
import os
import argparse

from src.system import System


# CLI
parser = argparse.ArgumentParser(description="Train NeuralRx.")
parser.add_argument(
    "--iterations", type=int, default=10000, help="Train for N more iterations"
)
parser.add_argument(
    "--fresh", action="store_true", help="Start fresh (ignore checkpoint)"
)
args = parser.parse_args()

# GPU setup
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", tf.config.list_logical_devices("GPU"))

# Training config
BATCH_SIZE = 32
EBN0_DB_MIN = -3.0
EBN0_DB_MAX = 7.0
ACCUMULATION_STEPS = 4

# Filesystem
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# System (no direction argument)
system = System(
    training=True,
    use_neural_rx=True,
    num_conv2d_filters=512,
    num_res_blocks=12,
)

# Warm-up (variable creation)
_ = system(tf.constant(BATCH_SIZE, tf.int32), tf.fill([BATCH_SIZE], 10.0))
print("num trainables:", len(system.trainable_variables))

# Optimizer / checkpoint
optimizer = tf.keras.optimizers.Adam()
rng = tf.random.Generator.from_seed(42)
checkpoint = tf.train.Checkpoint(
    model=system,
    optimizer=optimizer,
    rng=rng,
)

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


# Train step
@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([None], tf.float32),
    ],
)
def train_step(batch_size, ebno_vec):
    with tf.GradientTape() as tape:
        loss = system(batch_size, ebno_vec)
    grads = tape.gradient(loss, system.trainable_variables)
    grads = [
        g if g is not None else tf.zeros_like(w)
        for g, w in zip(grads, system.trainable_variables)
    ]
    return loss, grads


# Sanity: accumulation alignment
if start_iteration % ACCUMULATION_STEPS != 0:
    raise ValueError("start_iteration must be a multiple of ACCUMULATION_STEPS")

if target_iteration % ACCUMULATION_STEPS != 0:
    raise ValueError("target_iteration must be a multiple of ACCUMULATION_STEPS")

# Training loop
accumulated_grads = None
for i in range(start_iteration, target_iteration):
    ebno_db = rng.uniform([BATCH_SIZE], EBN0_DB_MIN, EBN0_DB_MAX, tf.float32)
    loss, grads = train_step(tf.constant(BATCH_SIZE, tf.int32), ebno_db)

    if accumulated_grads is None:
        accumulated_grads = [tf.Variable(g, trainable=False) for g in grads]
    else:
        for acc_g, g in zip(accumulated_grads, grads):
            acc_g.assign_add(g)

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

# Save state
checkpoint.save(os.path.join(ckpt_dir, "ckpt"))
open(os.path.join(ckpt_dir, "iter.txt"), "w").write(str(target_iteration))
np.save(os.path.join(ckpt_dir, "loss.npy"), loss_history)
np.save(
    os.path.join("results", "loss.npy"),
    np.array(loss_history, dtype=np.float32),
)

with open(os.path.join("results", "mimo-ofdm-neuralrx-weights"), "wb") as f:
    pickle.dump(system.get_weights(), f)
print("Saved checkpoints, loss history, and weights.")
