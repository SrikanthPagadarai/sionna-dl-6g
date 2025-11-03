import tensorflow as tf
import numpy as np
import pickle
import os
import argparse
from src.system import System

# CLI: --direction {uplink,downlink,both}
parser = argparse.ArgumentParser(description="Train NeuralRx for uplink, downlink, or both (default).")
parser.add_argument("--direction", choices=["uplink", "downlink", "both"], default="both",
                    help="Which link direction(s) to train. Default: both.")
args = parser.parse_args()
directions = ["uplink", "downlink"] if args.direction == "both" else [args.direction]

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", tf.config.list_logical_devices('GPU'))

# Training config
BATCH_SIZE = 32
EBN0_DB_MIN = -3.
EBN0_DB_MAX = 7.
NUM_TRAINING_ITERATIONS = 10000

# Train NeuralRx in the selected direction(s)
os.makedirs("results", exist_ok=True)
for direction in directions:
    print(f"\n=== Training direction: {direction} ===")

    system = System(training=True, use_neural_rx=True, direction=direction, num_conv2d_filters=256)
    print(type(system), getattr(system, "__class__", None))

    # Warm-up call on the SAME instance that we'll train (creates variables deterministically)
    _ = system(tf.constant(BATCH_SIZE, tf.int32), tf.fill([BATCH_SIZE], 10.0))
    print("num trainables:", len(system.trainable_variables))

    optimizer = tf.keras.optimizers.Adam()

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec([], tf.int32),          # batch size (scalar)
            tf.TensorSpec([None], tf.float32),    # ebno vector (length == batch size)
        ],
    )
    def train_step(batch_size, ebno_vec):
        with tf.GradientTape() as tape:
            loss = system(batch_size, ebno_vec)

        weights = system.trainable_variables
        grads = tape.gradient(loss, weights)

        # Keep structure stable: replace None grads with 0-like tensors
        safe_grads = [g if g is not None else tf.zeros_like(w) for g, w in zip(grads, weights)]

        optimizer.apply_gradients(zip(safe_grads, weights))
        return loss

    # Training loop for this direction
    loss_history = []
    for i in range(NUM_TRAINING_ITERATIONS):
        ebno_db = tf.random.uniform(
            shape=[BATCH_SIZE],
            minval=EBN0_DB_MIN,
            maxval=EBN0_DB_MAX,
            dtype=tf.float32
        )
        loss = train_step(tf.constant(BATCH_SIZE, tf.int32), ebno_db)
        loss_value = float(loss.numpy())
        loss_history.append(loss_value)
        print(f"\r[{direction}] Step {i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss_value:.4f}",
              end="", flush=True)
    print(f"\n\nTraining complete for {direction}.")

    # Save separate loss history per direction
    loss_files_path = os.path.join("results", f"loss_{direction}.npy")
    np.save(loss_files_path, np.array(loss_history, dtype=np.float32))
    print(f"\nSaved loss history to {loss_files_path}.")

    # Save trained weights separately per direction
    weights = system.get_weights()
    weights_files_path = os.path.join("results", f"mimo-ofdm-neuralrx-weights-{direction}")
    with open(weights_files_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"\nSaved weights to {weights_files_path}.")
