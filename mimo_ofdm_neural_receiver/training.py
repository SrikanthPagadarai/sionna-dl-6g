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
# Add iterations argument
parser.add_argument("--iterations", type=int, default=10000,
                    help="Train for this many iterations (default: 10000)")
parser.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint")
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

# Train NeuralRx in the selected direction(s)
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)  # checkpoint directory

for direction in directions:
    print(f"\n=== Training direction: {direction} ===")
    
    system = System(training=True, use_neural_rx=True, direction=direction, num_conv2d_filters=256)
    print(type(system), getattr(system, "__class__", None))
    
    # Warm-up call on the SAME instance that we'll train (creates variables deterministically)
    _ = system(tf.constant(BATCH_SIZE, tf.int32), tf.fill([BATCH_SIZE], 10.0))
    print("num trainables:", len(system.trainable_variables))
    
    optimizer = tf.keras.optimizers.Adam()
    
    # Checkpoint setup
    checkpoint = tf.train.Checkpoint(
        model=system,
        optimizer=optimizer,
        total_iterations=tf.Variable(0, trainable=False, dtype=tf.int64)
    )
    manager = tf.train.CheckpointManager(checkpoint, f"checkpoints/{direction}", max_to_keep=2)
    
    # Load checkpoint if resuming
    start_iteration = 0
    loss_history = []
    
    if args.resume and manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        start_iteration = int(checkpoint.total_iterations.numpy())
        # Load previous loss history
        loss_path = f"checkpoints/{direction}/loss_history.npy"
        if os.path.exists(loss_path):
            loss_history = np.load(loss_path).tolist()
        print(f"Resumed from iteration {start_iteration}")
    
    # Calculate target iteration for this run
    target_iteration = start_iteration + args.iterations
    print(f"Training from iteration {start_iteration} to {target_iteration}")
    
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
    for i in range(start_iteration, target_iteration):
        tf.random.set_seed(42 + i)
        
        ebno_db = tf.random.uniform(
            shape=[BATCH_SIZE],
            minval=EBN0_DB_MIN,
            maxval=EBN0_DB_MAX,
            dtype=tf.float32
        )
        loss = train_step(tf.constant(BATCH_SIZE, tf.int32), ebno_db)
        loss_value = float(loss.numpy())
        loss_history.append(loss_value)
        print(f"\r[{direction}] Step {i+1}/{target_iteration}  Loss: {loss_value:.4f}", end="", flush=True)
    
    print(f"\n\nCompleted {args.iterations} iterations for {direction}.")
    
    # Save checkpoint after training batch
    checkpoint.total_iterations.assign(target_iteration)
    manager.save()
    np.save(f"checkpoints/{direction}/loss_history.npy", loss_history)
    print(f"Checkpoint saved at iteration {target_iteration}")
    
    # Save final results (always save to results/ with full history)
    loss_files_path = os.path.join("results", f"loss_{direction}.npy")
    np.save(loss_files_path, np.array(loss_history, dtype=np.float32))
    print(f"Saved loss history to {loss_files_path}.")
    
    # Save trained weights
    weights = system.get_weights()
    weights_files_path = os.path.join("results", f"mimo-ofdm-neuralrx-weights-{direction}")
    with open(weights_files_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Saved weights to {weights_files_path}.")
    
    print(f"\nTotal iterations completed: {target_iteration}")