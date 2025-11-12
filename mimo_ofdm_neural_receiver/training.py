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
parser.add_argument("--iterations", type=int, default=10000, help="Train for N more iterations")
parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore checkpoint)")
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
os.makedirs("checkpoints", exist_ok=True)

for direction in directions:
    print(f"\n=== Training direction: {direction} ===")
    
    system = System(training=True, use_neural_rx=True, direction=direction, num_conv2d_filters=512, num_res_blocks = 12)
    print(type(system), getattr(system, "__class__", None))
    
    # Warm-up call on the SAME instance that we'll train (creates variables deterministically)
    _ = system(tf.constant(BATCH_SIZE, tf.int32), tf.fill([BATCH_SIZE], 10.0))
    print("num trainables:", len(system.trainable_variables))
    
    optimizer = tf.keras.optimizers.Adam()
    
    # checkpoint
    rng = tf.random.Generator.from_seed(42)
    checkpoint = tf.train.Checkpoint(model=system, optimizer=optimizer, rng=rng)
    
    start_iteration = 0
    loss_history = []
    if not args.fresh and tf.train.latest_checkpoint(f"checkpoints/{direction}"):
        checkpoint.restore(tf.train.latest_checkpoint(f"checkpoints/{direction}"))
        start_iteration = int(open(f"checkpoints/{direction}/iter.txt").read())
        loss_history = np.load(f"checkpoints/{direction}/loss.npy").tolist()
        print(f"Resumed from iteration {start_iteration}")
    
    target_iteration = start_iteration + args.iterations
    print(f"Training from {start_iteration} to {target_iteration}")
    
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
        grads = tape.gradient(loss, system.trainable_variables)
        safe_grads = [g if g is not None else tf.zeros_like(w) 
                    for g, w in zip(grads, system.trainable_variables)]
        return loss, safe_grads
    
    # Validate iteration alignment    
    ACCUMULATION_STEPS = 4
    if start_iteration % ACCUMULATION_STEPS != 0:
        raise ValueError(
            f"start_iteration ({start_iteration}) must be a multiple of "
            f"ACCUMULATION_STEPS ({ACCUMULATION_STEPS}). "
            f"Use --fresh to start from 0, or train to a multiple of {ACCUMULATION_STEPS}."
        )

    if target_iteration % ACCUMULATION_STEPS != 0:
        suggested_target = (target_iteration // ACCUMULATION_STEPS) * ACCUMULATION_STEPS
        suggested_iterations = suggested_target - start_iteration
        raise ValueError(
            f"target_iteration ({target_iteration}) must be a multiple of "
            f"ACCUMULATION_STEPS ({ACCUMULATION_STEPS}). "
            f"Use --iterations {suggested_iterations} instead (target: {suggested_target})."
        )
    
    # Training loop for this direction
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
        print(f"\r[{direction}] Step {i}/{target_iteration}  Loss: {loss_value:.4f}",
            end="", flush=True)
    
    print(f"\n\nCompleted batch for {direction}.")
    
    # Save checkpoint
    checkpoint.save(f"checkpoints/{direction}/ckpt")
    open(f"checkpoints/{direction}/iter.txt", "w").write(str(target_iteration))
    np.save(f"checkpoints/{direction}/loss.npy", loss_history)
    
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
