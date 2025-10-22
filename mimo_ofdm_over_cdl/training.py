import tensorflow as tf
import pickle
from src.system import System

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
        raise RuntimeError("No gradients to apply. The loss is likely disconnected from all trainable variables.")

    optimizer.apply_gradients(grads_and_vars)

    # Progress
    print(f"\rStep {i}/{NUM_TRAINING_ITERATIONS}  Loss: {float(loss.numpy()):.4f}", end='', flush=True)
print()
print("\nTraining complete.")

# Save trained weights
weights = system.get_weights()
with open('weights-ofdm-neuralrx', 'wb') as f:
    pickle.dump(weights, f)

