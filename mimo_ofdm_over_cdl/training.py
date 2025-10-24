import tensorflow as tf
import pickle
from src.system import System
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", tf.config.list_logical_devices('GPU'))

# Training config
BATCH_SIZE = 32
EBN0_DB_MIN = -3.
EBN0_DB_MAX = 5.
NUM_TRAINING_ITERATIONS = 100

system = System(training=True, use_neural_rx=True, direction="uplink", perfect_csi=True)
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

# Training loop (Python only generates inputs; the heavy work is compiled above)
B_CONST = tf.constant(BATCH_SIZE, tf.int32)
for i in range(NUM_TRAINING_ITERATIONS):
    ebno_db = tf.random.uniform(shape=[BATCH_SIZE], minval=EBN0_DB_MIN, maxval=EBN0_DB_MAX, dtype=tf.float32)
    loss = train_step(B_CONST, ebno_db)
    print(f"\rStep {i}/{NUM_TRAINING_ITERATIONS}  Loss: {float(loss.numpy()):.4f}", end="", flush=True)

print("\n\nTraining complete.")

# Save trained weights
weights = system.get_weights()
with open('weights-ofdm-neuralrx', 'wb') as f:
    pickle.dump(weights, f)
