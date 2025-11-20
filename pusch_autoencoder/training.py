import os
import pickle
import tensorflow as tf
from src.cir_manager import CIRManager
from src.system import PUSCHLinkE2E
from src.config import Config

# Get configuration
_cfg = Config()
batch_size = _cfg.batch_size

# Build channel model
cir_manager = CIRManager()
channel_model = cir_manager.load_from_tfrecord()

# channel_model = (a, tau)
# channel_model = cir_manager.build_channel_model()

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

### training
# parameters
ebno_db_min = -2.0
ebno_db_max = 10.0
training_batch_size = batch_size
num_training_iterations = 100

# Optimizer used to apply gradients
optimizer = tf.keras.optimizers.Adam()

@tf.function(jit_compile=False)
def train_step():
    # Sampling a batch of SNRs
    ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
    # Forward pass
    with tf.GradientTape() as tape:
        loss = model(training_batch_size,  ebno_db)
    # Computing and applying gradients
    weights = model.trainable_variables
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    return loss

for i in range(num_training_iterations):
    loss = train_step()
    # Printing periodically the progress
    if i % 10 == 0:
        print('Iteration {}/{}  BCE: {:.4f}'.format(i, num_training_iterations, loss.numpy()), end='\r',flush=True)

# Save weights
os.makedirs("results", exist_ok=True)
weights_path = os.path.join("results", "PUSCH_autoencoder_weights_conventional_training")
weights = model.get_weights()
with open(weights_path, 'wb') as f:
    pickle.dump(weights, f)