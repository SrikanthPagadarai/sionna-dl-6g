import tensorflow as tf
import numpy as np
import sionna as sn
import pickle
from src.system import System
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCH_SIZE = 32
EBN0_DB_MIN = -3
EBN0_DB_MAX = 5

# Evaluation: instantiate fresh inference model, load weights, plot BER
eval_system = System(training=False, use_neural_rx=True)

# Build eval model & load weights
_ = eval_system(tf.constant(1, tf.int32), tf.fill([1], tf.constant(10.0, tf.float32)))
with open('weights-ofdm-neuralrx', 'rb') as f:
    weights = pickle.load(f)
    eval_system.set_weights(weights)

# ---- Minimal wrapper for PlotBER: fix kwarg name to 'ebno_db' ----
@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([], tf.int32),    # scalar batch_size
        tf.TensorSpec([], tf.float32),  # scalar ebno_db
    ],
)
def mc_fun(batch_size, ebno_db):
    ebno_vec = tf.fill([batch_size], ebno_db)  # expand to shape (BATCH_SIZE,)
    return eval_system(batch_size, ebno_vec)   # reuse vector-SNR path

# Compute and plot BER
outdir = Path("sim/results")
outdir.mkdir(parents=True, exist_ok=True)

# Start a fresh figure so we know what we're saving
plt.close("all")
plt.figure(figsize=(6, 4))

ber_plots = sn.phy.utils.PlotBER("Advanced neural receiver")

ber_plots.simulate(
    mc_fun,
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
