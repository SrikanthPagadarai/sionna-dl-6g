import tensorflow as tf
import numpy as np
import sionna as sn
import pickle
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

from demos.mimo_ofdm_neural_receiver.src.system import System  # noqa: E402

BATCH_SIZE = 32
EBN0_DB_MIN = -3
EBN0_DB_MAX = 7

# Common BER plotter utility (no figure shown)
ber_plots = sn.phy.utils.PlotBER("Advanced neural receiver")

# folder to read weights from / folder to save results to
outdir = Path("results")
outdir.mkdir(parents=True, exist_ok=True)

# Build eval model
eval_system = System(training=False, use_neural_rx=True, num_conv2d_filters=256)

# Build eval model & load weights
_ = eval_system(tf.constant(1, tf.int32), tf.fill([1], tf.constant(10.0, tf.float32)))
weight_file = outdir / "mimo-ofdm-neuralrx-weights"
with open(weight_file, "rb") as f:
    weights = pickle.load(f)
    eval_system.set_weights(weights)


# mc_fun bound to eval_system
@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([], tf.int32),  # scalar batch_size
        tf.TensorSpec([], tf.float32),  # scalar ebno_db
    ],
)
def mc_fun(batch_size, ebno_db):
    ebno_vec = tf.fill([batch_size], ebno_db)  # expand to shape (B,)
    return eval_system(batch_size, ebno_vec)


# Compute BER/BLER
ebno_vec = np.arange(EBN0_DB_MIN, EBN0_DB_MAX, 1)
ber, bler = ber_plots.simulate(
    mc_fun,
    ebno_dbs=ebno_vec,
    batch_size=BATCH_SIZE,
    max_mc_iter=2,
    num_target_block_errors=100,
    target_bler=1e-2,
    soft_estimates=True,
    show_fig=False,
)

# Save results (single-run)
outfile = outdir / "inference_results.npz"
np.savez(
    outfile,
    ebno_db=ebno_vec,
    ber=ber.numpy(),
    bler=bler.numpy(),
)
