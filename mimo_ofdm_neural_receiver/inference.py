import tensorflow as tf
import numpy as np
import sionna as sn
import pickle
from src.system import System
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

BATCH_SIZE = 32
EBN0_DB_MIN = -3
EBN0_DB_MAX = 7

# Common BER plotter utility (no figure shown)
ber_plots = sn.phy.utils.PlotBER("Advanced neural receiver")

# Storage for results across directions
all_directions = []
all_ber = []
all_bler = []

# folder to read weights from / folder to save results to
outdir = Path("results")
outdir.mkdir(parents=True, exist_ok=True)

# Evaluate the neural receiver for both directions one after the other
for direction in ["uplink", "downlink"]:
    # Parametrize eval_system's direction
    eval_system = System(training=False, use_neural_rx=True, direction=direction)

    # Build eval model & load weights specific to this direction
    _ = eval_system(tf.constant(1, tf.int32), tf.fill([1], tf.constant(10.0, tf.float32)))
    weight_file = outdir / f"mimo-ofdm-neuralrx-weights-{direction}"
    with open(weight_file, 'rb') as f:
        weights = pickle.load(f)
        eval_system.set_weights(weights)

    # Direction-scoped mc_fun to bind the current eval_system
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

    # Compute BER/BLER for this direction
    ebno_vec = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 1)
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

    all_directions.append(direction)
    all_ber.append(ber.numpy())
    all_bler.append(bler.numpy())

# Save ber, bler in a single file but separately for both directions (baseline-style)
outfile = outdir / "all_inference_results.npz"
np.savez(
    outfile,
    ebno_db=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 1),
    directions=np.array(all_directions),
    ber=np.array(all_ber),
    bler=np.array(all_bler),
)