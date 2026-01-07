# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Inference script for evaluating the trained neural receiver.

This script loads trained weights and evaluates BER/BLER performance
across an Eb/N0 range using Sionna's Monte Carlo simulation framework.

Pipeline
--------
1. Load trained model weights from ``results/mimo-ofdm-neuralrx-weights``
2. Configure evaluation system (inference mode, neural Rx enabled)
3. Run Monte Carlo simulation across Eb/N0 range
4. Save BER/BLER results to ``results/inference_results.npz``

Usage
-----
Run after training to evaluate performance::

    python inference.py

Prerequisites
-------------
- Trained weights must exist at ``results/mimo-ofdm-neuralrx-weights``
- Run ``training.py`` first if weights don't exist

Output Files
------------
``results/inference_results.npz``:
    NumPy archive containing:
    - ``ebno_db``: Eb/N0 values tested (dB)
    - ``ber``: Bit error rate at each Eb/N0
    - ``bler``: Block error rate at each Eb/N0

Note
----
The Monte Carlo simulation uses Sionna's ``PlotBER.simulate()`` which:
- Runs multiple iterations per SNR point until convergence
- Stops early if target BLER or block error count is reached
- Returns both BER and BLER for comprehensive analysis

The ``mc_fun`` wrapper expands scalar Eb/N0 to vector form because
the System expects per-sample SNR values (enables mixed-SNR batches
during training, though inference uses uniform SNR per batch).
"""

import tensorflow as tf
import numpy as np
import sionna as sn
import pickle
from pathlib import Path
import matplotlib

# Use non-interactive backend for headless environments (servers, CI)
matplotlib.use("Agg")

from demos.mimo_ofdm_neural_receiver.src.system import System  # noqa: E402

# =============================================================================
# Evaluation Configuration
# =============================================================================
BATCH_SIZE = 32

# Eb/N0 range for evaluation (should match or exceed training range)
EBN0_DB_MIN = -3
EBN0_DB_MAX = 7

# =============================================================================
# Output Directory and BER Plotter
# =============================================================================
# PlotBER provides Monte Carlo simulation with automatic convergence detection
ber_plots = sn.phy.utils.PlotBER("Advanced neural receiver")

# Results directory (same as training for weight loading)
DEMO_DIR = Path(__file__).parent
outdir = DEMO_DIR / "results"
outdir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Model Setup
# Build evaluation system in inference mode (full encoding/decoding enabled)
# =============================================================================
eval_system = System(training=False, use_neural_rx=True, num_conv2d_filters=256)

# Warm-up call to create variables before loading weights
# Shape [1] is sufficient; just need to trigger variable instantiation
_ = eval_system(tf.constant(1, tf.int32), tf.fill([1], tf.constant(10.0, tf.float32)))

# =============================================================================
# Load Trained Weights
# Weights are pickled list from system.get_weights() during training
# =============================================================================
weight_file = outdir / "mimo-ofdm-neuralrx-weights"
with open(weight_file, "rb") as f:
    weights = pickle.load(f)
    eval_system.set_weights(weights)


# =============================================================================
# Monte Carlo Function
# Wrapper to adapt System's vector-SNR interface to PlotBER's scalar interface
# =============================================================================
@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([], tf.int32),  # scalar batch_size
        tf.TensorSpec([], tf.float32),  # scalar ebno_db
    ],
)
def mc_fun(batch_size, ebno_db):
    """
    Monte Carlo evaluation function for PlotBER.simulate().

    Parameters
    ----------
    batch_size : tf.Tensor, int32, scalar
        Number of channel realizations per Monte Carlo iteration.

    ebno_db : tf.Tensor, float32, scalar
        Eb/N0 in dB for this evaluation point.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        (b, b_hat) tuple where:
        - b: transmitted information bits
        - b_hat: decoded information bits

    Note
    ----
    PlotBER.simulate() calls this function repeatedly at each SNR point,
    accumulating bit/block errors until convergence criteria are met.
    The function broadcasts scalar Eb/N0 to vector form for System compatibility.
    """
    # Expand scalar Eb/N0 to vector (System expects per-sample SNR)
    ebno_vec = tf.fill([batch_size], ebno_db)
    return eval_system(batch_size, ebno_vec)


# =============================================================================
# Run Monte Carlo Simulation
# =============================================================================
# Eb/N0 points to evaluate (1 dB steps)
ebno_vec = np.arange(EBN0_DB_MIN, EBN0_DB_MAX, 1)

# simulate() returns (ber, bler) tensors
# - max_mc_iter: maximum iterations per SNR point
# - num_target_block_errors: stop early if this many block errors accumulated
# - target_bler: stop early if BLER drops below this threshold
# - soft_estimates: True because System returns float tensors
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

# =============================================================================
# Save Results
# =============================================================================
outfile = outdir / "inference_results.npz"
np.savez(
    outfile,
    ebno_db=ebno_vec,
    ber=ber.numpy(),
    bler=bler.numpy(),
)
