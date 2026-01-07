# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Visualization script for PUSCH autoencoder results.

This script generates publication-quality plots comparing the trained
autoencoder against baseline LMMSE performance. It produces three types
of visualizations for each antenna configuration (16 and 32):

1. **BLER Comparison**: Baseline (perfect/imperfect CSI) vs trained autoencoder
2. **Training Loss Curve**: BCE loss evolution during training (iterations 500-5000)
3. **Constellation Evolution**: How constellation geometry changes during training

Prerequisites
-------------
Before running this script, you must have generated:

1. ``results/baseline_results_ant{num_bs_ant}.npz`` - from baseline.py
2. ``results/inference_results_ant{num_bs_ant}.npz`` - from inference.py
3. ``results/training_loss_ant{num_bs_ant}.npy`` - from training.py
4. ``results/PUSCH_autoencoder_weights_ant{num_bs_ant}`` - from training.py
5. Checkpoint files at iterations 1000, 2000, 3000, 4000 (optional)

Output
------
All plots are saved to ``results/`` directory:

- ``bler_plot_{num_bs}bs_{num_bs_ant}bs_ant_x_{num_ue}ue_{num_ue_ant}ue_ant.png``: BLER comparison
- ``ber_plot_{num_bs}bs_{num_bs_ant}bs_ant_x_{num_ue}ue_{num_ue_ant}ue_ant.png``: BER comparison
- ``training_loss_ant{num_bs_ant}.png``: Loss curve with best iteration marked
- ``constellation_normalized_ant{num_bs_ant}.png``: Final trained vs standard 16-QAM
- ``constellation_iter_{N}_ant{num_bs_ant}.png``: Intermediate constellation snapshots

Usage
-----
Run from the repository root after baseline.py, training.py, and inference.py::

    python -m demos.pusch_autoencoder.plots

Interpretation Guide
--------------------
**BLER Plot**:

- Perfect CSI curve shows theoretical upper bound
- Gap between perfect and imperfect CSI is the "CSI penalty"
- Autoencoder should approach or exceed imperfect CSI baseline

**BER Plot**:

- Shows bit-level error rate (finer granularity than BLER)
- Useful for understanding uncoded performance
- Lower BER generally correlates with lower BLER after decoding

**Constellation Plot**:

- Significant point movement indicates channel adaptation
- Points should remain roughly symmetric (balanced I/Q)
- Minimum distance between points affects high-SNR performance
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from demos.pusch_autoencoder.src.config import Config

# get directory name of file
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Constellation Utilities
# =============================================================================
def normalize_constellation(points_r, points_i):
    """
    Apply centering and power normalization to constellation points.

    This matches the normalization in ``PUSCHTrainableTransmitter.get_normalized_constellation()``,
    ensuring consistent visualization regardless of how raw variables have drifted.

    Parameters
    ----------
    points_r : np.ndarray
        Real parts of constellation points.
    points_i : np.ndarray
        Imaginary parts of constellation points.

    Returns
    -------
    np.ndarray, complex128
        Normalized constellation with zero mean and unit average power.
    """
    points = points_r + 1j * points_i

    # Center: remove DC offset for balanced I/Q
    points = points - np.mean(points)

    # Normalize: scale to unit average power for fair comparison
    energy = np.mean(np.abs(points) ** 2)
    points = points / np.sqrt(energy)

    return points


def standard_16qam():
    """
    Generate standard Gray-coded 16-QAM constellation.

    Returns
    -------
    np.ndarray, complex128
        16 constellation points with unit average power.

    Notes
    -----
    Standard 16-QAM uses levels {-3, -1, +1, +3} on each axis,
    normalized by sqrt(10) to achieve unit average power:

    E[|x|^2] = (2*9 + 2*1 + 2*9 + 2*1) * 4 / 16 / 10 = 1
    """
    levels = np.array([-3, -1, 1, 3])
    real, imag = np.meshgrid(levels, levels)
    points = (real.flatten() + 1j * imag.flatten()) / np.sqrt(10)
    return points


# Reference constellation for comparison
init_const = standard_16qam()


# =============================================================================
# Parametrized BS Antenna Configurations
# =============================================================================
NUM_BS_ANT_VALUES = [16, 32]

for num_bs_ant in NUM_BS_ANT_VALUES:
    print(f"\n{'='*60}")
    print(f"Generating plots for num_bs_ant = {num_bs_ant}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Configuration
    # =========================================================================
    _cfg = Config(num_bs_ant=num_bs_ant)
    batch_size = _cfg.batch_size
    num_ue = _cfg.num_ue
    num_ue_ant = _cfg.num_ue_ant
    num_bs = _cfg.num_bs

    ant_suffix = f"_ant{num_bs_ant}"

    # =========================================================================
    # Load Results Data
    # =========================================================================
    # Baseline results (from baseline.py)
    baseline_path = os.path.join(
        DEMO_DIR, "results", f"baseline_results{ant_suffix}.npz"
    )
    if not os.path.exists(baseline_path):
        print(
            f"Warning: {baseline_path} not found. "
            f"Run `python3 baseline.py` first. Skipping {num_bs_ant} antenna plots."
        )
        continue

    # Autoencoder inference results (from inference.py)
    inference_path = os.path.join(
        DEMO_DIR, "results", f"inference_results{ant_suffix}.npz"
    )
    if not os.path.exists(inference_path):
        print(
            f"Warning: {inference_path} not found. "
            f"Run `python3 inference.py` first. Skipping {num_bs_ant} antenna plots."
        )
        continue

    baseline_data = np.load(baseline_path)
    inference_data = np.load(inference_path)

    ebno_db = baseline_data["ebno_db"]
    bler = baseline_data[
        "bler"
    ]  # Shape: [2, num_snr_points] - [perfect_csi, imperfect_csi]
    inference_bler = inference_data["bler"]

    # =========================================================================
    # BLER Comparison Plot
    # =========================================================================
    # Compare baseline LMMSE (perfect/imperfect CSI) against trained autoencoder
    plt.figure()
    for idx, csi_label in enumerate(["(Perfect CSI)", "(Imperfect CSI)"]):
        plt.semilogy(
            ebno_db,
            bler[idx],
            marker="o",
            linestyle="-",
            label=f"Conventional Detector {csi_label}",
        )
    plt.semilogy(
        ebno_db,
        inference_bler,
        marker="o",
        linestyle="-",
        label="Neural MIMO Detector (Imperfect CSI)",
    )
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("BLER")
    plt.title(f"PUSCH - BLER vs Eb/N0 ({num_bs_ant} BS Antennas)")
    plt.grid(True, which="both")
    plt.legend()

    outfile = os.path.join(
        DEMO_DIR,
        "results",
        f"bler_plot_{num_bs}bs_{num_bs_ant}bs_ant_x_{num_ue}ue_{num_ue_ant}ue_ant.png",
    )
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved BLER plot to {outfile}")

    # =========================================================================
    # BER Comparison Plot
    # =========================================================================
    # Compare baseline LMMSE (perfect/imperfect CSI) against trained autoencoder
    # BER provides finer granularity than BLER for understanding bit-level performance
    if "ber" in baseline_data and "ber" in inference_data:
        ber = baseline_data[
            "ber"
        ]  # Shape: [2, num_snr_points] - [perfect_csi, imperfect_csi]
        inference_ber = inference_data["ber"]

        plt.figure()
        for idx, csi_label in enumerate(["(Perfect CSI)", "(Imperfect CSI)"]):
            plt.semilogy(
                ebno_db,
                ber[idx],
                marker="o",
                linestyle="-",
                label=f"Conventional Detector {csi_label}",
            )
        plt.semilogy(
            ebno_db,
            inference_ber,
            marker="o",
            linestyle="-",
            label="Neural MIMO Detector (Imperfect CSI)",
        )
        plt.xlabel("Eb/N0 [dB]")
        plt.ylabel("BER")
        plt.title(f"PUSCH - BER vs Eb/N0 ({num_bs_ant} BS Antennas)")
        plt.grid(True, which="both")
        plt.legend()

        ber_outfile = os.path.join(
            DEMO_DIR,
            "results",
            f"ber_plot_{num_bs}bs_{num_bs_ant}bs_ant_x_{num_ue}ue_{num_ue_ant}ue_ant.png",
        )
        plt.savefig(ber_outfile, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved BER plot to {ber_outfile}")
    else:
        print(
            f"Warning: BER data not found in baseline or inference results. "
            f"Skipping BER plot for {num_bs_ant} antennas."
        )

    # =========================================================================
    # Training Loss Analysis
    # =========================================================================
    loss_path = os.path.join(DEMO_DIR, "results", f"training_loss{ant_suffix}.npy")
    if os.path.exists(loss_path):
        loss_values = np.load(loss_path)

        # Find best (minimum) loss for reference line
        best_loss = np.min(loss_values)
        best_iteration = np.argmin(loss_values)
        print(f"Best loss: {best_loss:.6f} at iteration {best_iteration}")

        # Plot loss from iteration 500 onwards (skip initial transient)
        # Early iterations have high variance and dominate the y-axis scale
        start_iter = 500
        end_iter = min(5000, len(loss_values))
        iterations_range = np.arange(start_iter, end_iter)
        loss_to_plot = loss_values[start_iter:end_iter]

        plt.figure(figsize=(10, 5))
        plt.plot(iterations_range, loss_to_plot, linewidth=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Training Loss - Iterations 500-5000 ({num_bs_ant} BS Antennas)")
        plt.grid(True, linestyle="--", alpha=0.7)
        # Reference line at best loss helps identify convergence
        plt.axhline(
            best_loss,
            color="r",
            linestyle="--",
            linewidth=0.8,
            label=f"Best: {best_loss:.4f} @ iter {best_iteration}",
        )
        plt.legend()

        loss_outfile = os.path.join(
            DEMO_DIR, "results", f"training_loss{ant_suffix}.png"
        )
        plt.savefig(loss_outfile, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved loss plot to {loss_outfile}")
    else:
        print(f"Warning: {loss_path} not found, skipping loss analysis.")

    # =========================================================================
    # Load Final Trained Weights
    # =========================================================================
    final_weights_path = os.path.join(
        DEMO_DIR, "results", f"PUSCH_autoencoder_weights{ant_suffix}"
    )
    if not os.path.exists(final_weights_path):
        print(f"Warning: {final_weights_path} not found, skipping constellation plots.")
        continue

    with open(final_weights_path, "rb") as f:
        final_weights = pickle.load(f)

    # =========================================================================
    # Display Correction Scale Values
    # =========================================================================
    # These scales indicate how much the neural network deviates from classical LMMSE:
    # - Values near 0: Neural corrections have minimal effect (classical dominates)
    # - Large values: Neural network significantly modifies classical estimates
    if "rx_weights" in final_weights:
        rx_weights = final_weights["rx_weights"]
        # Weight ordering matches PUSCHNeuralDetector.trainable_variables:
        # [h_correction_scale, err_var_correction_scale_raw, llr_correction_scale, ...nn_weights...]
        h_correction_scale = float(rx_weights[0])
        err_var_correction_scale_raw = float(rx_weights[1])
        llr_correction_scale = float(rx_weights[2])

        # Apply softplus to get actual scale: softplus(x) = log(1 + exp(x))
        # This transformation ensures the error variance scale is always positive
        err_var_correction_scale = np.log(1 + np.exp(err_var_correction_scale_raw))

        print("Correction scales:")
        print(f"  h_correction_scale: {h_correction_scale:.6f}")
        print(f"  err_var_correction_scale (softplus): {err_var_correction_scale:.6f}")
        print(f"  llr_correction_scale: {llr_correction_scale:.6f}")

    # =========================================================================
    # Final Constellation Plot
    # =========================================================================
    # Compare trained constellation against standard 16-QAM
    # tx_weights layout: [points_r, points_i] from PUSCHTrainableTransmitter
    final_const = normalize_constellation(
        final_weights["tx_weights"][0], final_weights["tx_weights"][1]
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        init_const.real, init_const.imag, s=40, marker="o", label="Standard 16-QAM"
    )
    ax.scatter(final_const.real, final_const.imag, s=40, marker="x", label="Trained")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(f"Normalized Constellation: Standard vs Trained ({num_bs_ant} BS Ant)")
    ax.legend()

    const_outfile = os.path.join(
        DEMO_DIR, "results", f"constellation_normalized{ant_suffix}.png"
    )
    fig.savefig(const_outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved constellation plot to {const_outfile}")

    # =========================================================================
    # Constellation Evolution at Checkpoints
    # =========================================================================
    # Visualize how constellation geometry evolves during training
    # Useful for understanding optimization dynamics and detecting problems
    iterations = [1000, 2000, 3000, 4000]

    for iteration in iterations:
        weights_path = os.path.join(
            DEMO_DIR,
            "results",
            f"PUSCH_autoencoder_weights_iter_{iteration}{ant_suffix}",
        )

        if not os.path.exists(weights_path):
            print(f"Warning: {weights_path} not found, skipping.")
            continue

        with open(weights_path, "rb") as f:
            weights = pickle.load(f)

        trained_const = normalize_constellation(
            weights["tx_weights"][0], weights["tx_weights"][1]
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            init_const.real, init_const.imag, s=40, marker="o", label="Standard 16-QAM"
        )
        ax.scatter(
            trained_const.real,
            trained_const.imag,
            s=40,
            marker="x",
            label=f"Iter {iteration}",
        )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.set_title(f"Constellation at Iteration {iteration} ({num_bs_ant} BS Ant)")
        ax.legend()

        iter_outfile = os.path.join(
            DEMO_DIR, "results", f"constellation_iter_{iteration}{ant_suffix}.png"
        )
        fig.savefig(iter_outfile, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved constellation plot to {iter_outfile}")

print("\n" + "=" * 60)
print("Plot generation complete for all antenna configurations.")
print("=" * 60)
