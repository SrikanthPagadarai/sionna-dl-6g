#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Visualization script for neural network DPD results.

This script generates plots from data saved by
``inference.py`` and ``training_nn.py``. It visualizes three key aspects
of NN-DPD performance:

1. **Training Loss Curve** : Shows MSE loss evolution during gradient
   descent training. Useful for diagnosing convergence and determining
   if more training iterations are needed.

2. **Power Spectral Density** : Compares PA output spectrum with and
   without DPD. The key metric is out-of-band spectral regrowth reduction.

3. **Constellation Diagram** : Shows demodulated QAM symbols overlaid.
   Tighter clustering around reference points indicates better EVM.

Input Files
-----------
All input files are expected in the ``results/`` directory:

- ``loss.npy`` : Training loss history from training_nn.py
- ``psd_data_nn.npz`` : Power spectral density data from inference
- ``constellation_data_nn.npz`` : Demodulated symbols from inference

Output Files
------------
Generated plots are saved to the ``results/`` directory:

- ``training_loss.png`` : Loss curve (log scale)
- ``psd_comparison_nn.png`` : PSD comparison plot
- ``constellation_comparison_nn.png`` : Constellation diagram

Usage
-----
::

    # First run training and inference
    python training_nn.py --iterations 10000
    python inference.py --dpd_method nn

    # Then generate plots
    python plots_nn.py

See Also
--------
plots_ls.py : Equivalent plotting script for LS-DPD results.
inference.py : Script that generates the PSD and constellation data.
training_nn.py : Script that generates the loss history.
"""

import numpy as np
import matplotlib

# Use non-interactive backend for headless operation.
# Must be set before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402

# Directory containing input data and output plots.
DEMO_DIR = Path(__file__).parent
RESULTS_DIR = DEMO_DIR / "results"


def plot_training_loss():
    """
    Plot NN-DPD training loss curve on logarithmic scale.

    Visualizes the MSE loss evolution during gradient descent training.

    Input File
    ----------
    ``results/loss.npy`` : ndarray, shape ``[n_iterations]``
        MSE loss value at each training iteration. Note that loss is
        scaled by 1000x during training for readability.

    Output File
    -----------
    ``results/training_loss.png``
        Semilogy plot showing loss vs iteration.

    """
    loss_file = RESULTS_DIR / "loss.npy"
    if not loss_file.exists():
        print(f"Skipping training loss plot: {loss_file} not found")
        return

    loss_history = np.load(loss_file)

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)  # Log scale for loss spanning orders of magnitude.
    plt.title("NN DPD Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "training_loss.png", dpi=150)
    plt.close()
    print("Saved training_loss.png")


def plot_psd():
    """
    Plot power spectral density comparison showing DPD effectiveness.

    Compares three signals:
    - PA Input (reference) : The ideal linear signal
    - PA Output (no DPD) : Shows spectral regrowth from PA nonlinearity
    - PA Output (with DPD) : Should closely match reference

    Input File
    ----------
    ``results/psd_data_nn.npz`` : Compressed numpy archive containing:
        - ``freqs_mhz`` : Frequency axis in MHz
        - ``psd_input_db`` : PA input PSD in dBc
        - ``psd_no_dpd_db`` : PA output (no DPD) PSD in dBc
        - ``psd_with_dpd_db`` : PA output (with DPD) PSD in dBc

    Output File
    -----------
    ``results/psd_comparison_nn.png``
        Line plot comparing PSDs of all three signals.

    Notes
    -----
    PSDs are normalized to 0 dBc (in-band power = 0 dB) for fair
    comparison.
    """
    data_file = RESULTS_DIR / "psd_data_nn.npz"
    if not data_file.exists():
        print(f"Skipping PSD plot: {data_file} not found")
        return

    data = np.load(data_file)

    plt.figure(figsize=(12, 6))
    plt.plot(
        data["freqs_mhz"], data["psd_input_db"], label="PA Input (Reference)", alpha=0.8
    )
    plt.plot(
        data["freqs_mhz"], data["psd_no_dpd_db"], label="PA Output (No DPD)", alpha=0.8
    )
    plt.plot(
        data["freqs_mhz"],
        data["psd_with_dpd_db"],
        label="PA Output (NN DPD)",
        alpha=0.8,
    )
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dBc)")
    plt.title("Power Spectral Density: Effect of NN DPD")
    plt.legend()
    plt.grid(True)
    plt.ylim([-120, 10])  # Focus on relevant dynamic range.
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psd_comparison_nn.png", dpi=150)
    plt.close()
    print("Saved psd_comparison_nn.png")


def plot_constellation():
    """
    Plot constellation diagram showing in-band distortion.

    Overlays demodulated QAM symbols from multiple signal paths:
    - Original TX symbols
    - PA Input
    - PA Output without DPD
    - PA Output with DPD

    EVM (Error Vector Magnitude) is annotated for each signal path.

    Input File
    ----------
    ``results/constellation_data_nn.npz`` : Compressed numpy archive containing:
        - ``fd_symbols`` : Reference transmitted symbols
        - ``sym_input`` : Demodulated symbols from PA input
        - ``sym_no_dpd`` : Demodulated symbols from PA output (no DPD)
        - ``sym_with_dpd`` : Demodulated symbols from PA output (with DPD)
        - ``evm_input``, ``evm_no_dpd``, ``evm_with_dpd`` : EVM values in %

    Output File
    -----------
    ``results/constellation_comparison_nn.png``
        Scatter plot showing all constellation points overlaid.

    """
    data_file = RESULTS_DIR / "constellation_data_nn.npz"
    if not data_file.exists():
        print(f"Skipping constellation plot: {data_file} not found")
        return

    data = np.load(data_file)
    fd_symbols = data["fd_symbols"]
    sym_input = data["sym_input"]
    sym_no_dpd = data["sym_no_dpd"]
    sym_with_dpd = data["sym_with_dpd"]
    evm_input = float(data["evm_input"])
    evm_no_dpd = float(data["evm_no_dpd"])
    evm_with_dpd = float(data["evm_with_dpd"])

    plt.figure(figsize=(10, 10))
    # Plot reference constellation.
    plt.plot(
        fd_symbols.real.flatten(),
        fd_symbols.imag.flatten(),
        "o",
        ms=4,
        label="Original (TX)",
        alpha=0.5,
    )
    # Plot PA input
    plt.plot(
        sym_input.real.flatten(),
        sym_input.imag.flatten(),
        "s",
        ms=3,
        label=f"PA Input (EVM={evm_input:.1f}%)",
        alpha=0.5,
    )
    # Plot PA output without DPD
    plt.plot(
        sym_no_dpd.real.flatten(),
        sym_no_dpd.imag.flatten(),
        "x",
        ms=3,
        label=f"PA Output, no DPD (EVM={evm_no_dpd:.1f}%)",
        alpha=0.5,
    )
    # Plot PA output with DPD
    plt.plot(
        sym_with_dpd.real.flatten(),
        sym_with_dpd.imag.flatten(),
        ".",
        ms=3,
        label=f"PA Output, with NN DPD (EVM={evm_with_dpd:.1f}%)",
        alpha=0.5,
    )
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title("Constellation Comparison: Effect of NN DPD")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")  # Ensure I and Q axes have same scale.
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "constellation_comparison_nn.png", dpi=150)
    plt.close()
    print("Saved constellation_comparison_nn.png")


if __name__ == "__main__":
    print("Generating NN-DPD plots...")
    plot_training_loss()
    plot_psd()
    plot_constellation()
    print("Done.")
