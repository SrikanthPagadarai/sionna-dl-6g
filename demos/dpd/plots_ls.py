#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Visualization script for least-squares DPD results.

This script generates plots from data saved by ``inference.py``
and ``training_ls.py``. It visualizes three key aspects
of DPD performance:

1. **Coefficient Convergence** : Shows how LS-DPD coefficients evolve
   across indirect learning iterations. Useful for verifying convergence
   and understanding the learning dynamics.

2. **Power Spectral Density** : Compares PA output spectrum with and
   without DPD. The key metric is out-of-band spectral regrowth reduction.

3. **Constellation Diagram** : Shows demodulated QAM symbols overlaid.
   Tighter clustering around reference points indicates better EVM.

Input Files
-----------
All input files are expected in the ``results/`` directory:

- ``ls-dpd-coeff-history.npy`` : Coefficient history from training
- ``psd_data_ls.npz`` : Power spectral density data from inference
- ``constellation_data_ls.npz`` : Demodulated symbols from inference

Output Files
------------
Generated plots are saved to the ``results/`` directory:

- ``ls_dpd_convergence.png`` : Coefficient convergence plot
- ``psd_comparison_ls.png`` : PSD comparison plot
- ``constellation_comparison_ls.png`` : Constellation diagram

Usage
-----
::

    # First run training and inference
    python training_ls.py
    python inference.py --dpd_method ls

    # Then generate plots
    python plots_ls.py

See Also
--------
plots_nn.py : Equivalent plotting script for NN-DPD results.
inference.py : Script that generates the input data files.
training_ls.py : Script that generates coefficient history.
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


def plot_coefficient_convergence():
    """
    Plot LS-DPD coefficient magnitude evolution across iterations.

    Visualizes how each coefficient converges during indirect learning.

    Input File
    ----------
    ``results/ls-dpd-coeff-history.npy`` : ndarray, shape ``[n_coeffs, n_iterations+1]``
        Coefficient values at each iteration. First column is initial
        (identity) coefficients.

    Output File
    -----------
    ``results/ls_dpd_convergence.png``
        Line plot showing coefficient magnitudes vs iteration.

    Notes
    -----
    Only the first 10 coefficients are plotted to avoid legend clutter.
    These typically include the most significant terms (low-order, low-memory).

    Convergence is usually achieved within 4-6 iterations for LS-DPD.
    If coefficients continue changing significantly after 6+ iterations,
    consider increasing the regularization or checking for numerical issues.
    """
    coeff_file = RESULTS_DIR / "ls-dpd-coeff-history.npy"
    if not coeff_file.exists():
        print(f"Skipping convergence plot: {coeff_file} not found")
        return

    coeff_history = np.load(coeff_file)

    plt.figure(figsize=(10, 6))
    # Plot first 10 coefficients to avoid legend clutter.
    for i in range(min(10, coeff_history.shape[0])):
        plt.plot(np.abs(coeff_history[i, :]), label=f"Coeff {i}")
    plt.title("LS-DPD Coefficient Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("|Coefficient|")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "ls_dpd_convergence.png", dpi=150)
    plt.close()
    print("Saved ls_dpd_convergence.png")


def plot_psd():
    """
    Plot power spectral density comparison showing DPD effectiveness.

    Compares three signals:
    - PA Input (reference) : The ideal linear signal
    - PA Output (no DPD) : Shows spectral regrowth from PA nonlinearity
    - PA Output (with DPD) : Should closely match reference

    The key observation is the reduction in out-of-band emissions
    (spectral regrowth) when DPD is applied.

    Input File
    ----------
    ``results/psd_data_ls.npz`` : Compressed numpy archive containing:
        - ``freqs_mhz`` : Frequency axis in MHz
        - ``psd_input_db`` : PA input PSD in dBc
        - ``psd_no_dpd_db`` : PA output (no DPD) PSD in dBc
        - ``psd_with_dpd_db`` : PA output (with DPD) PSD in dBc

    Output File
    -----------
    ``results/psd_comparison_ls.png``
        Line plot comparing PSDs of all three signals.

    Notes
    -----
    PSDs are normalized to 0 dBc (in-band power = 0 dB) for fair
    comparison.
    """
    data_file = RESULTS_DIR / "psd_data_ls.npz"
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
        label="PA Output (LS DPD)",
        alpha=0.8,
    )
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dBc)")
    plt.title("Power Spectral Density: Effect of LS DPD")
    plt.legend()
    plt.grid(True)
    plt.ylim([-120, 10])  # Focus on relevant dynamic range.
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psd_comparison_ls.png", dpi=150)
    plt.close()
    print("Saved psd_comparison_ls.png")


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
    ``results/constellation_data_ls.npz`` : Compressed numpy archive containing:
        - ``fd_symbols`` : Reference transmitted symbols
        - ``sym_input`` : Demodulated symbols from PA input
        - ``sym_no_dpd`` : Demodulated symbols from PA output (no DPD)
        - ``sym_with_dpd`` : Demodulated symbols from PA output (with DPD)
        - ``evm_input``, ``evm_no_dpd``, ``evm_with_dpd`` : EVM values in %

    Output File
    -----------
    ``results/constellation_comparison_ls.png``
        Scatter plot showing all constellation points overlaid.

    """
    data_file = RESULTS_DIR / "constellation_data_ls.npz"
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
        label=f"PA Output, with LS DPD (EVM={evm_with_dpd:.1f}%)",
        alpha=0.5,
    )
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title("Constellation Comparison: Effect of LS DPD")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")  # Ensure I and Q axes have same scale.
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "constellation_comparison_ls.png", dpi=150)
    plt.close()
    print("Saved constellation_comparison_ls.png")


if __name__ == "__main__":
    print("Generating LS-DPD plots...")
    plot_coefficient_convergence()
    plot_psd()
    plot_constellation()
    print("Done.")
