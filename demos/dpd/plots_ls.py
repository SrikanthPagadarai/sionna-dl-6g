#!/usr/bin/env python3
"""Generate all LS-DPD plots from saved data files."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402

RESULTS_DIR = Path("results")


def plot_coefficient_convergence():
    """Plot LS-DPD coefficient convergence history."""
    coeff_file = RESULTS_DIR / "ls-dpd-coeff-history.npy"
    if not coeff_file.exists():
        print(f"Skipping convergence plot: {coeff_file} not found")
        return

    coeff_history = np.load(coeff_file)

    plt.figure(figsize=(10, 6))
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
    """Plot PSD comparison."""
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
    plt.ylim([-120, 10])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psd_comparison_ls.png", dpi=150)
    plt.close()
    print("Saved psd_comparison_ls.png")


def plot_constellation():
    """Plot constellation comparison."""
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
    plt.plot(
        fd_symbols.real.flatten(),
        fd_symbols.imag.flatten(),
        "o",
        ms=4,
        label="Original (TX)",
        alpha=0.5,
    )
    plt.plot(
        sym_input.real.flatten(),
        sym_input.imag.flatten(),
        "s",
        ms=3,
        label=f"PA Input (EVM={evm_input:.1f}%)",
        alpha=0.5,
    )
    plt.plot(
        sym_no_dpd.real.flatten(),
        sym_no_dpd.imag.flatten(),
        "x",
        ms=3,
        label=f"PA Output, no DPD (EVM={evm_no_dpd:.1f}%)",
        alpha=0.5,
    )
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
    plt.axis("equal")
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
