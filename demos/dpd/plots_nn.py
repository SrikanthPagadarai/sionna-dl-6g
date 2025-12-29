#!/usr/bin/env python3
"""Generate all NN-DPD plots from saved data files."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402

RESULTS_DIR = Path("results")


def plot_training_loss():
    """Plot NN-DPD training loss history."""
    loss_file = RESULTS_DIR / "loss.npy"
    if not loss_file.exists():
        print(f"Skipping training loss plot: {loss_file} not found")
        return

    loss_history = np.load(loss_file)

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.title("NN DPD Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "training_loss.png", dpi=150)
    plt.close()
    print("Saved training_loss.png")


def plot_psd():
    """Plot PSD comparison."""
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
    plt.ylim([-120, 10])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psd_comparison_nn.png", dpi=150)
    plt.close()
    print("Saved psd_comparison_nn.png")


def plot_constellation():
    """Plot constellation comparison."""
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
        label=f"PA Output, with NN DPD (EVM={evm_with_dpd:.1f}%)",
        alpha=0.5,
    )
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title("Constellation Comparison: Effect of NN DPD")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
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
