# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Plotting script for BER/BLER visualization.

Generates comparison plots of baseline receiver vs neural receiver performance:

1. **Training loss curve**: BCE loss over iterations (semilogy scale)
2. **BER plot**: Bit error rate vs Eb/N0 for all receivers
3. **BLER plot**: Block error rate vs Eb/N0 for all receivers

Usage
-----
Run after both baseline.py and inference.py::

    python plots.py

Input Files
-----------
- ``results/baseline_results_cdlC.npz``: Baseline BER/BLER
- ``results/inference_results.npz``: Neural receiver BER/BLER
- ``results/loss.npy``: Training loss history (optional)

Output Files
------------
- ``results/training_loss.png``: Loss curve
- ``results/ber_cdlC.png``: BER comparison plot
- ``results/bler_cdlC.png``: BLER comparison plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# File Paths
# =============================================================================
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(DEMO_DIR, "results", "baseline_results_cdlC.npz")
INFERENCE_FILE = os.path.join(DEMO_DIR, "results", "inference_results.npz")
LOSS_FILE = os.path.join(DEMO_DIR, "results", "loss.npy")

os.makedirs(os.path.join(DEMO_DIR, "results"), exist_ok=True)

# =============================================================================
# Load Data
# =============================================================================
# Baseline results: perfect and imperfect CSI
data = np.load(BASELINE_FILE, allow_pickle=True)
ebno_db = data["ebno_db"]
perfect_csi = data["perfect_csi"]  # shape (2,) typically [True, False]
ber = data["ber"]  # shape (2, len(ebno_db))
bler = data["bler"]  # shape (2, len(ebno_db))
cdl_model = str(data["cdl_model"])

# Neural receiver inference results
inf = np.load(INFERENCE_FILE, allow_pickle=True)
inf_ebno_db = inf["ebno_db"]
inf_ber = inf["ber"]
inf_bler = inf["bler"]

# =============================================================================
# Training Loss Plot
# =============================================================================
if os.path.exists(LOSS_FILE):
    loss = np.load(LOSS_FILE)
    outfile = os.path.join(DEMO_DIR, "results", "training_loss.png")
    plt.figure()
    plt.semilogy(loss)
    plt.xlabel("iteration")
    plt.ylabel("loss (log scale)")
    plt.title("Training Loss Curve")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot to {outfile}")
else:
    print(f"Loss file not found: {LOSS_FILE}")

# =============================================================================
# BER Plot
# Compares baseline (perfect/imperfect CSI) with neural receiver
# =============================================================================
outfile_ber = os.path.join(DEMO_DIR, "results", "ber_cdlC.png")
plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")
plt.ylim([1e-4, 1.1])
plt.title(f"BER - CDL-{cdl_model}")

# Baseline curves
for i in range(len(perfect_csi)):
    label = "perfect CSI" if bool(perfect_csi[i]) else "imperfect CSI"
    plt.semilogy(ebno_db, ber[i], label=label, marker="o", linestyle="-")

# Neural receiver curve
plt.semilogy(
    inf_ebno_db, inf_ber, label="NeuralRx (inference)", marker="x", linestyle="--"
)

plt.legend()
plt.savefig(outfile_ber, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved BER plot to {outfile_ber}")


# =============================================================================
# BLER Plot
# Compares baseline (perfect/imperfect CSI) with neural receiver
# =============================================================================
outfile_bler = os.path.join(DEMO_DIR, "results", "bler_cdlC.png")
plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim([1e-3, 1.1])
plt.title(f"BLER - CDL-{cdl_model}")

# Baseline curves
for i in range(len(perfect_csi)):
    label = "perfect CSI" if bool(perfect_csi[i]) else "imperfect CSI"
    plt.semilogy(ebno_db, bler[i], label=label, marker="o", linestyle="-")

# Neural receiver curve
plt.semilogy(
    inf_ebno_db, inf_bler, label="NeuralRx (inference)", marker="x", linestyle="--"
)
plt.legend()
plt.savefig(outfile_bler, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved BLER plot to {outfile_bler}")
