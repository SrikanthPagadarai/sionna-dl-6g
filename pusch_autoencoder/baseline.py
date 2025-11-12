# Minimal, strictly-necessary imports so this file runs on its own
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sionna.phy.utils import PlotBER

# Read-only parameters for plotting/title, etc.
from config import Config
_cfg = Config()

# Import the symbols created in system.py
from system import Model
from cir import (
    channel_model,   # created in cir.py
    batch_size,      # created in cir.py (from _cfg.BATCH_SIZE)
    num_ue,          # used for filename formatting
    num_bs_ant,      # used for filename formatting
    num_ue_ant       # used for filename formatting
)

# Quick functional check (unchanged behavior)
ebno_db = 10.
e2e_model = Model(channel_model, perfect_csi=False)

# We can draw samples from the end-2-end link-level simulations
b, b_hat = e2e_model(batch_size, ebno_db)

# SNR sweep
ebno_db = np.arange(-3, 18, 2)

# Build the BER/BLER simulator (plotting helper)
ber_plot = PlotBER("Site-Specific MU-MIMO 5G NR PUSCH")

# Collect results in the order: [Perf. CSI, Imperf. CSI]
ber_list, bler_list = [], []
for perf_csi in [True, False]:
    # Model uses LMMSE internally
    e2e_model = Model(channel_model, perfect_csi=perf_csi)

    ber_i, bler_i = ber_plot.simulate(
        e2e_model,
        ebno_dbs=ebno_db,
        max_mc_iter=50,
        num_target_block_errors=200,
        batch_size=batch_size,
        soft_estimates=False,
        show_fig=False,
        add_bler=True,
    )

    # Ensure NumPy arrays
    ber_list.append(ber_i.numpy() if hasattr(ber_i, "numpy") else np.asarray(ber_i))
    bler_list.append(bler_i.numpy() if hasattr(bler_i, "numpy") else np.asarray(bler_i))

# Stack to arrays with shape [2, len(ebno_db)]
ber = np.stack(ber_list, axis=0)
bler = np.stack(bler_list, axis=0)

# Plot BLER only (two curves: Perf./Imperf. CSI)
os.makedirs("results", exist_ok=True)
plt.figure()
for idx, csi_label in enumerate(["Perfect CSI", "Imperfect CSI"]):
    plt.semilogy(ebno_db, bler[idx], marker="o", linestyle="-", label=f"LMMSE {csi_label}")

plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BLER")
plt.title("PUSCH - BLER vs Eb/N0")
plt.grid(True, which="both")
plt.legend()

outfile = os.path.join(
    "results",
    f"bler_plot_bs{batch_size}_ue{num_ue}_ant{num_bs_ant}x{num_ue_ant}.png"
)
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved BLER plot to {outfile}")
