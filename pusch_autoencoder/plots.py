import os
import numpy as np
import matplotlib.pyplot as plt
from src.config import Config

# Load config (for consistent naming in output file)
_cfg = Config()
batch_size = _cfg.batch_size
num_ue = _cfg.num_ue
num_bs_ant = _cfg.num_bs_ant
num_ue_ant = _cfg.num_ue_ant

# Load precomputed results
results_path = os.path.join("results", "baseline_results.npz")
if not os.path.exists(results_path):
    raise FileNotFoundError(
        f"{results_path} not found. "
        "Run `python3 baseline.py` first to generate BER/BLER results."
    )
conv_inf_results_path = os.path.join("results", "inference_results_conventional.npz")

data = np.load(results_path)
conv_inf_data = np.load(conv_inf_results_path)
ebno_db = data["ebno_db"]
bler = data["bler"]
conv_inf_bler = conv_inf_data["bler"]

# Plot BLER
plt.figure()
for idx, csi_label in enumerate(["(Perfect CSI)", "(Imperfect CSI)"]):
    plt.semilogy(
        ebno_db,
        bler[idx],
        marker="o",
        linestyle="-",
        label=f"LMMSE {csi_label}",
    )
plt.semilogy(ebno_db,conv_inf_bler,marker="o",linestyle="-",label=f"Neural MIMO Detector (Imperfect CSI, SGD training)")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BLER")
plt.title("PUSCH - BLER vs Eb/N0")
plt.grid(True, which="both")
plt.legend()

outfile = os.path.join(
    "results",
    f"bler_plot_bs{batch_size}_ue{num_ue}_ant{num_bs_ant}x{num_ue_ant}.png",
)
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved BLER plot to {outfile}")