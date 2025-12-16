import os
import numpy as np
import matplotlib.pyplot as plt

# Files
BASELINE_FILE = os.path.join("results", "all_baseline_results_cdlC.npz")
INFERENCE_FILE = os.path.join("results", "inference_results.npz")
LOSS_FILE = os.path.join("results", "loss.npy")

os.makedirs("results", exist_ok=True)

# Load baseline data (single-run)
data = np.load(BASELINE_FILE, allow_pickle=True)
ebno_db = data["ebno_db"]
perfect_csi = data["perfect_csi"]   # shape (2,) typically [True, False]
ber = data["ber"]                   # shape (2, len(ebno_db))
bler = data["bler"]                 # shape (2, len(ebno_db))
cdl_model = str(data["cdl_model"])

# Load inference results (single-run)
inf = np.load(INFERENCE_FILE, allow_pickle=True)
inf_ebno_db = inf["ebno_db"]
inf_ber = inf["ber"]
inf_bler = inf["bler"]

# Loss plot (single-run)
if os.path.exists(LOSS_FILE):
    loss = np.load(LOSS_FILE)
    outfile = os.path.join("results", "loss.png")
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

# BER plot (baseline + inference overlay)
outfile_ber = os.path.join("results", "ber_cdlC.png")
plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")
plt.ylim([1e-4, 1.1])
plt.title(f"BER - CDL-{cdl_model}")

# baseline curves
for i in range(len(perfect_csi)):
    label = "perfect CSI" if bool(perfect_csi[i]) else "imperfect CSI"
    plt.semilogy(ebno_db, ber[i], label=label, marker="o", linestyle="-")

# overlay inference
plt.semilogy(inf_ebno_db, inf_ber, label="NeuralRx (inference)", marker="x", linestyle="--")

plt.legend()
plt.savefig(outfile_ber, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved BER plot to {outfile_ber}")


# BLER plot (baseline + inference overlay)
outfile_bler = os.path.join("results", "bler_cdlC.png")
plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim([1e-3, 1.1])
plt.title(f"BLER - CDL-{cdl_model}")

# baseline curves
for i in range(len(perfect_csi)):
    label = "perfect CSI" if bool(perfect_csi[i]) else "imperfect CSI"
    plt.semilogy(ebno_db, bler[i], label=label, marker="o", linestyle="-")

# overlay inference
plt.semilogy(inf_ebno_db, inf_bler, label="NeuralRx (inference)", marker="x", linestyle="--")
plt.legend()
plt.savefig(outfile_bler, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved BLER plot to {outfile_bler}")
