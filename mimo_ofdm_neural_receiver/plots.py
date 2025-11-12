import os
import numpy as np
import matplotlib.pyplot as plt

# Constants
BASELINE_FILE = os.path.join("results", "all_baseline_results_cdlC.npz")
INFERENCE_FILE = os.path.join("results", "all_inference_results.npz")

# Load Data
data = np.load(BASELINE_FILE, allow_pickle=True)
ebno_db = data["ebno_db"]
directions = data["directions"]
perfect_csi = data["perfect_csi"]
ber = data["ber"]
bler = data["bler"]
cdl_model = str(data["cdl_model"])

# Load inference results for overlap
inference_data = np.load(INFERENCE_FILE, allow_pickle=True)
inf_directions = inference_data["directions"]
inf_ber = inference_data["ber"]
inf_bler = inference_data["bler"]

os.makedirs("results", exist_ok=True)
for direction in ["uplink", "downlink"]:
    loss = np.load(os.path.join("results", f"loss_{direction}.npy"))
    outfile = os.path.join("results", f"loss_{direction}.png")
    plt.semilogy(loss)
    plt.xlabel("iteration")
    plt.ylabel("loss (log scale)")
    plt.title(f"Training Loss Curve ({direction})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {direction}-loss plot to {outfile}")

## baseline plots (renamed and overlapped with inference results)
OUTFILE_UPLINK_BER = os.path.join("results", "uplink_ber_cdlC.png")
OUTFILE_DOWNLINK_BER = os.path.join("results", "downlink_ber_cdlC.png")
for direction, outfile in [("uplink", OUTFILE_UPLINK_BER), ("downlink", OUTFILE_DOWNLINK_BER)]:
    plt.figure()
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.ylim([1e-4, 1.1])
    if direction == "uplink":
        plt.title(f"4x8 MIMO - {direction.capitalize()} - CDL-{cdl_model}")
    elif direction == "downlink":
        plt.title(f"8x4 MIMO - {direction.capitalize()} - CDL-{cdl_model}")

    # baseline plots
    for i in range(len(directions)):
        if directions[i] == direction:
            label = f"{'perfect CSI' if perfect_csi[i] else 'imperfect CSI'}"
            plt.semilogy(ebno_db, ber[i], label=label, marker='o', linestyle='-')

    # overlay inference results
    for j in range(len(inf_directions)):
        if inf_directions[j] == direction:
            plt.semilogy(ebno_db, inf_ber[j], label="NeuralRx (inference)", marker='x', linestyle='--')

    plt.legend()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {direction} BER plot to {outfile}")

OUTFILE_UPLINK_BLER = os.path.join("results", "uplink_bler_cdlC.png")
OUTFILE_DOWNLINK_BLER = os.path.join("results", "downlink_bler_cdlC.png")
for direction, outfile in [("uplink", OUTFILE_UPLINK_BLER), ("downlink", OUTFILE_DOWNLINK_BLER)]:
    plt.figure()
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.ylim([1e-3, 1.1])
    if direction == "uplink":
        plt.title(f"4x8 MIMO - {direction.capitalize()} - CDL-{cdl_model}")
    elif direction == "downlink":
        plt.title(f"8x4 MIMO - {direction.capitalize()} - CDL-{cdl_model}")

    # baseline plots
    for i in range(len(directions)):
        if directions[i] == direction:
            label = f"{'perfect CSI' if perfect_csi[i] else 'imperfect CSI'}"
            plt.semilogy(ebno_db, bler[i], label=label, marker='o', linestyle='-')

    # overlay inference results
    for j in range(len(inf_directions)):
        if inf_directions[j] == direction:
            plt.semilogy(ebno_db, inf_bler[j], label="NeuralRx (inference)", marker='x', linestyle='--')

    plt.legend()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {direction} BLER plot to {outfile}")
