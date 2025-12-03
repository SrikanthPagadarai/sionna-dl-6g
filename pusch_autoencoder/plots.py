import os
import pickle
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
two_phase_inf_results_path = os.path.join("results", "inference_results_two_phase.npz")

data = np.load(results_path)
conv_inf_data = np.load(conv_inf_results_path)
two_phase_inf_data = np.load(two_phase_inf_results_path)
ebno_db = data["ebno_db"]
bler = data["bler"]
conv_inf_bler = conv_inf_data["bler"]
two_phase_inf_bler = two_phase_inf_data["bler"]

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
plt.semilogy(ebno_db,conv_inf_bler,marker="o",linestyle="-",label=f"Neural MIMO Detector (Imperfect CSI, SGD)")
plt.semilogy(ebno_db,two_phase_inf_bler,marker="o",linestyle="-",label=f"Neural MIMO Detector (Imperfect CSI, Two-Phase)")
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

# Constellation: Initial vs Final (Normalized)
def normalize_constellation(points_r, points_i):
    """Apply same normalization as get_normalized_constellation()."""
    points = points_r + 1j * points_i
    
    # Center (subtract mean)
    points = points - np.mean(points)
    
    # Normalize to unit power
    energy = np.mean(np.abs(points) ** 2)
    points = points / np.sqrt(energy)
    
    return points

def standard_16qam():
    """Generate standard 16-QAM constellation (unit power)."""
    levels = np.array([-3, -1, 1, 3])
    real, imag = np.meshgrid(levels, levels)
    points = (real.flatten() + 1j * imag.flatten()) / np.sqrt(10)
    return points

# Initial: standard 16-QAM
init_const = standard_16qam()

# Load final weights
final_weights_path = os.path.join("results", "PUSCH_autoencoder_weights_conventional_training")
with open(final_weights_path, 'rb') as f:
    final_weights = pickle.load(f)

# tx_weights[0] = points_r, tx_weights[1] = points_i
final_const = normalize_constellation(final_weights['tx_weights'][0], final_weights['tx_weights'][1])

# Plot final constellation
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(init_const.real, init_const.imag, s=40, marker='o', label='Standard 16-QAM')
ax.scatter(final_const.real, final_const.imag, s=40, marker='x', label='Trained')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_aspect('equal', 'box')
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_xlabel('In-phase')
ax.set_ylabel('Quadrature')
ax.set_title('Normalized Constellation: Standard vs Trained')
ax.legend()

const_outfile = os.path.join("results", "constellation_normalized.png")
fig.savefig(const_outfile, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved constellation plot to {const_outfile}")

# Plot constellation evolution at intermediate iterations
iterations = [1000, 2000, 3000, 4000]

for iteration in iterations:
    weights_path = os.path.join("results", f"PUSCH_autoencoder_weights_conv_iter_{iteration}")
    
    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} not found, skipping.")
        continue
    
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    
    trained_const = normalize_constellation(weights['tx_weights'][0], weights['tx_weights'][1])
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(init_const.real, init_const.imag, s=40, marker='o', label='Standard 16-QAM')
    ax.scatter(trained_const.real, trained_const.imag, s=40, marker='x', label=f'Iter {iteration}')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlabel('In-phase')
    ax.set_ylabel('Quadrature')
    ax.set_title(f'Constellation at Iteration {iteration}')
    ax.legend()
    
    iter_outfile = os.path.join("results", f"constellation_iter_{iteration}.png")
    fig.savefig(iter_outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved constellation plot to {iter_outfile}")