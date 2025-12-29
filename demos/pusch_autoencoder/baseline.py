import os
import sys
import numpy as np
import tensorflow as tf
from sionna.phy.utils import PlotBER

from demos.pusch_autoencoder.src.config import Config
from demos.pusch_autoencoder.src.system import PUSCHLinkE2E
from demos.pusch_autoencoder.src.cir_manager import CIRManager

# Set GPU device if not already specified
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Suppress TensorFlow info/warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Configure GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}", file=sys.stderr)

# Get configuration
_cfg = Config()
batch_size = _cfg.batch_size
num_ue = _cfg.num_ue
num_bs_ant = _cfg.num_bs_ant
num_ue_ant = _cfg.num_ue_ant
num_time_steps = _cfg.num_time_steps  # kept for completeness

# Build channel model
cir_manager = CIRManager()
channel_model = cir_manager.build_channel_model()

# quick functional check
ebno_db_test = 10.0
e2e_model_test = PUSCHLinkE2E(
    channel_model,
    perfect_csi=False,
    use_autoencoder=False,
)
b, b_hat = e2e_model_test(batch_size, ebno_db_test)
print("Quick check shapes:", b.shape, b_hat.shape)

# BER/BLER Simulation
ebno_db = np.arange(-2, 10, 1)

ber_plot = PlotBER("Site-Specific MU-MIMO 5G NR PUSCH")

ber_list, bler_list = [], []
for perf_csi in [True, False]:
    e2e_model = PUSCHLinkE2E(
        channel_model,
        perfect_csi=perf_csi,
        use_autoencoder=False,
    )

    ber_i, bler_i = ber_plot.simulate(
        e2e_model,
        ebno_dbs=ebno_db,
        max_mc_iter=500,
        num_target_block_errors=2000,
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

# Save results to NPZ (no plotting here)
os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", "baseline_results.npz")
np.savez(
    out_path,
    ebno_db=ebno_db,
    ber=ber,
    bler=bler,
    batch_size=batch_size,
    num_ue=num_ue,
    num_bs_ant=num_bs_ant,
    num_ue_ant=num_ue_ant,
    num_time_steps=num_time_steps,
)
print(f"Saved BER/BLER results to {out_path}")
