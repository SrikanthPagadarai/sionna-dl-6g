import os
import time
from typing import Any, Dict, List
import numpy as np
import tensorflow as tf
from sionna.phy.utils import sim_ber

from src.system import System

# User-tunable parameters
base_params: Dict[str, Any] = dict(
    cdl_model="C",
    delay_spread=100e-9,
    carrier_frequency=2.6e9,
    speed=0.0,
    use_neural_rx=False,
    batch_size=32,
    max_mc_iter=1000,
    num_target_block_errors=1000,
    target_bler=1e-3,
    ebno_db=np.arange(-3, 7, 1),
)

# parametrizable values
perfect_csi_values: List[bool] = [True, False]


# Run simulation
t0 = time.time()
all_csi: List[bool] = []
all_ber: List[np.ndarray] = []
all_bler: List[np.ndarray] = []
for perfect_csi in perfect_csi_values:
    cfg = {
        **base_params,
        "perfect_csi": bool(perfect_csi),
    }

    system = System(
        perfect_csi=cfg["perfect_csi"],
        cdl_model=cfg["cdl_model"],
        delay_spread=cfg["delay_spread"],
        carrier_frequency=cfg["carrier_frequency"],
        speed=cfg["speed"],
        use_neural_rx=cfg["use_neural_rx"],
    )

    # Adapter: sim_ber provides scalar ebno_db,
    # whereas System expects a vector (len == batch_size).
    @tf.function
    def mc_fun(batch_size: tf.Tensor, ebno_db: tf.Tensor):
        ebno_vec = tf.fill([batch_size], tf.cast(ebno_db, tf.float32))  # (B,)
        return system(batch_size, ebno_vec)

    ber, bler = sim_ber(
        mc_fun,
        list(cfg["ebno_db"]),
        batch_size=int(cfg["batch_size"]),
        max_mc_iter=int(cfg["max_mc_iter"]),
        num_target_block_errors=int(cfg["num_target_block_errors"]),
        target_bler=float(cfg["target_bler"]),
    )

    all_csi.append(bool(perfect_csi))
    all_ber.append(ber.numpy())
    all_bler.append(bler.numpy())

    csi_tag = "perfect CSI" if perfect_csi else "imperfect CSI"
    print(
        f"[CDL-{cfg['cdl_model']}] uplink   | {csi_tag:13s} | "
        f"BLER={np.array2string(bler.numpy(), precision=3)}"
    )
dur_min = (time.time() - t0) / 60.0
print(f"Total duration: {dur_min:.2f} min")

# Save results
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

outfile = os.path.join(out_dir, "all_baseline_results_cdlC.npz")
np.savez(
    outfile,
    ebno_db=np.array(base_params["ebno_db"]),
    perfect_csi=np.array(all_csi),
    ber=np.array(all_ber),
    bler=np.array(all_bler),
    cdl_model=base_params["cdl_model"],
)

print(f"Saved results to {outfile}")
