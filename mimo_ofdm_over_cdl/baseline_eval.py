from typing import Dict, Any, Tuple, List
import os
import time
import numpy as np
import tensorflow as tf
from sionna.phy.utils import sim_ber
from src.system import System

def run_sim(params: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Run Sionna BER/BLER simulation using parameters in a single dict."""
    cfg = {
        "ebno_db": [-5, -1, 3, 7, 11, 15],
        "use_neural_rx": False,
        "direction": "uplink",
        "perfect_csi": True,
        "cdl_model": "D",
        "delay_spread": 100e-9,
        "carrier_frequency": 2.6e9,
        "speed": 0.0,
        "batch_size": 32,
        "max_mc_iter": 2,
        "num_target_block_errors": 20,
        "target_bler": 1e-2,
        **(params or {}),
    }

    system = System(
        direction=cfg["direction"],
        perfect_csi=cfg["perfect_csi"],
        cdl_model=cfg["cdl_model"],
        delay_spread=cfg["delay_spread"],
        carrier_frequency=cfg["carrier_frequency"],
        speed=cfg["speed"],
        use_neural_rx=cfg["use_neural_rx"],
    )

    # Adapter: sim_ber feeds a scalar ebno_db; System expects shape (None,)
    @tf.function
    def mc_fun(batch_size: tf.Tensor, ebno_db: tf.Tensor):
        ebno_vec = tf.reshape(tf.cast(ebno_db, tf.float32), [1])  # shape (1,)
        return system(batch_size, ebno_vec)

    ber, bler = sim_ber(
        mc_fun,
        list(cfg["ebno_db"]),
        batch_size=cfg["batch_size"],
        max_mc_iter=cfg["max_mc_iter"],
        num_target_block_errors=cfg["num_target_block_errors"],
        target_bler=cfg["target_bler"],
    )
    return ber, bler

if __name__ == "__main__":
    base_params = dict(
        cdl_model="C",
        max_mc_iter=1000,
        num_target_block_errors=1000,
        target_bler=1e-3,
        ebno_db=np.arange(-5, 10, 2),
    )

    directions: List[str] = ["uplink", "downlink"]
    perfect_csi_values: List[bool] = [True, False]

    t0 = time.time()
    all_directions, all_csi, all_ber, all_bler = [], [], [], []
    for direction in directions:
        for perfect_csi in perfect_csi_values:
            params = {**base_params, "direction": direction, "perfect_csi": perfect_csi}
            ber, bler = run_sim(params)

            all_directions.append(direction)
            all_csi.append(bool(perfect_csi))
            all_ber.append(ber.numpy())
            all_bler.append(bler.numpy())

            csi_tag = "perfect CSI" if perfect_csi else "imperfect CSI"
            print(f"[CDL-{base_params['cdl_model']}] {direction:8s} | {csi_tag:13s} | "
                  f"BLER={np.array2string(bler.numpy(), precision=3)}")

    dur_min = (time.time() - t0) / 60.0
    print(f"Total duration: {dur_min:.2f} min")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    outfile = os.path.join(out_dir, "all_baseline_results_cdlC.npz")
    np.savez(
        outfile,
        ebno_db=base_params["ebno_db"],
        directions=np.array(all_directions),
        perfect_csi=np.array(all_csi),
        ber=np.array(all_ber),
        bler=np.array(all_bler),
        cdl_model=base_params["cdl_model"],
    )
    print(f"Saved results to {outfile}")
