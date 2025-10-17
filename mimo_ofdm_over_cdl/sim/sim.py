# sim.py
from typing import Dict, Any, Tuple
import tensorflow as tf
from sionna.phy.utils import sim_ber
from src.system import System

def run_sim(params: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Run Sionna BER/BLER simulation using parameters in a single dict."""

    # Defaults
    cfg = {
        "ebno_db": [-5, -1, 3, 7, 11, 15],  # default sweep
        "use_neural_rx": False,
        "direction": "uplink",
        "perfect_csi": True,
        "cdl_model": "B",
        "delay_spread": 100e-9,
        "carrier_frequency": 2.6e9,
        "speed": 0.0,
        "batch_size": 256,
        "max_mc_iter": 1,
        "num_target_block_errors": 10,
        "target_bler": 1e-2,
        **(params or {}),
    }

    # Build system with the chosen settings
    system = System(
        direction=cfg["direction"],
        perfect_csi=cfg["perfect_csi"],
        cdl_model=cfg["cdl_model"],
        delay_spread=cfg["delay_spread"],
        carrier_frequency=cfg["carrier_frequency"],
        speed=cfg["speed"],
        use_neural_rx=cfg["use_neural_rx"],
    )

    # Run Monte Carlo
    ber, bler = sim_ber(
        system,
        list(cfg["ebno_db"]),
        batch_size=cfg["batch_size"],
        max_mc_iter=cfg["max_mc_iter"],
        num_target_block_errors=cfg["num_target_block_errors"],
        target_bler=cfg["target_bler"],
    )
    return ber, bler


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # Base params shared by all CDL models
    base_params = dict(
        direction="uplink",
        perfect_csi=True,
        delay_spread=100e-9,
        batch_size=256,
        max_mc_iter=1,
        num_target_block_errors=10,
        target_bler=1e-2,
        use_neural_rx=False,
        ebno_db=[-5, -1, 3, 7, 11, 15],
    )

    cdl_models = ["A", "B", "C", "D", "E"]
    results = {"models": [], "ber": [], "bler": [], "ebno_db": base_params["ebno_db"]}

    t0 = time.time()
    for model in cdl_models:
        params = {**base_params, "cdl_model": model}
        ber, bler = run_sim(params)
        results["models"].append(model)
        results["ber"].append(ber.numpy())
        results["bler"].append(bler.numpy())
        print(f"CDL-{model}: BLER={np.array2string(bler.numpy(), precision=3)}")

    dur = time.time() - t0
    print(f"Total duration: {dur/60:.2f} min")

    # Plot
    os.makedirs("sim/results", exist_ok=True)
    plt.figure()
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.title("8x4 MIMO Uplink - Frequency Domain Modeling")
    plt.ylim([1e-3, 1.1])
    for model, bler in zip(results["models"], results["bler"]):
        plt.semilogy(results["ebno_db"], bler, label=f"CDL-{model}")
    plt.legend()
    save_path = "sim/results/ul_bler_all_cdl.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
