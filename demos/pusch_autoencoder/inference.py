import os
import sys
import tensorflow as tf

# ---------------------------------------------------------------------------
# TensorFlow / GPU setup
# ---------------------------------------------------------------------------
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

# Disable layout optimizer to avoid ConvLSTM graph cycles at inference time
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

import pickle  # noqa: E402
import numpy as np  # noqa: E402
from sionna.phy.utils import PlotBER  # noqa: E402

from src.config import Config  # noqa: E402
from src.system import PUSCHLinkE2E  # noqa: E402
from src.cir_manager import CIRManager  # noqa: E402


# ---------------------------------------------------------------------------
# Mode selection based on CLI argument
# ---------------------------------------------------------------------------
if len(sys.argv) != 2 or sys.argv[1] not in ("conventional", "two_phase"):
    print("Usage: python inference.py [conventional|two_phase]")
    sys.exit(1)

mode = sys.argv[1]


# ---------------------------------------------------------------------------
# Helper: build model, load weights, restore constellation
# ---------------------------------------------------------------------------
def load_model_weights(
    model: tf.keras.Model, weights_path: str, batch_size: int
) -> bool:
    """
    Build the model, load weights from a pickle file, and restore the
    trainable variables directly.

    Returns:
        True  if weights were successfully loaded,
        False if the weights file was not found (model left with random init).
    """
    # Build model first
    ebno_db_build = tf.fill([batch_size], 10.0)
    _ = model(tf.cast(batch_size, tf.int32), ebno_db_build)

    if not os.path.exists(weights_path):
        print(
            f"[WARN] Weights file not found at '{weights_path}'. "
            "Running inference with randomly initialized weights."
        )
        return False

    with open(weights_path, "rb") as f:
        weights_dict = pickle.load(f)

    # Load TX weights
    tx_vars = model._pusch_transmitter.trainable_variables
    for var, arr in zip(tx_vars, weights_dict["tx_weights"]):
        var.assign(arr)
    print(f"[INFO] Restored {len(tx_vars)} TX variables.")

    # Load RX weights
    rx_vars = model._pusch_receiver.trainable_variables
    for var, arr in zip(rx_vars, weights_dict["rx_weights"]):
        var.assign(arr)
    print(f"[INFO] Restored {len(rx_vars)} RX variables.")

    # Sync constellation object if present
    tx = model._pusch_transmitter
    if hasattr(tx, "_points_r") and hasattr(tx, "_points_i"):
        pts = tf.complex(tx._points_r, tx._points_i)
        if hasattr(tx, "_constellation"):
            tx._constellation.points = pts
        print("[INFO] Synced trainable constellation points.")

    print(f"[INFO] Loaded weights from '{weights_path}'.")
    return True


# ---------------------------------------------------------------------------
# Configuration & channel model
# ---------------------------------------------------------------------------
_cfg = Config()
batch_size = _cfg.batch_size
num_ue = _cfg.num_ue
num_bs_ant = _cfg.num_bs_ant
num_ue_ant = _cfg.num_ue_ant
num_time_steps = _cfg.num_time_steps

cir_manager = CIRManager()
# Same as training.py: use raw (a, tau) tensors from TFRecord.
channel_model = cir_manager.load_from_tfrecord(group_for_mumimo=True)

# ---------------------------------------------------------------------------
# Instantiate model (autoencoder, inference mode)
# ---------------------------------------------------------------------------
e2e_model = PUSCHLinkE2E(
    channel_model,
    perfect_csi=False,
    use_autoencoder=True,
    training=False,  # inference mode, but architecture same as in training
)  # :contentReference[oaicite:5]{index=5}

# Select weights file based on mode
if mode == "conventional":
    weights_filename = "PUSCH_autoencoder_weights_conventional_training"
else:  # mode == "two_phase"
    weights_filename = "PUSCH_autoencoder_weights_two_phase_training"

weights_path = os.path.join("results", weights_filename)
_ = load_model_weights(e2e_model, weights_path, batch_size)

# ---------------------------------------------------------------------------
# Quick functional check (match training ebno_db shape)
# ---------------------------------------------------------------------------
ebno_db_test = tf.fill([batch_size], 10.0)  # vector, same shape as training
b_test, b_hat_test = e2e_model(batch_size, ebno_db_test)
print("Quick check shapes (autoencoder inference):", b_test.shape, b_hat_test.shape)


# ---------------------------------------------------------------------------
# Wrapper for PlotBER: always pass ebno_db as [batch_size]
# ---------------------------------------------------------------------------
def ae_model_for_ber(batch_size, ebno_db):
    """
    Adapter so that PlotBER.simulate can pass scalar Eb/N0, while the
    underlying PUSCHLinkE2E always sees a [batch_size] vector, as in training.
    """
    # Convert to tf.float32
    ebno_db = tf.cast(ebno_db, tf.float32)

    # If scalar, expand to [batch_size]; if already a vector, leave as is.
    if ebno_db.shape.rank == 0:
        ebno_vec = tf.fill([batch_size], ebno_db)
    else:
        ebno_vec = ebno_db

    # Call your actual model
    return e2e_model(batch_size, ebno_vec)


# ---------------------------------------------------------------------------
# BER/BLER Simulation
# ---------------------------------------------------------------------------
ebno_db = np.arange(-2, 10, 1)

ber_plot = PlotBER("PUSCH Autoencoder Inference (Trained)")

# NOTE: we pass ae_model_for_ber (the wrapper), not e2e_model directly.
ber, bler = ber_plot.simulate(
    ae_model_for_ber,
    ebno_dbs=ebno_db,
    max_mc_iter=50,
    num_target_block_errors=200,
    batch_size=batch_size,
    soft_estimates=False,
    show_fig=False,
    add_bler=True,
)

# Ensure NumPy arrays
if hasattr(ber, "numpy"):
    ber = ber.numpy()
if hasattr(bler, "numpy"):
    bler = bler.numpy()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

if mode == "conventional":
    results_filename = "inference_results_conventional.npz"
else:  # mode == "two_phase"
    results_filename = "inference_results_two_phase.npz"

out_path = os.path.join("results", results_filename)

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
    perfect_csi=False,
    use_autoencoder=True,
    training=False,
)

print(f"Saved autoencoder BER/BLER inference results to {out_path}")
