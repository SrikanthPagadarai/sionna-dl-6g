#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
DPD inference and evaluation script.

This script evaluates trained DPD models by computing standard linearization
metrics and saving data for visualization.

Evaluation Metrics
------------------
**ACLR (Adjacent Channel Leakage Ratio):**
    Measures out-of-band spectral regrowth.
    Computed as ratio of adjacent channel power to main channel power.

**NMSE (Normalized Mean Square Error):**
    Measures overall signal distortion.
    Computed between PA input and gain-normalized PA output.

**EVM (Error Vector Magnitude):**
    Measures in-band constellation distortion.
    Computed from demodulated QAM symbols vs reference.

Workflow
--------
1. Load trained DPD weights from ``results/`` directory
2. Generate test signals using the same OFDM configuration as training
3. Run inference with and without DPD
4. Compute metrics (ACLR, NMSE, EVM)
5. Save PSD and constellation data for plotting

Output Files
------------
- ``results/psd_data_{method}.npz`` : Power spectral density data
- ``results/constellation_data_{method}.npz`` : Demodulated symbol data
- ``results/inference_results_{method}.npz`` : Summary metrics

Usage
-----
::

    # Evaluate Neural Network DPD
    python inference.py --dpd_method nn --batch_size 16

    # Evaluate Least-Squares DPD
    python inference.py --dpd_method ls --batch_size 16

After running, use ``plots_nn.py`` or ``plots_ls.py`` to generate figures.

See Also
--------
training_nn.py : Neural network DPD training script.
training_ls.py : Least-squares DPD training script.
plots_nn.py : Plotting script for NN-DPD results.
plots_ls.py : Plotting script for LS-DPD results.
"""

import os

# Suppress TensorFlow logging before import.
# Level 3 = ERROR only (no INFO/WARNING).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings  # noqa: E402

# Suppress Sionna complex64->float32 cast warnings (expected behavior).
warnings.filterwarnings("ignore", message=".*complex64.*float32.*")

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")

# Configure GPU memory growth to avoid OOM on multi-tenant systems.
# Must happen before any TensorFlow operations.
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

import numpy as np  # noqa: E402
import pickle  # noqa: E402
import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from scipy.signal import welch  # noqa: E402
from scipy.signal.windows import kaiser  # noqa: E402

from demos.dpd.src.config import Config  # noqa: E402
from demos.dpd.src.nn_dpd_system import NN_DPDSystem  # noqa: E402
from demos.dpd.src.ls_dpd_system import LS_DPDSystem  # noqa: E402


# =============================================================================
# Command-line argument parsing
# =============================================================================
parser = argparse.ArgumentParser(
    description="DPD Inference and Evaluation",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python inference.py --dpd_method nn   # Neural Network DPD
  python inference.py --dpd_method ls   # Least-Squares DPD
    """,
)
parser.add_argument(
    "--dpd_method",
    type=str,
    default="nn",
    choices=["nn", "ls"],
    help="DPD method: 'nn' (Neural Network) or 'ls' (Least-Squares)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Batch size for signal generation (default: 16)",
)
args = parser.parse_args()

# =============================================================================
# Configuration constants
# =============================================================================
DPD_METHOD = args.dpd_method
BATCH_SIZE = args.batch_size
PA_SAMPLE_RATE = 122.88e6  # PA operating sample rate (Hz)
CHANNEL_BW = 10e6  # LTE channel bandwidth for ACLR calculation (Hz)

# Display labels and file names based on method.
DPD_LABEL = "NN DPD" if DPD_METHOD == "nn" else "LS DPD"
WEIGHT_FILE = "nn-dpd-weights" if DPD_METHOD == "nn" else "ls-dpd-weights"

# Output directory for results.
DEMO_DIR = Path(__file__).parent
outdir = DEMO_DIR / "results"
outdir.mkdir(parents=True, exist_ok=True)


def compute_aclr(signal, sample_rate, channel_bw=10e6):
    """
    Compute Adjacent Channel Leakage Ratio (ACLR).

    ACLR measures the power leaked into adjacent channels relative to the
    main channel power. It is a key metric for PA linearization compliance
    with 3GPP specifications.

    Parameters
    ----------
    signal : tf.Tensor or np.ndarray
        Complex baseband signal, any shape (will be flattened).
    sample_rate : float
        Signal sample rate in Hz.
    channel_bw : float, optional
        Main channel bandwidth in Hz. Default: 10 MHz (LTE).

    Returns
    -------
    aclr_lower : float
        Lower adjacent channel leakage ratio in dB.
    aclr_upper : float
        Upper adjacent channel leakage ratio in dB.

    Notes
    -----
    Channel definitions (centered at DC):

    - Main channel: ``[-BW/2, +BW/2]``
    - Lower adjacent: ``[-1.5*BW, -0.5*BW]``
    - Upper adjacent: ``[+0.5*BW, +1.5*BW]``

    Typical 3GPP ACLR requirements are -45 dBc or better.
    DPD typically improves ACLR by 10-20 dB.
    """
    signal_flat = tf.reshape(signal, [-1]).numpy()

    # Compute two-sided PSD using Welch's method.
    f, psd = welch(signal_flat, fs=sample_rate, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)

    # Define frequency masks for main and adjacent channels.
    main_mask = np.abs(f) <= channel_bw / 2
    lower_mask = (f >= -1.5 * channel_bw) & (f < -0.5 * channel_bw)
    upper_mask = (f > 0.5 * channel_bw) & (f <= 1.5 * channel_bw)

    # Integrate power in each band.
    main_power = np.sum(psd[main_mask])
    lower_power = np.sum(psd[lower_mask])
    upper_power = np.sum(psd[upper_mask])

    # ACLR = 10*log10(adjacent_power / main_power).
    aclr_lower = 10 * np.log10(lower_power / main_power) if main_power > 0 else -np.inf
    aclr_upper = 10 * np.log10(upper_power / main_power) if main_power > 0 else -np.inf

    return aclr_lower, aclr_upper


def compute_nmse(reference, signal):
    """
    Compute Normalized Mean Square Error (NMSE) in dB.

    NMSE measures the overall distortion between two signals after
    gain alignment. It captures both linear and nonlinear distortion.

    Parameters
    ----------
    reference : tf.Tensor
        Reference signal (typically PA input).
    signal : tf.Tensor
        Signal to compare (typically PA output).

    Returns
    -------
    float
        NMSE in dB. More negative values indicate less distortion.

    Notes
    -----
    The signal is scaled to match reference power before computing error.
    This isolates nonlinear distortion from linear gain differences.

    Formula: ``NMSE = 10*log10( E[|x_scaled - x_ref|^2] / E[|x_ref|^2] )``

    Typical values:

    - No DPD: -20 to -25 dB
    - With DPD: -35 to -45 dB
    """
    ref_flat = tf.reshape(reference, [-1])
    sig_flat = tf.reshape(signal, [-1])

    # Compute optimal scaling to match reference power.
    scale = tf.sqrt(
        tf.reduce_mean(tf.abs(ref_flat) ** 2)
        / (tf.reduce_mean(tf.abs(sig_flat) ** 2) + 1e-12)
    )
    sig_scaled = sig_flat * tf.cast(scale, sig_flat.dtype)

    # Compute normalized error.
    error = sig_scaled - ref_flat
    nmse = tf.reduce_mean(tf.abs(error) ** 2) / tf.reduce_mean(tf.abs(ref_flat) ** 2)

    return 10 * np.log10(nmse.numpy())


def save_psd_data(
    pa_input,
    pa_output_no_dpd,
    pa_output_with_dpd,
    sample_rate,
    save_path,
    channel_bw=10e6,
):
    """
    Compute and save power spectral density data for plotting.

    PSDs are normalized to 0 dBc (in-band power = 0 dB) for easy comparison
    of out-of-band suppression between signals with different power levels.

    Parameters
    ----------
    pa_input : tf.Tensor
        PA input signal (reference), shape ``[batch, samples]``.
    pa_output_no_dpd : tf.Tensor
        PA output without DPD, shape ``[batch, samples]``.
    pa_output_with_dpd : tf.Tensor
        PA output with DPD, shape ``[batch, samples]``.
    sample_rate : float
        Signal sample rate in Hz.
    save_path : Path or str
        Output file path for ``.npz`` file.
    channel_bw : float, optional
        Channel bandwidth for normalization. Default: 10 MHz.

    Notes
    -----
    PSDs are computed per-batch and averaged to avoid artifacts from
    discontinuities at batch boundaries. Uses Kaiser window (beta=9) for
    good sidelobe suppression.

    Output file contains:

    - ``freqs_mhz`` : Frequency axis in MHz
    - ``psd_input_db`` : PA input PSD in dBc
    - ``psd_no_dpd_db`` : PA output (no DPD) PSD in dBc
    - ``psd_with_dpd_db`` : PA output (with DPD) PSD in dBc
    """

    def to_numpy(x):
        """Convert tensor to numpy, preserving batch dimension."""
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.array(x)

    def compute_psd_averaged(signal):
        """
        Compute batch-averaged PSD using Welch's method.

        Averaging over batches reduces variance and avoids artifacts
        from batch boundary discontinuities.
        """
        window = kaiser(1000, 9)  # Kaiser window with beta=9.
        signal = to_numpy(signal)

        # Handle both batched [B, N] and flat [N] inputs.
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)

        # Compute PSD for each batch element and average.
        psds = []
        for i in range(signal.shape[0]):
            f, psd = welch(
                signal[i],
                fs=sample_rate,
                window=window,
                nfft=1024,
                return_onesided=False,
            )
            psds.append(psd)

        psd_avg = np.mean(psds, axis=0)
        return np.fft.fftshift(f), np.fft.fftshift(psd_avg)

    def compute_inband_power(f, psd, bw):
        """Compute mean in-band PSD for normalization to 0 dBc."""
        inband_mask = np.abs(f) <= bw / 2
        return np.mean(psd[inband_mask])

    # Compute averaged PSDs for all three signals.
    f_input, psd_input = compute_psd_averaged(pa_input)
    f_no_dpd, psd_no_dpd = compute_psd_averaged(pa_output_no_dpd)
    f_with_dpd, psd_with_dpd = compute_psd_averaged(pa_output_with_dpd)

    # Normalize each PSD so in-band power = 0 dBc.
    inband_power_input = compute_inband_power(f_input, psd_input, channel_bw)
    inband_power_no_dpd = compute_inband_power(f_no_dpd, psd_no_dpd, channel_bw)
    inband_power_with_dpd = compute_inband_power(f_with_dpd, psd_with_dpd, channel_bw)

    psd_input_norm = psd_input / inband_power_input
    psd_no_dpd_norm = psd_no_dpd / inband_power_no_dpd
    psd_with_dpd_norm = psd_with_dpd / inband_power_with_dpd

    # Convert to dB scale.
    freqs_mhz = f_input / 1e6
    psd_input_db = 10 * np.log10(psd_input_norm + 1e-12)
    psd_no_dpd_db = 10 * np.log10(psd_no_dpd_norm + 1e-12)
    psd_with_dpd_db = 10 * np.log10(psd_with_dpd_norm + 1e-12)

    # Save to compressed numpy archive.
    np.savez(
        save_path,
        freqs_mhz=freqs_mhz,
        psd_input_db=psd_input_db,
        psd_no_dpd_db=psd_no_dpd_db,
        psd_with_dpd_db=psd_with_dpd_db,
    )


def save_constellation_data(
    eval_system,
    pa_input,
    pa_output_no_dpd,
    pa_output_with_dpd,
    tx_baseband,
    fd_symbols,
    save_path,
):
    """
    Demodulate signals and save constellation data for plotting.

    Uses a minimal OFDM receiver to perform complete receiver-side processing
    including downsampling, synchronization, FFT, and equalization.

    Parameters
    ----------
    eval_system : NN_DPDSystem or LS_DPDSystem
        DPD system instance (provides access to receiver).
    pa_input : tf.Tensor
        PA input signal at PA sample rate.
    pa_output_no_dpd : tf.Tensor
        PA output without DPD.
    pa_output_with_dpd : tf.Tensor
        PA output with DPD.
    tx_baseband : tf.Tensor
        Baseband transmit signal (for receiver synchronization).
    fd_symbols : tf.Tensor
        Transmitted frequency-domain symbols (reference for EVM).
    save_path : Path or str
        Output file path for ``.npz`` file.

    Returns
    -------
    evm_input : float
        EVM of PA input path (sanity check, should be near 0%).
    evm_no_dpd : float
        EVM of PA output without DPD.
    evm_with_dpd : float
        EVM of PA output with DPD.

    Notes
    -----
    Output file contains:

    - ``fd_symbols`` : Reference transmitted symbols
    - ``sym_input`` : Demodulated symbols from PA input
    - ``sym_no_dpd`` : Demodulated symbols from PA output (no DPD)
    - ``sym_with_dpd`` : Demodulated symbols from PA output (with DPD)
    - ``evm_input/no_dpd/with_dpd`` : EVM values in percent
    """
    # Use a minimal OFDM receiver to demodulate all signals and compute EVM.
    result = eval_system.minimal_ofdm_receiver.process_and_compute_evm(
        pa_input=pa_input,
        pa_output_no_dpd=pa_output_no_dpd,
        pa_output_with_dpd=pa_output_with_dpd,
        tx_baseband=tx_baseband,
        fd_symbols=fd_symbols,
    )

    # Convert reference symbols to numpy for saving.
    fd_np = fd_symbols.numpy() if isinstance(fd_symbols, tf.Tensor) else fd_symbols

    # Save all constellation data to compressed archive.
    np.savez(
        save_path,
        fd_symbols=fd_np,
        sym_input=result["symbols_input"],
        sym_no_dpd=result["symbols_no_dpd"],
        sym_with_dpd=result["symbols_with_dpd"],
        evm_input=result["evm_input"],
        evm_no_dpd=result["evm_no_dpd"],
        evm_with_dpd=result["evm_with_dpd"],
    )

    return result["evm_input"], result["evm_no_dpd"], result["evm_with_dpd"]


"""
Main inference workflow.

Steps:
    1. Build evaluation system (NN or LS DPD)
    2. Load trained weights
    3. Generate test signals and run inference
    4. Compute metrics (ACLR, NMSE, EVM)
    5. Save data for plotting
    6. Print summary table
"""
print("=" * 70)
print(f"{DPD_LABEL} Inference")
print("=" * 70)

# Create configuration with specified batch size.
config = Config(batch_size=BATCH_SIZE)

# -------------------------------------------------------------------------
# Step 1: Build evaluation system
# -------------------------------------------------------------------------
print(f"\n[1] Building evaluation system with {DPD_LABEL}...")

if DPD_METHOD == "nn":
    eval_system = NN_DPDSystem(
        training=False,  # Inference mode.
        config=config,
        dpd_memory_depth=4,
        dpd_num_filters=64,
        dpd_num_layers_per_block=2,
        dpd_num_res_blocks=3,
        rms_input_dbm=0.5,
        pa_sample_rate=PA_SAMPLE_RATE,
    )
else:  # "ls"
    eval_system = LS_DPDSystem(
        training=False,  # Inference mode.
        config=config,
        dpd_order=7,
        dpd_memory_depth=4,
        rms_input_dbm=0.5,
        pa_sample_rate=PA_SAMPLE_RATE,
    )

# Warm-up pass to trigger lazy initialization.
x_warmup = eval_system.generate_signal(1)
_ = eval_system(x_warmup, training=False)

if DPD_METHOD == "nn":
    print(
        "    DPD parameters: ",
        f"{sum(tf.size(v).numpy() for v in eval_system.dpd.trainable_variables)}",
    )
else:
    print(f"    DPD coefficients: {eval_system.dpd.n_coeffs}")

# Estimate PA gain for normalization.
pa_gain = eval_system.estimate_pa_gain()
print(f"    Estimated PA gain: {pa_gain:.4f} ({20*np.log10(pa_gain):.2f} dB)")

# -------------------------------------------------------------------------
# Step 2: Load trained weights
# -------------------------------------------------------------------------
print("\n[2] Loading trained weights...")
weight_file = outdir / WEIGHT_FILE
if weight_file.exists():
    with open(weight_file, "rb") as f:
        weights = pickle.load(f)
        eval_system.set_weights(weights)
    print(f"    Loaded weights from {weight_file}")
else:
    print(f"    WARNING: Weight file not found at {weight_file}")
    print("    Using untrained model (identity DPD)")

# -------------------------------------------------------------------------
# Step 3: Run inference
# -------------------------------------------------------------------------
print("\n[3] Running inference...")
# Generate signal with extras for constellation analysis.
signal_data = eval_system.generate_signal(BATCH_SIZE, return_extras=True)
pa_input = signal_data["tx_upsampled"]
tx_baseband = signal_data["tx_baseband"]
fd_symbols = signal_data["fd_symbols"]

results = eval_system(pa_input, training=False)
pa_output_no_dpd = results["pa_output_no_dpd"]
pa_output_with_dpd = results["pa_output_with_dpd"]

print(f"    Signal shape: {pa_input.shape}")

# -------------------------------------------------------------------------
# Step 4: Compute ACLR
# -------------------------------------------------------------------------
print("\n[4] Computing ACLR...")
aclr_l_no, aclr_u_no = compute_aclr(pa_output_no_dpd, PA_SAMPLE_RATE, CHANNEL_BW)
aclr_l_dpd, aclr_u_dpd = compute_aclr(pa_output_with_dpd, PA_SAMPLE_RATE, CHANNEL_BW)

print(f"    ACLR (No DPD):   Lower = {aclr_l_no:.2f} dB, Upper = {aclr_u_no:.2f} dB")
print(
    f"    ACLR ({DPD_LABEL}):  "
    f"Lower = {aclr_l_dpd:.2f} dB, "
    f"Upper = {aclr_u_dpd:.2f} dB"
)

aclr_improvement = ((aclr_l_no - aclr_l_dpd) + (aclr_u_no - aclr_u_dpd)) / 2
print(f"    ACLR Improvement: {aclr_improvement:.2f} dB average")

# -------------------------------------------------------------------------
# Step 5: Compute NMSE
# -------------------------------------------------------------------------
print("\n[5] Computing NMSE...")
nmse_no_dpd = compute_nmse(pa_input, pa_output_no_dpd)
nmse_with_dpd = compute_nmse(pa_input, pa_output_with_dpd)

print(f"    NMSE (No DPD):   {nmse_no_dpd:.2f} dB")
print(f"    NMSE ({DPD_LABEL}):  {nmse_with_dpd:.2f} dB")
print(f"    NMSE Improvement: {nmse_no_dpd - nmse_with_dpd:.2f} dB")

# -------------------------------------------------------------------------
# Step 6: Save PSD data
# -------------------------------------------------------------------------
print("\n[6] Saving PSD data...")
psd_data_path = outdir / f"psd_data_{DPD_METHOD}.npz"
save_psd_data(
    pa_input,
    pa_output_no_dpd,
    pa_output_with_dpd,
    PA_SAMPLE_RATE,
    psd_data_path,
    channel_bw=CHANNEL_BW,
)
print(f"    Saved to {psd_data_path}")

# -------------------------------------------------------------------------
# Step 7: Save constellation data
# -------------------------------------------------------------------------
print("\n[7] Saving constellation data...")
const_data_path = outdir / f"constellation_data_{DPD_METHOD}.npz"
try:
    evm_input, evm_no_dpd, evm_with_dpd = save_constellation_data(
        eval_system,
        pa_input,
        pa_output_no_dpd,
        pa_output_with_dpd,
        tx_baseband,
        fd_symbols,
        const_data_path,
    )
    print(f"    Saved to {const_data_path}")
    print(f"    EVM (PA Input):  {evm_input:.2f}%")
    print(f"    EVM (No DPD):    {evm_no_dpd:.2f}%")
    print(f"    EVM ({DPD_LABEL}):   {evm_with_dpd:.2f}%")
except Exception as e:
    print(f"    Constellation data save failed: {e}")
    import traceback

    traceback.print_exc()
    evm_input, evm_no_dpd, evm_with_dpd = None, None, None

# -------------------------------------------------------------------------
# Summary table
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"\n{'Metric':<25} {'No DPD':>15} {DPD_LABEL:>15} {'Improvement':>15}")
print("-" * 70)
print(
    f"{'ACLR Lower (dB)':<25} "
    f"{aclr_l_no:>15.2f} "
    f"{aclr_l_dpd:>15.2f} "
    f"{aclr_l_no - aclr_l_dpd:>15.2f}"
)
print(
    f"{'ACLR Upper (dB)':<25} "
    f"{aclr_u_no:>15.2f} "
    f"{aclr_u_dpd:>15.2f} "
    f"{aclr_u_no - aclr_u_dpd:>15.2f}"
)
print(
    f"{'NMSE (dB)':<25} "
    f"{nmse_no_dpd:>15.2f} "
    f"{nmse_with_dpd:>15.2f} "
    f"{nmse_no_dpd - nmse_with_dpd:>15.2f}"
)
if evm_no_dpd is not None:
    print(
        f"{'EVM (%)':<25} "
        f"{evm_no_dpd:>15.2f} "
        f"{evm_with_dpd:>15.2f} "
        f"{evm_no_dpd - evm_with_dpd:>15.2f}"
    )
print("-" * 70)

# Save summary results.
results_file = outdir / f"inference_results_{DPD_METHOD}.npz"
np.savez(
    results_file,
    dpd_method=DPD_METHOD,
    aclr_lower_no_dpd=aclr_l_no,
    aclr_upper_no_dpd=aclr_u_no,
    aclr_lower_dpd=aclr_l_dpd,
    aclr_upper_dpd=aclr_u_dpd,
    nmse_no_dpd=nmse_no_dpd,
    nmse_with_dpd=nmse_with_dpd,
    evm_input=evm_input if evm_input else 0,
    evm_no_dpd=evm_no_dpd if evm_no_dpd else 0,
    evm_with_dpd=evm_with_dpd if evm_with_dpd else 0,
)
print(f"\nResults saved to {results_file}")
print("\nData files saved:")
print(f"  - {psd_data_path}")
print(f"  - {const_data_path}")
plot_script = "plots_nn.py" if DPD_METHOD == "nn" else "plots_ls.py"
print(f"\nRun 'python {plot_script}' to generate plots.")
