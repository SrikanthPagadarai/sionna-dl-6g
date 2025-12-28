#!/usr/bin/env python3
"""
Inference script for DPD evaluation.

Supports both Neural Network (nn) and Least-Squares (ls) DPD methods.

Demonstrates:
- Out-of-band suppression (ACLR improvement)
- In-band distortion reduction (NMSE/EVM improvement)
- PSD comparison plots
- Overlapped constellation plots showing 16-QAM

Usage:
    python inference.py --dpd_method nn   # Neural Network DPD
    python inference.py --dpd_method ls   # Least-Squares DPD
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*complex64.*float32.*")

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")

# GPU setup - must happen before any TensorFlow operations
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

import numpy as np  # noqa: E402
import pickle  # noqa: E402
import argparse  # noqa: E402
from pathlib import Path  # noqa: E402
from fractions import Fraction  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import welch, resample_poly  # noqa: E402
from scipy.signal.windows import kaiser  # noqa: E402

from src.system import DPDSystem  # noqa: E402


# CLI Arguments
parser = argparse.ArgumentParser(description="DPD Inference and Evaluation")
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

# Configuration
DPD_METHOD = args.dpd_method
BATCH_SIZE = args.batch_size
PA_SAMPLE_RATE = 122.88e6
CHANNEL_BW = 10e6

# Method display name
DPD_LABEL = "NN DPD" if DPD_METHOD == "nn" else "LS DPD"
WEIGHT_FILE = "nn-dpd-weights" if DPD_METHOD == "nn" else "ls-dpd-weights"

# Folders
outdir = Path("results")
outdir.mkdir(parents=True, exist_ok=True)


def compute_aclr(signal, sample_rate, channel_bw=10e6):
    """
    Compute Adjacent Channel Leakage Ratio (ACLR).

    Returns:
        Tuple of (ACLR_lower, ACLR_upper) in dB
    """
    signal_flat = tf.reshape(signal, [-1]).numpy()

    f, psd = welch(signal_flat, fs=sample_rate, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)

    main_mask = np.abs(f) <= channel_bw / 2
    lower_mask = (f >= -1.5 * channel_bw) & (f < -0.5 * channel_bw)
    upper_mask = (f > 0.5 * channel_bw) & (f <= 1.5 * channel_bw)

    main_power = np.sum(psd[main_mask])
    lower_power = np.sum(psd[lower_mask])
    upper_power = np.sum(psd[upper_mask])

    aclr_lower = 10 * np.log10(lower_power / main_power) if main_power > 0 else -np.inf
    aclr_upper = 10 * np.log10(upper_power / main_power) if main_power > 0 else -np.inf

    return aclr_lower, aclr_upper


def compute_nmse(reference, signal):
    """
    Compute Normalized Mean Square Error (NMSE) in dB.
    """
    ref_flat = tf.reshape(reference, [-1])
    sig_flat = tf.reshape(signal, [-1])

    # Scale signal to match reference power
    scale = tf.sqrt(
        tf.reduce_mean(tf.abs(ref_flat) ** 2)
        / (tf.reduce_mean(tf.abs(sig_flat) ** 2) + 1e-12)
    )
    sig_scaled = sig_flat * tf.cast(scale, sig_flat.dtype)

    error = sig_scaled - ref_flat
    nmse = tf.reduce_mean(tf.abs(error) ** 2) / tf.reduce_mean(tf.abs(ref_flat) ** 2)

    return 10 * np.log10(nmse.numpy())


def plot_psd_comparison(
    pa_input,
    pa_output_no_dpd,
    pa_output_with_dpd,
    sample_rate,
    save_path,
    channel_bw=10e6,
    dpd_label="DPD",
):
    """Plot PSD comparison normalized to 0 dBc (in-band power = 0 dB).

    Computes PSD per-batch and averages to avoid artifacts from
    discontinuities at batch boundaries.
    """

    def to_numpy(x):
        """Convert to numpy, keeping batch dimension."""
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.array(x)

    def compute_psd_averaged(signal):
        """Compute averaged PSD over batches using Welch method.

        Args:
            signal: [B, N] batched signal

        Returns:
            f: frequency array (shifted)
            psd: averaged PSD (shifted)
        """
        window = kaiser(1000, 9)
        signal = to_numpy(signal)

        # Handle both batched [B, N] and flat [N] inputs
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)

        # Compute PSD for each batch and average
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

    # Compute averaged PSDs
    f_input, psd_input = compute_psd_averaged(pa_input)
    f_no_dpd, psd_no_dpd = compute_psd_averaged(pa_output_no_dpd)
    f_with_dpd, psd_with_dpd = compute_psd_averaged(pa_output_with_dpd)

    # Normalize each PSD to 0 dBc (in-band power = 0 dB)
    inband_power_input = compute_inband_power(f_input, psd_input, channel_bw)
    inband_power_no_dpd = compute_inband_power(f_no_dpd, psd_no_dpd, channel_bw)
    inband_power_with_dpd = compute_inband_power(f_with_dpd, psd_with_dpd, channel_bw)

    psd_input_norm = psd_input / inband_power_input
    psd_no_dpd_norm = psd_no_dpd / inband_power_no_dpd
    psd_with_dpd_norm = psd_with_dpd / inband_power_with_dpd

    # Convert to dB
    freqs_mhz = f_input / 1e6
    psd_input_db = 10 * np.log10(psd_input_norm + 1e-12)
    psd_no_dpd_db = 10 * np.log10(psd_no_dpd_norm + 1e-12)
    psd_with_dpd_db = 10 * np.log10(psd_with_dpd_norm + 1e-12)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs_mhz, psd_input_db, label="PA Input (Reference)", alpha=0.8)
    plt.plot(freqs_mhz, psd_no_dpd_db, label="PA Output (No DPD)", alpha=0.8)
    plt.plot(freqs_mhz, psd_with_dpd_db, label=f"PA Output ({dpd_label})", alpha=0.8)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dBc)")
    plt.title(f"Power Spectral Density: Effect of {dpd_label}")
    plt.legend()
    plt.grid(True)
    plt.ylim([-120, 10])  # Typical range for DPD plots
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_constellation(
    eval_system,
    pa_input,
    pa_output_no_dpd,
    pa_output_with_dpd,
    tx_baseband,
    fd_symbols,
    save_path,
    dpd_label="DPD",
):
    """
    Plot overlapped constellation comparison showing 16-QAM points.

    Uses proper OFDM demodulation and per-subcarrier equalization.
    """
    signal_fs = eval_system.signal_fs
    pa_sample_rate = eval_system.pa_sample_rate
    fft_size = eval_system.fft_size
    cp_length = eval_system.cp_length
    num_symbols = eval_system.num_ofdm_symbols

    # Flatten all signals
    def flatten(x):
        if len(x.shape) > 1:
            return tf.reshape(x, [-1]).numpy()
        return x.numpy() if hasattr(x, "numpy") else x

    pa_input_flat = flatten(pa_input)
    pa_no_dpd_flat = flatten(pa_output_no_dpd)
    pa_with_dpd_flat = flatten(pa_output_with_dpd)
    tx_baseband_np = flatten(tx_baseband)

    # Downsample from PA rate to signal rate
    frac = Fraction(signal_fs / pa_sample_rate).limit_denominator(1000)
    data_input = resample_poly(pa_input_flat, frac.numerator, frac.denominator)
    data_no_dpd = resample_poly(pa_no_dpd_flat, frac.numerator, frac.denominator)
    data_with_dpd = resample_poly(pa_with_dpd_flat, frac.numerator, frac.denominator)

    original_len = (fft_size + cp_length) * num_symbols

    # Synchronize using cross-correlation
    sync_len = min(1000, len(tx_baseband_np) // 2)

    def find_delay(signal, ref):
        return np.argmax(np.abs(np.correlate(signal, ref[:sync_len], mode="valid")))

    delay_input = find_delay(data_input, tx_baseband_np)
    delay_no_dpd = find_delay(data_no_dpd, tx_baseband_np)
    delay_with_dpd = find_delay(data_with_dpd, tx_baseband_np)

    data_input_sync = data_input[delay_input : delay_input + original_len]
    data_no_dpd_sync = data_no_dpd[delay_no_dpd : delay_no_dpd + original_len]
    data_with_dpd_sync = data_with_dpd[delay_with_dpd : delay_with_dpd + original_len]

    # Demodulate using DPDSystem's demod method
    symbols_input = eval_system.demod(data_input_sync)
    symbols_no_dpd = eval_system.demod(data_no_dpd_sync)
    symbols_with_dpd = eval_system.demod(data_with_dpd_sync)

    # Per-subcarrier equalization
    def equalize(rx, tx):
        rx = tf.cast(rx, tf.complex64)
        tx = tf.cast(tx, tf.complex64)
        H = tf.reduce_sum(rx * tf.math.conj(tx), axis=1, keepdims=True) / tf.cast(
            tf.reduce_sum(tf.abs(tx) ** 2, axis=1, keepdims=True), tf.complex64
        )
        return rx / H

    symbols_input = equalize(symbols_input, fd_symbols)
    symbols_no_dpd = equalize(symbols_no_dpd, fd_symbols)
    symbols_with_dpd = equalize(symbols_with_dpd, fd_symbols)

    # Convert to numpy for plotting
    fd_np = fd_symbols.numpy() if isinstance(fd_symbols, tf.Tensor) else fd_symbols
    sym_input_np = symbols_input.numpy()
    sym_no_dpd_np = symbols_no_dpd.numpy()
    sym_with_dpd_np = symbols_with_dpd.numpy()

    # Compute EVM
    def compute_evm(rx, tx):
        error = rx - tx
        evm = np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(tx) ** 2)) * 100
        return evm

    evm_input = compute_evm(sym_input_np, fd_np)
    evm_no_dpd = compute_evm(sym_no_dpd_np, fd_np)
    evm_with_dpd = compute_evm(sym_with_dpd_np, fd_np)

    # Plot overlapped constellation
    plt.figure(figsize=(10, 10))
    plt.plot(
        fd_np.real.flatten(),
        fd_np.imag.flatten(),
        "o",
        ms=4,
        label="Original (TX)",
        alpha=0.5,
    )
    plt.plot(
        sym_input_np.real.flatten(),
        sym_input_np.imag.flatten(),
        "s",
        ms=3,
        label=f"PA Input (EVM={evm_input:.1f}%)",
        alpha=0.5,
    )
    plt.plot(
        sym_no_dpd_np.real.flatten(),
        sym_no_dpd_np.imag.flatten(),
        "x",
        ms=3,
        label=f"PA Output, no DPD (EVM={evm_no_dpd:.1f}%)",
        alpha=0.5,
    )
    plt.plot(
        sym_with_dpd_np.real.flatten(),
        sym_with_dpd_np.imag.flatten(),
        ".",
        ms=3,
        label=f"PA Output, with {dpd_label} (EVM={evm_with_dpd:.1f}%)",
        alpha=0.5,
    )
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(f"Constellation Comparison: Effect of {dpd_label}")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return evm_input, evm_no_dpd, evm_with_dpd


def main():
    print("=" * 70)
    print(f"{DPD_LABEL} Inference")
    print("=" * 70)

    # Build eval system
    print(f"\n[1] Building evaluation system with {DPD_LABEL}...")

    if DPD_METHOD == "nn":
        eval_system = DPDSystem(
            training=False,
            dpd_method="nn",
            tx_config_path="src/tx_config.json",
            pa_order=7,
            pa_memory_depth=4,
            dpd_memory_depth=4,
            dpd_num_filters=64,
            dpd_num_layers_per_block=2,
            dpd_num_res_blocks=3,
            rms_input_dbm=0.5,
            pa_sample_rate=PA_SAMPLE_RATE,
        )
    else:  # "ls"
        eval_system = DPDSystem(
            training=False,
            dpd_method="ls",
            tx_config_path="src/tx_config.json",
            pa_order=7,
            pa_memory_depth=4,
            dpd_order=7,
            dpd_memory_depth=4,
            rms_input_dbm=0.5,
            pa_sample_rate=PA_SAMPLE_RATE,
        )

    # Warm up
    x_warmup = eval_system.generate_signal(1)
    _ = eval_system(x_warmup, training=False)

    if DPD_METHOD == "nn":
        print(
            "    DPD parameters: ",
            f"{sum(tf.size(v).numpy() for v in eval_system.dpd.trainable_variables)}",
        )
    else:
        print(f"    DPD coefficients: {eval_system.dpd.n_coeffs}")

    # Estimate PA gain
    pa_gain = eval_system.estimate_pa_gain()
    print(f"    Estimated PA gain: {pa_gain:.4f} ({20*np.log10(pa_gain):.2f} dB)")

    # Load weights
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

    # Run inference with extras for constellation
    print("\n[3] Running inference...")
    signal_data = eval_system.generate_signal(BATCH_SIZE, return_extras=True)
    pa_input = signal_data["tx_upsampled"]
    tx_baseband = signal_data["tx_baseband"]
    fd_symbols = signal_data["fd_symbols"]

    results = eval_system(pa_input, training=False)
    pa_output_no_dpd = results["pa_output_no_dpd"]
    pa_output_with_dpd = results["pa_output_with_dpd"]

    print(f"    Signal shape: {pa_input.shape}")

    # Compute ACLR
    print("\n[4] Computing ACLR...")
    aclr_l_no, aclr_u_no = compute_aclr(pa_output_no_dpd, PA_SAMPLE_RATE, CHANNEL_BW)
    aclr_l_dpd, aclr_u_dpd = compute_aclr(
        pa_output_with_dpd, PA_SAMPLE_RATE, CHANNEL_BW
    )

    print(
        f"    ACLR (No DPD):   Lower = {aclr_l_no:.2f} dB, Upper = {aclr_u_no:.2f} dB"
    )
    print(
        f"    ACLR ({DPD_LABEL}):  "
        f"Lower = {aclr_l_dpd:.2f} dB, "
        f"Upper = {aclr_u_dpd:.2f} dB"
    )

    aclr_improvement = ((aclr_l_no - aclr_l_dpd) + (aclr_u_no - aclr_u_dpd)) / 2
    print(f"    ACLR Improvement: {aclr_improvement:.2f} dB average")

    # Compute NMSE
    print("\n[5] Computing NMSE...")
    nmse_no_dpd = compute_nmse(pa_input, pa_output_no_dpd)
    nmse_with_dpd = compute_nmse(pa_input, pa_output_with_dpd)

    print(f"    NMSE (No DPD):   {nmse_no_dpd:.2f} dB")
    print(f"    NMSE ({DPD_LABEL}):  {nmse_with_dpd:.2f} dB")
    print(f"    NMSE Improvement: {nmse_no_dpd - nmse_with_dpd:.2f} dB")

    # Generate PSD plot
    print("\n[6] Generating PSD comparison plot...")
    psd_path = outdir / f"psd_comparison_{DPD_METHOD}.png"
    plot_psd_comparison(
        pa_input,
        pa_output_no_dpd,
        pa_output_with_dpd,
        PA_SAMPLE_RATE,
        psd_path,
        channel_bw=CHANNEL_BW,
        dpd_label=DPD_LABEL,
    )
    print(f"    Saved to {psd_path}")

    # Generate constellation plot
    print("\n[7] Generating constellation plot...")
    const_path = outdir / f"constellation_comparison_{DPD_METHOD}.png"
    try:
        evm_input, evm_no_dpd, evm_with_dpd = plot_constellation(
            eval_system,
            pa_input,
            pa_output_no_dpd,
            pa_output_with_dpd,
            tx_baseband,
            fd_symbols,
            const_path,
            dpd_label=DPD_LABEL,
        )
        print(f"    Saved to {const_path}")
        print(f"    EVM (PA Input):  {evm_input:.2f}%")
        print(f"    EVM (No DPD):    {evm_no_dpd:.2f}%")
        print(f"    EVM ({DPD_LABEL}):   {evm_with_dpd:.2f}%")
    except Exception as e:
        print(f"    Constellation plot failed: {e}")
        import traceback

        traceback.print_exc()
        evm_input, evm_no_dpd, evm_with_dpd = None, None, None

    # Summary
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

    # Save results
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
    print("\nPlots saved:")
    print(f"  - {psd_path}")
    print(f"  - {const_path}")


if __name__ == "__main__":
    main()
