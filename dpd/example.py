#!/usr/bin/env python3
"""
Batched DPD Example: Compare PA output with and without digital predistortion.

Demonstrates:
1. Batched signal generation from Sionna
2. Batched upsampling using Interpolator
3. DPD learning using IndirectLearningDPD
4. PA processing using PowerAmplifier
5. PSD comparison plot showing PA input, PA output (no DPD), PA output (with DPD)
6. Verification of differentiability for future NN-based DPD
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import welch, resample_poly  # noqa: E402
from scipy.signal.windows import kaiser  # noqa: E402
from fractions import Fraction  # noqa: E402
import tensorflow as tf  # noqa: E402

# Suppress TF warnings
import logging  # noqa: E402

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from src.tx import build_dataset_from_tx  # noqa: E402
from sionna.phy.ofdm import OFDMDemodulator  # noqa: E402
from src.utilities import normalize_to_rms  # noqa: E402
from src.interpolator import Interpolator  # noqa: E402
from dpd.src.power_amplifier import PowerAmplifier  # noqa: E402
from dpd.src.indirect_learning_dpd import IndirectLearningDPD  # noqa: E402

os.makedirs("results", exist_ok=True)

# =============================================================================
# Configuration
# =============================================================================
CONFIG_PATH = "src/tx_config.json"
BATCH_SIZE = 16
RMS_INPUT = 0.50  # dBm
PA_SAMPLE_RATE = 122.88e6

DPD_PARAMS = {
    "order": 9,
    "memory_depth": 4,
    "lag_depth": 0,
    "nIterations": 6,
    "learning_rate": 0.9,
    "learning_method": "newton",
    "use_even": False,
    "use_conj": False,
    "use_dc_term": False,
}


def plot_psd_comparison_three(
    pa_input,
    pa_output_no_dpd,
    pa_output_with_dpd,
    sample_rate,
    save_path="results/psd_dpd_comparison.png",
):
    """
    Plot PSD comparison of PA input, PA output without DPD, and PA output with DPD.

    Args:
        pa_input: PA input tensor [B, num_samples] or [num_samples]
        pa_output_no_dpd: PA output without DPD
        pa_output_with_dpd: PA output with DPD
        sample_rate: Sample rate in Hz
        save_path: Path to save the plot
    """

    # Flatten if batched
    def flatten(x):
        if len(x.shape) > 1:
            return tf.reshape(x, [-1]).numpy()
        return x.numpy() if hasattr(x, "numpy") else x

    pa_input_flat = flatten(pa_input)
    pa_no_dpd_flat = flatten(pa_output_no_dpd)
    pa_with_dpd_flat = flatten(pa_output_with_dpd)

    # Use welch with same parameters as original
    window = kaiser(1000, 9)
    f, psd_input = welch(
        pa_input_flat, fs=sample_rate, window=window, nfft=1024, return_onesided=False
    )
    _, psd_no_dpd = welch(
        pa_no_dpd_flat, fs=sample_rate, window=window, nfft=1024, return_onesided=False
    )
    _, psd_with_dpd = welch(
        pa_with_dpd_flat,
        fs=sample_rate,
        window=window,
        nfft=1024,
        return_onesided=False,
    )

    freqs_mhz = np.fft.fftshift(f) / 1e6
    psd_input_db = 10 * np.log10(np.fft.fftshift(psd_input))
    psd_no_dpd_db = 10 * np.log10(np.fft.fftshift(psd_no_dpd))
    psd_with_dpd_db = 10 * np.log10(np.fft.fftshift(psd_with_dpd))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_mhz, psd_input_db, label="PA Input", alpha=0.8)
    plt.plot(freqs_mhz, psd_no_dpd_db, label="PA Output (no DPD)", alpha=0.8)
    plt.plot(freqs_mhz, psd_with_dpd_db, label="PA Output (with DPD)", alpha=0.8)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("PSD Comparison: Effect of Digital Predistortion")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"PSD plot saved to: {save_path}")


def compute_aclr(signal, sample_rate, channel_bw=10e6):
    """
    Compute Adjacent Channel Leakage Ratio (ACLR).

    Returns:
        Tuple of (ACLR_lower, ACLR_upper) in dB
    """
    signal_flat = tf.reshape(signal, [-1]).numpy()

    # Compute PSD
    f, psd = welch(signal_flat, fs=sample_rate, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)

    # Define channel boundaries
    main_mask = np.abs(f) <= channel_bw / 2
    lower_mask = (f >= -1.5 * channel_bw) & (f < -0.5 * channel_bw)
    upper_mask = (f > 0.5 * channel_bw) & (f <= 1.5 * channel_bw)

    # Compute power in each channel
    main_power = np.sum(psd[main_mask])
    lower_power = np.sum(psd[lower_mask])
    upper_power = np.sum(psd[upper_mask])

    aclr_lower = 10 * np.log10(lower_power / main_power) if main_power > 0 else -np.inf
    aclr_upper = 10 * np.log10(upper_power / main_power) if main_power > 0 else -np.inf

    return aclr_lower, aclr_upper


def plot_constellation(
    pa_input,
    pa_output_no_dpd,
    pa_output_with_dpd,
    tx_baseband,
    fd_symbols,
    demod,
    signal_fs,
    pa_sample_rate,
    fft_size,
    cp_length,
    nSymbols,
    save_path="results/constellation_dpd.png",
):
    """
    Plot constellation comparison for
    PA input, PA output without DPD, and PA output with DPD.

    Args:
        pa_input: PA input signal at PA sample rate [B, num_samples] or [num_samples]
        pa_output_no_dpd: PA output without DPD
        pa_output_with_dpd: PA output with DPD
        tx_baseband: Original baseband signal at signal_fs
        fd_symbols: Original frequency-domain symbols [num_subcarriers, num_symbols]
        demod: Demodulator object with demod() method
        signal_fs: Signal sample rate
        pa_sample_rate: PA sample rate
        fft_size: FFT size
        cp_length: Cyclic prefix length
        nSymbols: Number of OFDM symbols
        save_path: Path to save the plot
    """

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

    original_len = (fft_size + cp_length) * nSymbols

    # Synchronize using cross-correlation
    sync_len = min(1000, len(tx_baseband_np) // 2)
    delay_input = np.argmax(
        np.abs(np.correlate(data_input, tx_baseband_np[:sync_len], mode="valid"))
    )
    delay_no_dpd = np.argmax(
        np.abs(np.correlate(data_no_dpd, tx_baseband_np[:sync_len], mode="valid"))
    )
    delay_with_dpd = np.argmax(
        np.abs(np.correlate(data_with_dpd, tx_baseband_np[:sync_len], mode="valid"))
    )

    data_input_sync = data_input[delay_input : delay_input + original_len]
    data_no_dpd_sync = data_no_dpd[delay_no_dpd : delay_no_dpd + original_len]
    data_with_dpd_sync = data_with_dpd[delay_with_dpd : delay_with_dpd + original_len]

    # Demodulate
    symbols_input = demod.demod(data_input_sync)
    symbols_no_dpd = demod.demod(data_no_dpd_sync)
    symbols_with_dpd = demod.demod(data_with_dpd_sync)

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

    # Plot
    plt.figure(figsize=(10, 10))
    plt.plot(
        fd_np.real.flatten(),
        fd_np.imag.flatten(),
        "o",
        ms=3,
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
        label=f"PA Output, with DPD (EVM={evm_with_dpd:.1f}%)",
        alpha=0.5,
    )
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title("Constellation Comparison: Effect of Digital Predistortion")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Constellation plot saved to: {save_path}")
    return evm_input, evm_no_dpd, evm_with_dpd


def main():
    print("=" * 70)
    print("Batched DPD Example with IndirectLearningDPD")
    print("=" * 70)

    # =========================================================================
    # 1. Generate Signal
    # =========================================================================
    print("\n[1] Generating 5G NR signal...")
    data = build_dataset_from_tx(
        config_path=CONFIG_PATH, batch_size=BATCH_SIZE, shuffle_seed=42
    )
    cfg = data["cfg"]
    fft_size = data["fft_size"]
    cp_length = data["cp_len"]
    nSymbols = cfg["rg"]["num_ofdm_symbols"]
    subcarrier_spacing = cfg["rg"]["subcarrier_spacing"]
    signal_fs = fft_size * subcarrier_spacing

    x_time = data["x_time"]
    tx = tf.reshape(x_time, [BATCH_SIZE, -1])

    # Keep baseband copy for constellation sync
    tx_baseband = tf.reshape(x_time, [-1])

    # Extract frequency-domain symbols for constellation comparison
    x_rg = data["x_rg"]  # [B, num_sym, fft_size]
    num_guard_lower = cfg["rg"]["num_guard_carriers"][0]
    num_guard_upper = cfg["rg"]["num_guard_carriers"][1]
    dc_null = cfg["rg"]["dc_null"]

    lower_start, lower_end = num_guard_lower, fft_size // 2
    upper_start = fft_size // 2 + (1 if dc_null else 0)
    upper_end = fft_size - num_guard_upper

    fd_symbols_lower = tf.transpose(x_rg[0, :, lower_start:lower_end])
    fd_symbols_upper = tf.transpose(x_rg[0, :, upper_start:upper_end])
    fd_symbols = tf.concat([fd_symbols_lower, fd_symbols_upper], axis=0)

    # Create demodulator
    sionna_demod = OFDMDemodulator(
        fft_size=fft_size, l_min=0, cyclic_prefix_length=cp_length
    )

    class Demodulator:
        def __init__(self):
            self.fft_size = fft_size
            self.cp_length = cp_length
            self.nSymbols = nSymbols
            self.sampling_rate = signal_fs

        def demod(self, inp):
            """Demodulate - accepts tensor or numpy, returns tensor."""
            if not isinstance(inp, tf.Tensor):
                inp = tf.constant(inp, dtype=tf.complex64)
            inp_tf = tf.reshape(inp, [1, 1, 1, -1])
            rg = sionna_demod(inp_tf)[0, 0, 0, :, :]  # [num_sym, fft_size]
            fd_lower = tf.transpose(rg[:, lower_start:lower_end])
            fd_upper = tf.transpose(rg[:, upper_start:upper_end])
            return tf.concat([fd_lower, fd_upper], axis=0)

    demod = Demodulator()

    print(f"    Input shape:  {tx.shape}")
    print(f"    Signal fs:    {signal_fs/1e6:.2f} MHz")
    print(f"    Target fs:    {PA_SAMPLE_RATE/1e6:.2f} MHz")
    print(f"    FD symbols:   {fd_symbols.shape}")

    # =========================================================================
    # 2. Normalize and Upsample
    # =========================================================================
    print("\n[2] Normalizing and upsampling...")
    tx_normalized, scale_factor = normalize_to_rms(tx, RMS_INPUT)

    interpolator = Interpolator(input_rate=signal_fs, output_rate=PA_SAMPLE_RATE)
    tx_upsampled, actual_rate = interpolator(tx_normalized)

    print(f"    Upsampled shape: {tx_upsampled.shape}")
    print(f"    Upsample factor: {interpolator.factor}x")

    # =========================================================================
    # 3. Create PA and DPD
    # =========================================================================
    print("\n[3] Creating PA and DPD...")
    pa = PowerAmplifier(order=7, memory_depth=4)
    dpd = IndirectLearningDPD(DPD_PARAMS)

    print(f"    PA order: {pa.order}, memory: {pa.memory_depth}")
    print(f"    DPD order: {dpd.order}, memory: {dpd.memory_depth}")
    print(f"    DPD coefficients: {dpd.n_coeffs}")

    # =========================================================================
    # 4. PA Output without DPD
    # =========================================================================
    print("\n[4] Computing PA output without DPD...")
    pa_output_no_dpd = pa(tx_upsampled)

    aclr_l, aclr_u = compute_aclr(pa_output_no_dpd, PA_SAMPLE_RATE)
    print(f"    ACLR (no DPD): Lower={aclr_l:.2f} dB, Upper={aclr_u:.2f} dB")

    # =========================================================================
    # 5. DPD Learning
    # =========================================================================
    print("\n[5] Performing DPD learning...")
    dpd.perform_learning(tx_upsampled, pa, verbose=True)

    # =========================================================================
    # 6. PA Output with DPD
    # =========================================================================
    print("\n[6] Computing PA output with DPD...")
    tx_predistorted = dpd(tx_upsampled)
    pa_output_with_dpd = pa(tx_predistorted)

    aclr_l_dpd, aclr_u_dpd = compute_aclr(pa_output_with_dpd, PA_SAMPLE_RATE)
    print(f"ACLR (with DPD): Lower={aclr_l_dpd:.2f} dB, Upper={aclr_u_dpd:.2f} dB")
    print(
        f"ACLR improvement: {aclr_l - aclr_l_dpd:.2f} dB (lower), "
        f"{aclr_u - aclr_u_dpd:.2f} dB (upper)"
    )

    # =========================================================================
    # 7. Verify Differentiability
    # =========================================================================
    print("\n[7] Verifying differentiability...")

    # Test 1: Gradient w.r.t. input
    print("\n    --- Gradient w.r.t. input ---")
    with tf.GradientTape() as tape:
        tape.watch(tx_upsampled)
        predistorted = dpd(tx_upsampled)
        pa_out = pa(predistorted)
        loss = tf.reduce_mean(tf.abs(pa_out))

    grad_input = tape.gradient(loss, tx_upsampled)
    print(f"    Gradient exists: {grad_input is not None}")
    print(
        f"    Gradient shape: {grad_input.shape if grad_input is not None else 'N/A'}"
    )
    print(
        f"Gradient non-zero: "
        f"{tf.reduce_any(grad_input != 0).numpy() if grad_input is not None else False}"
    )

    # Test 2: Gradient w.r.t. DPD trainable variables (coefficients)
    print("\n    --- Gradient w.r.t. DPD trainable_variables ---")
    print(f"    DPD trainable_variables: {[v.name for v in dpd.trainable_variables]}")

    with tf.GradientTape() as tape:
        predistorted = dpd(tx_upsampled)
        pa_out = pa(predistorted)
        loss = tf.reduce_mean(tf.abs(pa_out))

    grad_coeffs = tape.gradient(loss, dpd.trainable_variables)
    for i, (g, v) in enumerate(zip(grad_coeffs, dpd.trainable_variables)):
        if g is not None:
            print(f"    {v.name}:")
            print("        Gradient exists: True")
            print(f"        Gradient shape: {g.shape}")
            print(f"        Gradient non-zero: {tf.reduce_any(g != 0).numpy()}")
            print(f"        Gradient max abs: {tf.reduce_max(tf.abs(g)).numpy():.6f}")
        else:
            print(f"    {v.name}:")
            print("        Gradient exists: False")

    # Test 3: Verify optimizer can update weights
    print("\n    --- Optimizer test ---")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    initial_real = dpd._coeffs_real.numpy().copy()
    initial_imag = dpd._coeffs_imag.numpy().copy()

    with tf.GradientTape() as tape:
        predistorted = dpd(tx_upsampled)
        pa_out = pa(predistorted)
        loss = tf.reduce_mean(tf.abs(pa_out))

    grads = tape.gradient(loss, dpd.trainable_variables)
    optimizer.apply_gradients(zip(grads, dpd.trainable_variables))

    real_changed = not np.allclose(initial_real, dpd._coeffs_real.numpy())
    imag_changed = not np.allclose(initial_imag, dpd._coeffs_imag.numpy())
    print(f"    Weights updated by optimizer: {real_changed or imag_changed}")

    # Restore original weights (we don't want to mess up the learned DPD)
    dpd._coeffs_real.assign(initial_real)
    dpd._coeffs_imag.assign(initial_imag)

    # =========================================================================
    # 8. Generate PSD Plot
    # =========================================================================
    print("\n[8] Generating PSD comparison plot...")
    plot_psd_comparison_three(
        tx_upsampled,
        pa_output_no_dpd,
        pa_output_with_dpd,
        PA_SAMPLE_RATE,
        save_path="results/psd_dpd_comparison.png",
    )

    # =========================================================================
    # 9. Generate Constellation Plot
    # =========================================================================
    print("\n[9] Generating constellation comparison plot...")
    evm_input, evm_no_dpd, evm_with_dpd = plot_constellation(
        tx_upsampled,
        pa_output_no_dpd,
        pa_output_with_dpd,
        tx_baseband,
        fd_symbols,
        demod,
        signal_fs,
        PA_SAMPLE_RATE,
        fft_size,
        cp_length,
        nSymbols,
        save_path="results/constellation_dpd.png",
    )

    # =========================================================================
    # 10. Plot Coefficient History
    # =========================================================================
    print("\n[10] Plotting DPD coefficient history...")
    dpd.plot_coeff_history(save_path="results/dpd_coeff_history.png")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"ACLR without DPD: Lower={aclr_l:.2f} dB, Upper={aclr_u:.2f} dB")
    print(f"ACLR with DPD:    Lower={aclr_l_dpd:.2f} dB, Upper={aclr_u_dpd:.2f} dB")
    print(
        f"ACLR improvement: "
        f"{(aclr_l - aclr_l_dpd + aclr_u - aclr_u_dpd) / 2:.2f} dB average"
    )
    print(f"\nEVM (PA input):       {evm_input:.2f}%")
    print(f"EVM (no DPD):         {evm_no_dpd:.2f}%")
    print(f"EVM (with DPD):       {evm_with_dpd:.2f}%")
    print(f"EVM improvement:      {evm_no_dpd - evm_with_dpd:.2f}%")
    print("\nResults saved to results/ folder")


if __name__ == "__main__":
    main()
