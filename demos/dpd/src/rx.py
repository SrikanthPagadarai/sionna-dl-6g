"""
Simple OFDM Rx with synchronization, demodulation, equalization, and EVM computation.
"""

import numpy as np
import tensorflow as tf
from fractions import Fraction
from scipy.signal import resample_poly
from sionna.phy.ofdm import OFDMDemodulator


class Rx(tf.keras.layers.Layer):
    """
    Complete OFDM receiver chain for DPD evaluation.

    Performs:
    1. Downsampling from PA rate to signal rate
    2. Time synchronization via cross-correlation
    3. OFDM demodulation
    4. Per-subcarrier equalization
    5. EVM computation

    This is only used during inference (training=False) for performance evaluation.

    Args:
        signal_fs: Signal sample rate in Hz
        pa_sample_rate: PA sample rate in Hz
        fft_size: FFT size for OFDM
        cp_length: Cyclic prefix length
        num_ofdm_symbols: Number of OFDM symbols
        num_guard_lower: Number of lower guard carriers
        num_guard_upper: Number of upper guard carriers
        dc_null: Whether DC subcarrier is nulled
    """

    def __init__(
        self,
        signal_fs: float,
        pa_sample_rate: float,
        fft_size: int,
        cp_length: int,
        num_ofdm_symbols: int,
        num_guard_lower: int,
        num_guard_upper: int,
        dc_null: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._signal_fs = signal_fs
        self._pa_sample_rate = pa_sample_rate
        self._fft_size = fft_size
        self._cp_length = cp_length
        self._num_ofdm_symbols = num_ofdm_symbols
        self._num_guard_lower = num_guard_lower
        self._num_guard_upper = num_guard_upper
        self._dc_null = dc_null

        # OFDM demodulator
        self._ofdm_demod = OFDMDemodulator(
            fft_size=fft_size,
            l_min=0,
            cyclic_prefix_length=cp_length,
        )

        # Subcarrier indices for data extraction
        self._lower_start = num_guard_lower
        self._lower_end = fft_size // 2
        self._upper_start = fft_size // 2 + (1 if dc_null else 0)
        self._upper_end = fft_size - num_guard_upper

    def process_and_compute_evm(
        self,
        pa_input,
        pa_output_no_dpd,
        pa_output_with_dpd,
        tx_baseband,
        fd_symbols,
    ):
        """
        Process PA outputs through complete receiver chain and compute EVM.

        Args:
            pa_input: PA input signal at PA rate
            pa_output_no_dpd: PA output without DPD at PA rate
            pa_output_with_dpd: PA output with DPD at PA rate
            tx_baseband: Baseband transmit signal (for sync reference)
            fd_symbols: Frequency domain symbols (for equalization reference)

        Returns:
            dict with:
                - symbols_input: Equalized PA input symbols
                - symbols_no_dpd: Equalized PA output symbols (no DPD)
                - symbols_with_dpd: Equalized PA output symbols (with DPD)
                - evm_input: EVM for PA input
                - evm_no_dpd: EVM for PA output (no DPD)
                - evm_with_dpd: EVM for PA output (with DPD)
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

        # Step 1: Downsample from PA rate to signal rate
        frac = Fraction(self._signal_fs / self._pa_sample_rate).limit_denominator(1000)
        data_input = resample_poly(pa_input_flat, frac.numerator, frac.denominator)
        data_no_dpd = resample_poly(pa_no_dpd_flat, frac.numerator, frac.denominator)
        data_with_dpd = resample_poly(
            pa_with_dpd_flat, frac.numerator, frac.denominator
        )

        # Step 2: Time synchronization via cross-correlation
        original_len = (self._fft_size + self._cp_length) * self._num_ofdm_symbols
        sync_len = min(1000, len(tx_baseband_np) // 2)

        def find_delay(signal, ref):
            return np.argmax(np.abs(np.correlate(signal, ref[:sync_len], mode="valid")))

        delay_input = find_delay(data_input, tx_baseband_np)
        delay_no_dpd = find_delay(data_no_dpd, tx_baseband_np)
        delay_with_dpd = find_delay(data_with_dpd, tx_baseband_np)

        data_input_sync = data_input[delay_input : delay_input + original_len]
        data_no_dpd_sync = data_no_dpd[delay_no_dpd : delay_no_dpd + original_len]
        data_with_dpd_sync = data_with_dpd[
            delay_with_dpd : delay_with_dpd + original_len
        ]

        # Step 3: OFDM demodulation
        symbols_input = self._demod(data_input_sync)
        symbols_no_dpd = self._demod(data_no_dpd_sync)
        symbols_with_dpd = self._demod(data_with_dpd_sync)

        # Step 4: Per-subcarrier equalization
        symbols_input = self._equalize(symbols_input, fd_symbols)
        symbols_no_dpd = self._equalize(symbols_no_dpd, fd_symbols)
        symbols_with_dpd = self._equalize(symbols_with_dpd, fd_symbols)

        # Convert to numpy
        fd_np = fd_symbols.numpy() if isinstance(fd_symbols, tf.Tensor) else fd_symbols
        sym_input_np = symbols_input.numpy()
        sym_no_dpd_np = symbols_no_dpd.numpy()
        sym_with_dpd_np = symbols_with_dpd.numpy()

        # Step 5: Compute EVM
        evm_input = self._compute_evm(sym_input_np, fd_np)
        evm_no_dpd = self._compute_evm(sym_no_dpd_np, fd_np)
        evm_with_dpd = self._compute_evm(sym_with_dpd_np, fd_np)

        return {
            "symbols_input": sym_input_np,
            "symbols_no_dpd": sym_no_dpd_np,
            "symbols_with_dpd": sym_with_dpd_np,
            "evm_input": evm_input,
            "evm_no_dpd": evm_no_dpd,
            "evm_with_dpd": evm_with_dpd,
        }

    def _demod(self, signal):
        """
        Demodulate OFDM signal to extract frequency-domain symbols.

        Args:
            signal: [num_samples] complex numpy array at baseband sample rate

        Returns:
            [num_subcarriers, num_symbols] complex tensor
        """
        if not isinstance(signal, tf.Tensor):
            signal = tf.constant(signal, dtype=tf.complex64)

        # Reshape for Sionna demodulator: [batch, rx, tx, samples]
        signal_4d = tf.reshape(signal, [1, 1, 1, -1])

        # Demodulate
        rg = self._ofdm_demod(signal_4d)[0, 0, 0, :, :]  # [num_symbols, fft_size]

        # Extract data subcarriers
        fd_lower = tf.transpose(rg[:, self._lower_start : self._lower_end])
        fd_upper = tf.transpose(rg[:, self._upper_start : self._upper_end])

        return tf.concat([fd_lower, fd_upper], axis=0)

    def _equalize(self, rx, tx):
        """
        Per-subcarrier zero-forcing equalization.

        Args:
            rx: Received symbols [num_subcarriers, num_symbols]
            tx: Transmitted symbols [num_subcarriers, num_symbols]

        Returns:
            Equalized symbols [num_subcarriers, num_symbols]
        """
        rx = tf.cast(rx, tf.complex64)
        tx = tf.cast(tx, tf.complex64)

        # Estimate channel per subcarrier
        H = tf.reduce_sum(rx * tf.math.conj(tx), axis=1, keepdims=True) / tf.cast(
            tf.reduce_sum(tf.abs(tx) ** 2, axis=1, keepdims=True), tf.complex64
        )

        # Zero-forcing equalization
        return rx / H

    @staticmethod
    def _compute_evm(rx, tx):
        """
        Compute Error Vector Magnitude (EVM) in percentage.

        Args:
            rx: Received symbols (numpy array)
            tx: Transmitted symbols (numpy array)

        Returns:
            EVM in percentage
        """
        error = rx - tx
        evm = np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(tx) ** 2)) * 100
        return float(evm)
