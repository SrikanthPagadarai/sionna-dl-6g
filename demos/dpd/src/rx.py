# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
OFDM receiver for DPD evaluation.

Implements a minimal OFDM receiver chain for measuring DPD
performance via Error Vector Magnitude (EVM). The receiver handles the
full path from PA output back to constellation symbols, enabling direct
comparison of signal quality with and without predistortion.

The receiver operates only during inference (not training):
1. DPD training uses indirect learning on PA input/output, not decoded symbols
2. EVM is an evaluation metric, not a training objective

The receive chain uses NumPy/SciPy for resampling (scipy.signal.resample_poly)
because this runs only at inference time where graph-mode compatibility
is not required.
"""

import numpy as np
import tensorflow as tf
from fractions import Fraction
from scipy.signal import resample_poly
from sionna.phy.ofdm import OFDMDemodulator


class Rx(tf.keras.layers.Layer):
    """
    Minimal OFDM receiver chain for DPD performance evaluation.

    Implements the full receive path from PA output to equalized QAM
    symbols, with EVM computation for quantifying signal distortion.
    This receiver is designed for DPD evaluation, not for conducting
    end-to-end communication link performance evaluation.

    Parameters
    ----------
    signal_fs : float
        Baseband signal sample rate in Hz (e.g., 15.36 MHz for 15 kHz
        subcarrier spacing with 1024-point FFT).
    pa_sample_rate : float
        PA operating sample rate in Hz (typically 8x signal rate for
        adequate reconstruction of PA nonlinear products).
    fft_size : int
        OFDM FFT size (number of subcarriers including guards).
    cp_length : int
        Cyclic prefix length in samples.
    num_ofdm_symbols : int
        Number of OFDM symbols per slot.
    num_guard_lower : int
        Number of guard subcarriers at lower band edge.
    num_guard_upper : int
        Number of guard subcarriers at upper band edge.
    dc_null : bool
        Whether the DC subcarrier is nulled.
    **kwargs
        Additional keyword arguments passed to Keras Layer.

    Attributes
    ----------
    _ofdm_demod : OFDMDemodulator
        Sionna OFDM demodulator (FFT + CP removal).
    _lower_start, _lower_end : int
        Subcarrier indices for lower data band.
    _upper_start, _upper_end : int
        Subcarrier indices for upper data band.

    Notes
    -----
    **Receiver Processing Steps:**

    1. **Downsample**: Convert from PA rate to baseband rate
    2. **Time sync**: Cross-correlation to find frame boundary
    3. **OFDM demod**: Remove CP and apply FFT
    4. **Equalize**: Zero-forcing per-subcarrier equalization
    5. **EVM**: Compute error vector magnitude vs. reference

    **Why Zero-Forcing Equalization?**

    In this loopback test scenario, the channel is essentially flat
    (no multipath). ZF equalization corrects only for the PA's linear
    gain and phase offset. More sophisticated equalizers (MMSE, etc.)
    are unnecessary and would complicate DPD performance attribution.

    **miscellaneous:**

    - PA output signals must be at ``pa_sample_rate``
    - Reference baseband signal must be at ``signal_fs``
    - All signals must represent the same transmitted frame
    - Equalized symbols are normalized to reference constellation
    - EVM is returned as percentage (0-100+ scale)

    Example
    -------
    >>> rx = Rx(
    ...     signal_fs=15.36e6,
    ...     pa_sample_rate=122.88e6,
    ...     fft_size=1024,
    ...     cp_length=72,
    ...     num_ofdm_symbols=14,
    ...     num_guard_lower=200,
    ...     num_guard_upper=199,
    ...     dc_null=True,
    ... )
    >>> results = rx.process_and_compute_evm(
    ...     pa_input, pa_output_no_dpd, pa_output_with_dpd,
    ...     tx_baseband, fd_symbols
    ... )
    >>> print(f"EVM with DPD: {results['evm_with_dpd']:.2f}%")
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

        # Sionna's OFDM demodulator handles CP removal and FFT.
        # l_min=0 means no negative delay taps (single-path channel).
        self._ofdm_demod = OFDMDemodulator(
            fft_size=fft_size,
            l_min=0,
            cyclic_prefix_length=cp_length,
        )

        # Precompute data subcarrier index ranges.
        # OFDM spectrum layout: [guard_lower | data_lower | DC | data_upper | guard_upper]
        # After FFT, indices 0..fft_size/2-1 are lower half, fft_size/2..fft_size-1 are upper.
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
        Process PA outputs and compute EVM for DPD performance comparison.

        Runs three signal paths (PA input, PA output without DPD, PA output
        with DPD) through the complete receiver chain and computes EVM for
        each. This enables direct comparison of DPD effectiveness.

        Parameters
        ----------
        pa_input : tf.Tensor or np.ndarray
            PA input signal at PA sample rate (reference for best-case EVM).
        pa_output_no_dpd : tf.Tensor or np.ndarray
            PA output without predistortion at PA sample rate.
        pa_output_with_dpd : tf.Tensor or np.ndarray
            PA output with predistortion at PA sample rate.
        tx_baseband : tf.Tensor or np.ndarray
            Original baseband transmit signal at signal sample rate
            (used as timing reference for synchronization).
        fd_symbols : tf.Tensor or np.ndarray
            Transmitted frequency-domain symbols, shape
            ``[num_data_subcarriers, num_ofdm_symbols]``.
            Used as reference for equalization and EVM calculation.

        Returns
        -------
        dict
            Results dictionary containing:

            - ``symbols_input`` : np.ndarray
                Equalized constellation symbols from PA input path.
            - ``symbols_no_dpd`` : np.ndarray
                Equalized symbols from PA output without DPD.
            - ``symbols_with_dpd`` : np.ndarray
                Equalized symbols from PA output with DPD.
            - ``evm_input`` : float
                EVM (%) for PA input (baseline, should be near-zero).
            - ``evm_no_dpd`` : float
                EVM (%) for PA output without DPD (shows PA distortion).
            - ``evm_with_dpd`` : float
                EVM (%) for PA output with DPD (shows DPD improvement).

        Notes
        -----
        The PA input path serves as a sanity check: its EVM should be
        very low since it hasn't passed through PA nonlinearity.
        """

        # Flatten all signals to 1D for processing.
        # Input tensors may have batch/stream dimensions that aren't needed here.
        def flatten(x):
            if len(x.shape) > 1:
                return tf.reshape(x, [-1]).numpy()
            return x.numpy() if hasattr(x, "numpy") else x

        pa_input_flat = flatten(pa_input)
        pa_no_dpd_flat = flatten(pa_output_no_dpd)
        pa_with_dpd_flat = flatten(pa_output_with_dpd)
        tx_baseband_np = flatten(tx_baseband)

        # Step 1: Downsample from PA rate to baseband rate.
        # Using rational resampling to handle non-integer rate ratios.
        frac = Fraction(self._signal_fs / self._pa_sample_rate).limit_denominator(1000)
        data_input = resample_poly(pa_input_flat, frac.numerator, frac.denominator)
        data_no_dpd = resample_poly(pa_no_dpd_flat, frac.numerator, frac.denominator)
        data_with_dpd = resample_poly(
            pa_with_dpd_flat, frac.numerator, frac.denominator
        )

        # Step 2: Time synchronization via cross-correlation.
        # Find the delay that maximizes correlation with known transmit signal.
        original_len = (self._fft_size + self._cp_length) * self._num_ofdm_symbols
        # Use only first portion of reference to reduce computation.
        sync_len = min(1000, len(tx_baseband_np) // 2)

        def find_delay(signal, ref):
            """Find sample delay by peak cross-correlation."""
            return np.argmax(np.abs(np.correlate(signal, ref[:sync_len], mode="valid")))

        delay_input = find_delay(data_input, tx_baseband_np)
        delay_no_dpd = find_delay(data_no_dpd, tx_baseband_np)
        delay_with_dpd = find_delay(data_with_dpd, tx_baseband_np)

        # Extract synchronized frame (exact length needed for OFDM demod).
        data_input_sync = data_input[delay_input : delay_input + original_len]
        data_no_dpd_sync = data_no_dpd[delay_no_dpd : delay_no_dpd + original_len]
        data_with_dpd_sync = data_with_dpd[
            delay_with_dpd : delay_with_dpd + original_len
        ]

        # Step 3: OFDM demodulation (CP removal + FFT).
        symbols_input = self._demod(data_input_sync)
        symbols_no_dpd = self._demod(data_no_dpd_sync)
        symbols_with_dpd = self._demod(data_with_dpd_sync)

        # Step 4: Per-subcarrier zero-forcing equalization.
        symbols_input = self._equalize(symbols_input, fd_symbols)
        symbols_no_dpd = self._equalize(symbols_no_dpd, fd_symbols)
        symbols_with_dpd = self._equalize(symbols_with_dpd, fd_symbols)

        # Convert to numpy for EVM calculation.
        fd_np = fd_symbols.numpy() if isinstance(fd_symbols, tf.Tensor) else fd_symbols
        sym_input_np = symbols_input.numpy()
        sym_no_dpd_np = symbols_no_dpd.numpy()
        sym_with_dpd_np = symbols_with_dpd.numpy()

        # Step 5: Compute EVM as percentage.
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
        Demodulate time-domain OFDM signal to frequency-domain symbols.

        Parameters
        ----------
        signal : np.ndarray
            Time-domain signal, shape ``[num_samples]``, at baseband rate.
            Length must equal ``(fft_size + cp_length) * num_ofdm_symbols``.

        Returns
        -------
        tf.Tensor
            Data subcarrier symbols, shape ``[num_data_subcarriers, num_symbols]``.
            Guard bands and DC null are excluded.

        Notes
        -----
        The output concatenates lower and upper data bands. This matches
        the order used by the transmitter's resource grid mapper.
        """
        if not isinstance(signal, tf.Tensor):
            signal = tf.constant(signal, dtype=tf.complex64)

        # Reshape for Sionna demodulator: [batch, rx_ant, tx_ant, samples].
        signal_4d = tf.reshape(signal, [1, 1, 1, -1])

        # Demodulate: output is [batch, rx_ant, tx_ant, num_symbols, fft_size].
        rg = self._ofdm_demod(signal_4d)[0, 0, 0, :, :]  # [num_symbols, fft_size]

        # Extract data subcarriers (exclude guards and DC).
        # Transpose to get [subcarriers, symbols] ordering.
        fd_lower = tf.transpose(rg[:, self._lower_start : self._lower_end])
        fd_upper = tf.transpose(rg[:, self._upper_start : self._upper_end])

        return tf.concat([fd_lower, fd_upper], axis=0)

    def _equalize(self, rx, tx):
        """
        Apply zero-forcing per-subcarrier equalization.

        Estimates a single complex gain per subcarrier by least-squares
        fit across all OFDM symbols, then divides received symbols by
        this estimate.

        Parameters
        ----------
        rx : tf.Tensor
            Received symbols, shape ``[num_subcarriers, num_symbols]``.
        tx : tf.Tensor
            Transmitted reference symbols, same shape as ``rx``.

        Returns
        -------
        tf.Tensor
            Equalized symbols, same shape as input.

        Notes
        -----
        The channel estimate is computed as:

        .. math::

            \\hat{H}_k = \\frac{\\sum_n r_{k,n} \\cdot t_{k,n}^*}{\\sum_n |t_{k,n}|^2}

        where :math:`k` is subcarrier index and :math:`n` is symbol index.
        This is the least-squares estimate assuming the channel is constant
        across all symbols (valid for this static loopback scenario).
        """
        rx = tf.cast(rx, tf.complex64)
        tx = tf.cast(tx, tf.complex64)

        # Least-squares channel estimate per subcarrier.
        # H = sum(rx * conj(tx)) / sum(|tx|^2)
        H = tf.reduce_sum(rx * tf.math.conj(tx), axis=1, keepdims=True) / tf.cast(
            tf.reduce_sum(tf.abs(tx) ** 2, axis=1, keepdims=True), tf.complex64
        )

        # Zero-forcing equalization: divide by channel estimate.
        return rx / H

    @staticmethod
    def _compute_evm(rx, tx):
        """
        Compute Error Vector Magnitude (EVM) as percentage.

        EVM quantifies the difference between received and ideal
        constellation points, normalized by the reference signal power.

        Parameters
        ----------
        rx : np.ndarray
            Received (equalized) symbols.
        tx : np.ndarray
            Transmitted reference symbols (same shape as ``rx``).

        Returns
        -------
        float
            EVM as percentage (e.g., 5.0 means 5% EVM).

        Notes
        -----
        EVM is computed as:

        .. math::

            \\text{EVM} = 100 \\times \\sqrt{\\frac{\\text{mean}(|r - t|^2)}{\\text{mean}(|t|^2)}}

        Lower EVM indicates better signal quality. Typical targets:

        - 64-QAM: < 8% EVM required
        - 256-QAM: < 3.5% EVM required
        """
        error = rx - tx
        evm = np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(tx) ** 2)) * 100
        return float(evm)
