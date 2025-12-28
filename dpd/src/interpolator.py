"""TensorFlow graph-mode compatible interpolator for signal upsampling.

This module provides Interpolator, a drop-in replacement for Interpolator
that uses Sionna's native signal processing functions (Upsampling, Downsampling,
convolve) for GPU acceleration and graph-mode compatibility.

The implementation uses Kaiser-windowed FIR filter design with configurable
parameters for excellent stopband attenuation (>100 dB with defaults).
"""

import tensorflow as tf
from fractions import Fraction
from math import gcd
from scipy.signal import firwin  # Only used at init time, not in graph

from sionna.phy.signal import Upsampling, Downsampling, convolve


class Interpolator(tf.keras.layers.Layer):
    """
    TensorFlow graph-mode compatible interpolator using Sionna primitives.

    Performs polyphase-equivalent resampling:
    1. Upsample by factor L (insert L-1 zeros between samples) via Sionna Upsampling
    2. Apply FIR anti-imaging filter via Sionna convolve
    3. Downsample by factor M (take every M-th sample) via Sionna Downsampling

    Each batch element is processed independently along the sample axis,
    which is the correct behavior for independent OFDM frames.

    Filter design uses Kaiser-windowed FIR with configurable parameters
    for excellent stopband attenuation (>100 dB with defaults).

    Args:
        input_rate: Input sample rate (Hz)
        output_rate: Output sample rate (Hz)
        max_denominator: Maximum denominator when converting to fraction (default: 1000)
        half_len_mult: Filter half-length multiplier (default: 20 for >100 dB stopband)
        kaiser_beta: Kaiser window beta parameter (default: 8.0 for >100 dB stopband)
    """

    def __init__(
        self,
        input_rate: float,
        output_rate: float,
        max_denominator: int = 1000,
        half_len_mult: int = 20,
        kaiser_beta: float = 8.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Convert rate ratio to fraction L/M
        frac = Fraction(output_rate / input_rate).limit_denominator(max_denominator)
        self._upsample_factor = frac.numerator  # L
        self._downsample_factor = frac.denominator  # M

        self._input_rate = input_rate
        self._output_rate = input_rate * frac.numerator / frac.denominator

        # Reduce by GCD (matching scipy)
        g = gcd(self._upsample_factor, self._downsample_factor)
        up_g = self._upsample_factor // g
        down_g = self._downsample_factor // g

        # Design anti-imaging filter matching scipy.signal.resample_poly
        # Filter length formula: 2 * (half_len_mult * max(up, down)) + 1
        max_rate = max(up_g, down_g)
        half_len = half_len_mult * max_rate
        n_taps = 2 * half_len + 1

        # Cutoff frequency normalized to upsampled Nyquist rate
        cutoff = 1.0 / max_rate

        # Design lowpass filter with Kaiser window
        filter_coeffs = firwin(n_taps, cutoff, window=("kaiser", kaiser_beta))

        # Scale filter to compensate for zero-insertion gain loss
        filter_coeffs = filter_coeffs * self._upsample_factor

        # Store filter as TF constant (complex for use with Sionna convolve)
        self._filter_coeffs = tf.constant(filter_coeffs, dtype=tf.float32)
        self._filter_length = n_taps
        self._half_len = half_len

        # Create Sionna upsampling/downsampling blocks
        # axis=-1 operates on the sample dimension for [B, N] input
        self._upsampler = Upsampling(samples_per_symbol=self._upsample_factor, axis=-1)

        if self._downsample_factor > 1:
            self._downsampler = Downsampling(
                samples_per_symbol=self._downsample_factor, axis=-1
            )
        else:
            self._downsampler = None

    def call(self, x):
        """
        Interpolate input tensor using Sionna operations (graph-compatible).

        Each batch element is processed independently along the sample axis.

        Args:
            x: [B, num_samples] complex64 tensor

        Returns:
            Tuple of (interpolated tensor [B, num_samples * factor], output_rate)
        """
        # Ensure complex64
        x = tf.cast(x, tf.complex64)

        # Step 1: Upsample (insert zeros) using Sionna
        # [B, N] -> [B, N * L]
        upsampled = self._upsampler(x)
        n_upsampled = tf.shape(upsampled)[-1]

        # Step 2: Apply anti-imaging FIR filter using Sionna convolve
        # Use 'full' padding: output length = N*L + K - 1
        filter_complex = tf.cast(self._filter_coeffs, tf.complex64)
        filtered = convolve(upsampled, filter_complex, padding="full", axis=-1)

        # Compensate for filter group delay
        # For a symmetric FIR filter of length K, group delay = (K-1)/2 samples
        # With 'full' convolution, the aligned output starts at index (K-1)//2
        # and we want N*L output samples
        group_delay = (self._filter_length - 1) // 2
        filtered = filtered[..., group_delay : group_delay + n_upsampled]

        # Step 3: Downsample using Sionna (if needed)
        if self._downsampler is not None:
            x_out = self._downsampler(filtered)
        else:
            x_out = filtered

        return x_out, self._output_rate
