# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
TensorFlow graph-mode compatible interpolator for signal upsampling.

Rational resampler layer that converts signals between the baseband sample
rate (e.g., 15.36 MHz in this demo) and the desired sample rate (e.g., the
PA operating rate of 122.88 MHz in this demo). The implementation uses
Sionna's native signal processing primitives for GPU acceleration and
``tf.function`` compatibility.

The core algorithm is polyphase-equivalent resampling via upsample-filter-downsample,
with a Kaiser-windowed FIR anti-imaging filter designed for excellent stopband
attenuation (>100 dB with default parameters).

Notes
-----
**Why not scipy.signal.resample_poly directly?**

While scipy's resample_poly is efficient, it operates on NumPy arrays and
breaks TensorFlow's computation graph. This implementation achieves the same
mathematical result using Sionna's ``Upsampling``, ``Downsampling``, and
``convolve`` operations, enabling end-to-end gradient flow for DPD training.
"""

import tensorflow as tf
from fractions import Fraction
from math import gcd
from scipy.signal import firwin  # Only used at init time, not in graph

from sionna.phy.signal import Upsampling, Downsampling, convolve


class Interpolator(tf.keras.layers.Layer):
    """
    Rational sample rate converter using polyphase-equivalent resampling.

    Converts between sample rates by a rational factor L/M, where L is the
    upsampling factor and M is the downsampling factor. The rate ratio is
    automatically converted to a reduced fraction for efficiency.

    Parameters
    ----------
    input_rate : float
        Input sample rate in Hz (e.g., 15.36e6 for OFDM baseband).
    output_rate : float
        Desired output sample rate in Hz (e.g., 122.88e6 for PA).
    max_denominator : int
        Maximum denominator when approximating the rate ratio as a fraction.
        Higher values give more accurate rate conversion at the cost of
        longer filters. Defaults to 1000.
    half_len_mult : int
        Filter half-length multiplier. The filter length is
        ``2 * half_len_mult * max(L, M) + 1``. Defaults to 20, which
        provides >100 dB stopband attenuation.
    kaiser_beta : float
        Kaiser window shape parameter. Higher values give better stopband
        attenuation but wider transition band. Defaults to 8.0 for >100 dB.

    Attributes
    ----------
    _upsample_factor : int
        Upsampling factor L (numerator of rate ratio).
    _downsample_factor : int
        Downsampling factor M (denominator of rate ratio).
    _output_rate : float
        Actual output rate (may differ slightly from requested due to
        rational approximation).
    _filter_coeffs : tf.Tensor
        FIR filter coefficients as a TensorFlow constant.

    Notes
    -----
    **Algorithm (Polyphase Resampling):**

    1. **Upsample by L**: Insert L-1 zeros between each input sample.
       This creates spectral images at multiples of the original Nyquist rate.

    2. **Anti-imaging filter**: Apply a lowpass FIR filter with cutoff at
       ``min(input_rate, output_rate) / 2`` to remove spectral images.

    3. **Downsample by M**: Keep every M-th sample.

    **miscellaneous:**

    - Input must be complex64 with shape ``[batch_size, num_samples]``
    - Input and output rates must be positive
    - Output length is approximately ``num_samples * L / M``
    - Output sample rate equals ``input_rate * L / M``

    **Filter Design Rationale:**

    The Kaiser window is chosen because it provides a near-optimal trade-off
    between main lobe width and side lobe level, controlled by a single
    parameter (beta). With beta=8.0 and half_len_mult=20, the stopband
    attenuation exceeds 100 dB, ensuring spectral images are sufficiently
    suppressed to meet typical PA spurious emission requirements.

    Example
    -------
    >>> # Upsample from 15.36 MHz to 122.88 MHz (8x)
    >>> interp = Interpolator(input_rate=15.36e6, output_rate=122.88e6)
    >>> x = tf.complex(tf.random.normal([16, 1024]), tf.random.normal([16, 1024]))
    >>> y, out_rate = interp(x)
    >>> y.shape  # [16, 8192]
    >>> out_rate  # 122880000.0
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

        # Convert rate ratio to fraction L/M using continued fraction approximation.
        # This finds the best rational approximation within the denominator limit.
        frac = Fraction(output_rate / input_rate).limit_denominator(max_denominator)
        self._upsample_factor = frac.numerator  # L
        self._downsample_factor = frac.denominator  # M

        self._input_rate = input_rate
        # Actual output rate may differ slightly from requested due to rational approx
        self._output_rate = input_rate * frac.numerator / frac.denominator

        # Reduce L and M by their GCD for computational efficiency.
        # This matches scipy.signal.resample_poly behavior.
        g = gcd(self._upsample_factor, self._downsample_factor)
        up_g = self._upsample_factor // g
        down_g = self._downsample_factor // g

        # --- Anti-imaging FIR filter design ---
        # Filter length scales with max(L, M) to maintain transition bandwidth
        # relative to the lower of the two sample rates.
        max_rate = max(up_g, down_g)
        half_len = half_len_mult * max_rate
        n_taps = 2 * half_len + 1

        # Cutoff at the lower Nyquist rate (normalized to upsampled rate)
        # This ensures no aliasing when downsampling.
        cutoff = 1.0 / max_rate

        # Design symmetric lowpass filter with Kaiser window.
        # firwin returns linear-phase (symmetric) coefficients.
        filter_coeffs = firwin(n_taps, cutoff, window=("kaiser", kaiser_beta))

        # Scale filter by L to compensate for energy loss from zero-insertion.
        # Without this, upsampling would reduce signal amplitude by factor L.
        filter_coeffs = filter_coeffs * self._upsample_factor

        # Store as TF constant for graph-mode compatibility
        self._filter_coeffs = tf.constant(filter_coeffs, dtype=tf.float32)
        self._filter_length = n_taps
        self._half_len = half_len

        # --- Create Sionna resampling blocks ---
        # axis=-1 processes along the sample dimension for [B, N] input
        self._upsampler = Upsampling(samples_per_symbol=self._upsample_factor, axis=-1)

        # Downsampler only needed if M > 1
        if self._downsample_factor > 1:
            self._downsampler = Downsampling(
                samples_per_symbol=self._downsample_factor, axis=-1
            )
        else:
            self._downsampler = None

    def call(self, x):
        """
        Resample input signal to the target sample rate.

        Parameters
        ----------
        x : tf.Tensor
            Input signal with shape ``[batch_size, num_samples]`` and dtype
            complex64. Each batch element is resampled independently.

        Returns
        -------
        x_resampled : tf.Tensor
            Resampled signal with shape ``[batch_size, num_samples * L / M]``
            and dtype complex64.
        output_rate : float
            Actual output sample rate in Hz. This may differ slightly from
            the requested rate due to rational approximation.

        Notes
        -----
        **Implementation Details:**

        The 'full' convolution padding is used to ensure all input samples
        contribute to the output. The group delay compensation then extracts
        the correctly aligned portion of the filtered signal.

        For a symmetric FIR filter of length K, the group delay is (K-1)/2
        samples. The output is trimmed to maintain the expected length
        relationship between input and output.
        """
        # Ensure complex64 for consistent dtype throughout the pipeline
        x = tf.cast(x, tf.complex64)

        # --- Step 1: Upsample by L (zero insertion) ---
        # Inserts L-1 zeros between each sample, creating spectral images
        # [B, N] -> [B, N * L]
        upsampled = self._upsampler(x)
        n_upsampled = tf.shape(upsampled)[-1]

        # --- Step 2: Anti-imaging lowpass filter ---
        # Removes spectral images created by upsampling.
        # 'full' padding: output length = N*L + K - 1
        filter_complex = tf.cast(self._filter_coeffs, tf.complex64)
        filtered = convolve(upsampled, filter_complex, padding="full", axis=-1)

        # Compensate for filter group delay to align output with input.
        # For symmetric FIR of length K: group_delay = (K-1)/2 samples.
        # Extract the centered portion to get N*L output samples.
        group_delay = (self._filter_length - 1) // 2
        filtered = filtered[..., group_delay : group_delay + n_upsampled]

        # --- Step 3: Downsample by M (decimation) ---
        # Keep every M-th sample. Only needed if M > 1.
        if self._downsampler is not None:
            x_out = self._downsampler(filtered)
        else:
            x_out = filtered

        return x_out, self._output_rate
