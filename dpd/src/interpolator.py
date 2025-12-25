"""Interpolator for batched signal upsampling using scipy."""

import numpy as np
import tensorflow as tf
from fractions import Fraction
from scipy.signal import resample_poly, firls, lfilter


class Interpolator(tf.keras.layers.Layer):
    """
    Interpolator using scipy's resample_poly + firls + lfilter.

    Matches the exact behavior of signal.py's upsample() method.
    Supports batched input [B, N] by flattening, processing, and reshaping.

    Args:
        input_rate: Input sample rate (Hz)
        output_rate: Output sample rate (Hz)
        max_denominator: Maximum denominator when converting to fraction (default: 1000)
    """

    def __init__(
        self,
        input_rate: float,
        output_rate: float,
        max_denominator: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Convert rate ratio to fraction L/M
        frac = Fraction(output_rate / input_rate).limit_denominator(max_denominator)
        self._upsample_factor = frac.numerator  # L
        self._downsample_factor = frac.denominator  # M

        self._input_rate = input_rate
        self._output_rate = input_rate * frac.numerator / frac.denominator

        # Design anti-imaging filter (same as signal.py)
        cutoff = self._downsample_factor / self._upsample_factor
        self._filter_coeffs = firls(51, [0, cutoff, cutoff + 0.1, 1], [1, 1, 0, 0])

    @property
    def upsample_factor(self):
        """Upsampling factor L (numerator)."""
        return self._upsample_factor

    @property
    def downsample_factor(self):
        """Downsampling factor M (denominator)."""
        return self._downsample_factor

    @property
    def factor(self):
        """Net resampling factor L/M."""
        return self._upsample_factor / self._downsample_factor

    @property
    def input_rate(self):
        return self._input_rate

    @property
    def output_rate(self):
        return self._output_rate

    @property
    def filter_length(self):
        return len(self._filter_coeffs)

    def call(self, x, padding="same"):
        """
        Interpolate input tensor using scipy (same as signal.py).

        For batched input [B, N], the signal is flattened, processed as
        one continuous signal, then reshaped back to [B, N*factor].

        Args:
            x: [num_samples] or [B, num_samples] tensor
            padding: unused (kept for API compatibility)

        Returns:
            Tuple of (interpolated tensor, output_rate)
        """
        # Store original shape info
        input_shape = tf.shape(x)
        input_ndims = len(x.shape)

        if input_ndims == 1:
            batch_size = 1
            x_flat = x
        elif input_ndims == 2:
            batch_size = input_shape[0]
            x_flat = tf.reshape(x, [-1])
        else:
            raise ValueError(f"Input must be 1D or 2D, got shape {x.shape}")

        # Convert to numpy for scipy processing
        data_np = x_flat.numpy()

        # Step 1: Polyphase resampling (same as signal.py)
        upsampled = resample_poly(
            data_np, self._upsample_factor, self._downsample_factor
        )

        # Step 2: Anti-imaging lowpass filter (same as signal.py)
        filtered = lfilter(
            self._filter_coeffs, 1, np.concatenate([upsampled, np.zeros(100)])
        )

        # Convert back to tensor
        x_out = tf.constant(filtered, dtype=tf.complex64)

        # Reshape back to original batch structure
        if input_ndims == 2:
            # Calculate output samples per batch
            total_output = len(filtered)
            out_samples_per_batch = total_output // batch_size
            x_out = tf.reshape(
                x_out[: batch_size * out_samples_per_batch],
                [batch_size, out_samples_per_batch],
            )

        return x_out, self._output_rate
