import numpy as np
import tensorflow as tf
from scipy.signal import welch, firls, lfilter, resample_poly
from fractions import Fraction


class Signal:
    """Wrapper class to encapsulate signals for easier manipulation."""

    def __init__(self, data, current_fs, desired_rms=None):
        # Keep as tensor if already tensor, otherwise convert
        if not isinstance(data, tf.Tensor):
            data = tf.constant(data, dtype=tf.complex64)
        self.data = tf.reshape(data, [-1])
        self.current_fs = current_fs
        self.original_fs = current_fs
        self.rms_power = None
        self.scale_factor = None
        self.resample_num = None
        self.resample_dem = None

        if desired_rms is not None:
            self._normalize_to_rms(desired_rms)
        self.papr = self._compute_papr()
        self.obw = self._compute_occupied_bandwidth()

    def _normalize_to_rms(self, target_rms):
        """Normalize signal to target RMS power in dBm."""
        norm = tf.cast(tf.norm(self.data), tf.float32)
        n = tf.cast(tf.shape(self.data)[0], tf.float32)
        target_power = tf.constant(10 ** ((target_rms - 30) / 10), dtype=tf.float32)
        self.scale_factor = tf.sqrt(50.0 * n * target_power) / norm
        self.data = self.data * tf.cast(self.scale_factor, tf.complex64)

        # Verify RMS
        new_norm = tf.cast(tf.norm(self.data), tf.float32)
        self.rms_power = float(
            10 * tf.math.log(new_norm**2 / 50.0 / n) / tf.math.log(10.0) + 30
        )
        if abs(target_rms - self.rms_power) > 0.01:
            raise ValueError("RMS normalization failed.")

    def _compute_papr(self):
        """Compute Peak-to-Average Power Ratio in dB."""
        n = tf.cast(tf.shape(self.data)[0], tf.float32)
        norm = tf.cast(tf.norm(self.data), tf.float32)
        peak = tf.cast(tf.reduce_max(tf.abs(self.data)), tf.float32)
        return float(20.0 * tf.math.log(peak * tf.sqrt(n) / norm) / tf.math.log(10.0))

    def _compute_occupied_bandwidth(self):
        """
        Compute occupied bandwidth (99% power bandwidth).
        Requires numpy for scipy.welch.
        """
        try:
            data_np = self.data.numpy()
            f, psd = welch(
                data_np,
                fs=self.current_fs,
                nperseg=min(len(data_np), 256),
                return_onesided=False,
            )
            f, psd = np.fft.fftshift(f), np.fft.fftshift(psd)
            cumulative = np.cumsum(psd) / np.sum(psd)
            return f[np.argmax(cumulative >= 0.995)] - f[np.argmax(cumulative >= 0.005)]
        except Exception:
            return 0

    def upsample(self, desired_rate):
        """Upsample signal to desired rate. Requires numpy for scipy.resample_poly."""
        frac = Fraction(desired_rate / self.current_fs).limit_denominator(1000)
        self.resample_num, self.resample_dem = frac.numerator, frac.denominator

        # Convert to numpy for scipy resample_poly
        data_np = self.data.numpy()
        upsampled = resample_poly(data_np, self.resample_num, self.resample_dem)
        self.current_fs *= self.resample_num / self.resample_dem

        # Anti-imaging lowpass filter (requires numpy)
        cutoff = self.resample_dem / self.resample_num
        b = firls(51, [0, cutoff, cutoff + 0.1, 1], [1, 1, 0, 0])
        filtered = lfilter(b, 1, np.concatenate([upsampled, np.zeros(100)]))

        # Convert back to tensor
        self.data = tf.constant(filtered, dtype=tf.complex64)

    def measure_all_powers(self):
        """
        Measure power in L1, main, and U1 channels.
        Returns array [L1, Main, U1] in dBm.
        """
        return np.array(
            [self._measure_channel_power(ch) for ch in ("L1", "main", "U1")]
        )

    def _measure_channel_power(self, channel):
        """Measure power in a specific channel. Requires numpy for scipy.welch."""
        ibw, offset = self._get_bandwidth_params()
        if ibw is None:
            return -99

        bounds = {
            "main": (-0.5 * ibw, 0.5 * ibw),
            "L1": (-0.5 * ibw - offset, 0.5 * ibw - offset),
            "U1": (-0.5 * ibw + offset, 0.5 * ibw + offset),
        }
        lower, upper = bounds[channel]

        try:
            power = self._bandpower(lower, upper) / 50
            return 10 * np.log10(power / 0.001)
        except Exception:
            return -99

    def _get_bandwidth_params(self):
        """Get integration bandwidth and offset based on occupied bandwidth."""
        bw_configs = [
            (15e6, 20e6, 18e6, 20e6),
            (8e6, 10e6, 9e6, 10e6),
            (3e6, 5e6, 4.5e6, 5e6),
        ]
        for low, high, ibw, offset in bw_configs:
            if low <= self.obw <= high:
                return ibw, offset

        if self.obw > 0:
            return 0.9 * self.obw, self.obw
        return None, None

    def _bandpower(self, f_low, f_high):
        """
        Compute band power between frequency bounds.
        Requires numpy for scipy.welch.
        """
        data_np = self.data.numpy()
        f, psd = welch(
            data_np,
            fs=self.current_fs,
            nperseg=min(len(data_np), 256),
            return_onesided=False,
        )
        f, psd = np.fft.fftshift(f), np.fft.fftshift(psd)
        mask = (f >= f_low) & (f <= f_high)
        return np.sum(psd[mask]) * (f[1] - f[0])
