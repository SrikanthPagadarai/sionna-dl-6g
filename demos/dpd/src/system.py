"""Base DPD System for training and inference.

Wraps signal generation, upsampling, and PA into a single differentiable system.
Subclasses implement specific DPD methods (Neural Network or Least-Squares).
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .config import Config
from .power_amplifier import PowerAmplifier
from .interpolator import Interpolator
from .tx import Tx
from .rx import Rx


class DPDSystem(Layer):
    """
    Base DPD system for training and inference.

    Wraps signal generation (Sionna Tx), upsampling, and PA.
    Subclasses implement specific DPD methods.

    The indirect learning approach:
    1. Generate baseband signal x
    2. Upsample to PA sample rate
    3. Apply predistorter: u = DPD(x)
    4. Pass through PA: y = PA(u)
    5. Normalize by PA gain: y_norm = y / G
    6. Train postdistorter: loss = ||DPD(y_norm) - u||²

    Args:
        training: Whether in training mode
        config: Config instance with system parameters
        rms_input_dbm: Target input RMS power in dBm (default: 0.5)
        pa_sample_rate: PA sample rate in Hz (default: 122.88e6)
    """

    def __init__(
        self,
        training: bool,
        config: Config,
        rms_input_dbm: float = 0.5,
        pa_sample_rate: float = 122.88e6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._training = training
        self._config = config
        self._rms_input_dbm = rms_input_dbm
        self._pa_sample_rate = pa_sample_rate

        # Signal sample rate from config
        self._signal_fs = config.signal_sample_rate

        # Build transmitter once (not inside tf.function)
        self._tx = Tx(config)

        # Interpolator for upsampling
        self._interpolator = Interpolator(
            input_rate=self._signal_fs,
            output_rate=self._pa_sample_rate,
        )

        # Power Amplifier (uses fixed order=7, memory_depth=4)
        self._pa = PowerAmplifier()

        # DPD layer - to be set by subclass
        self._dpd = None

        # PA gain (estimated during first forward pass)
        self._pa_gain = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self._pa_gain_initialized = False

        # OFDM receiver for inference (demodulation, equalization, EVM)
        if not training:
            self._rx = Rx(
                signal_fs=self._signal_fs,
                pa_sample_rate=self._pa_sample_rate,
                fft_size=config.fft_size,
                cp_length=config.cyclic_prefix_length,
                num_ofdm_symbols=config.num_ofdm_symbols,
                num_guard_lower=config.num_guard_carriers[0],
                num_guard_upper=config.num_guard_carriers[1],
                dc_null=config.dc_null,
            )
        else:
            self._rx = None

        # Store config parameters for property access
        self._fft_size = config.fft_size
        self._cp_length = config.cyclic_prefix_length
        self._num_ofdm_symbols = config.num_ofdm_symbols

        # Subcarrier indices for generate_signal() frequency-domain symbol extraction
        num_guard_lower = config.num_guard_carriers[0]
        num_guard_upper = config.num_guard_carriers[1]
        dc_null = config.dc_null
        self._lower_start = num_guard_lower
        self._lower_end = config.fft_size // 2
        self._upper_start = config.fft_size // 2 + (1 if dc_null else 0)
        self._upper_end = config.fft_size - num_guard_upper

    @property
    def dpd(self):
        """Access the DPD layer."""
        return self._dpd

    @property
    def ofdm_receiver(self):
        """Access the OFDM receiver (only available in inference mode)."""
        return self._rx

    @property
    def signal_fs(self):
        """Signal sample rate."""
        return self._signal_fs

    @property
    def pa_sample_rate(self):
        """PA sample rate."""
        return self._pa_sample_rate

    @property
    def fft_size(self):
        """FFT size for OFDM."""
        return self._fft_size

    @property
    def cp_length(self):
        """Cyclic prefix length."""
        return self._cp_length

    @property
    def num_ofdm_symbols(self):
        """Number of OFDM symbols."""
        return self._num_ofdm_symbols

    def estimate_pa_gain(self, num_samples=10000):
        """
        Estimate PA small-signal gain by measuring input/output power ratio.

        This should be called once before training to determine G.

        Args:
            num_samples: Number of samples to use for estimation

        Returns:
            Estimated gain (linear scale)
        """
        gain = self._pa.estimate_gain(num_samples)

        self._pa_gain.assign(gain)
        self._pa_gain_initialized = True

        return float(gain.numpy())

    @staticmethod
    def normalize_to_rms(data, target_rms):
        """
        Normalize batched signal to target RMS power.

        Computes global statistics across all batches (as if concatenated).

        Args:
            data: [B, num_samples] complex tensor
            target_rms: target RMS in dBm

        Returns:
            normalized data [B, num_samples], scale_factor
        """
        # Compute norm using abs to avoid complex->float cast warning
        abs_data = tf.abs(data)  # float32
        sum_sq = tf.reduce_sum(abs_data * abs_data)
        norm = tf.sqrt(sum_sq)

        n = tf.cast(tf.size(data), tf.float32)

        target_power = tf.constant(10 ** ((target_rms - 30) / 10), dtype=tf.float32)
        scale_factor = tf.sqrt(50.0 * n * target_power) / norm

        normalized = data * tf.cast(scale_factor, data.dtype)

        return normalized, scale_factor

    def generate_signal(self, batch_size, return_extras=False):
        """
        Generate a batch of baseband signals.

        Args:
            batch_size: Number of signals to generate (Python int or tf.Tensor)
            return_extras: If True, return additional info for constellation plotting

        Returns:
            If return_extras=False: tx_upsampled [B, num_samples]
            If return_extras=True: dict with tx_upsampled, tx_baseband, x_rg, fd_symbols
        """
        # Call pre-built transmitter
        batch_size_tensor = tf.cast(batch_size, tf.int32)
        tx_out = self._tx(batch_size_tensor)
        x_time = tx_out["x_time"]  # [B, 1, 1, num_samples]
        x_rg = tx_out["x_rg"]  # [B, 1, 1, num_symbols, fft_size]

        # Remove singleton dimensions: [B, num_samples]
        tx = tf.squeeze(x_time, axis=(1, 2))

        # Keep baseband copy for constellation sync (flattened)
        tx_baseband = tf.reshape(x_time, [-1])

        # Normalize to target RMS
        tx_normalized, _ = self.normalize_to_rms(tx, self._rms_input_dbm)

        # Upsample to PA rate
        tx_upsampled, _ = self._interpolator(tx_normalized)

        if not return_extras:
            return tx_upsampled

        # Extract frequency-domain symbols for constellation comparison
        # x_rg shape: [B, 1, 1, num_symbols, fft_size] -> [num_symbols, fft_size]
        x_rg_squeezed = tf.squeeze(
            x_rg[0], axis=(0, 1)
        )  # First batch, [num_sym, fft_size]

        # Extract data subcarriers
        fd_lower = tf.transpose(x_rg_squeezed[:, self._lower_start : self._lower_end])
        fd_upper = tf.transpose(x_rg_squeezed[:, self._upper_start : self._upper_end])
        fd_symbols = tf.concat(
            [fd_lower, fd_upper], axis=0
        )  # [num_subcarriers, num_symbols]

        return {
            "tx_upsampled": tx_upsampled,
            "tx_baseband": tx_baseband,
            "x_rg": x_rg,
            "fd_symbols": fd_symbols,
        }

    def call(self, batch_size_or_signal, training=None):
        """
        Forward pass through the DPD system.

        In training mode, returns the indirect learning loss (NN-DPD only).
        In inference mode, returns dict with PA outputs (with and without DPD).

        Args:
            batch_size_or_signal: Either:
                - Batch size (Python int or scalar tensor) to generate signal
                - Pre-generated signal tensor [B, num_samples]
            training: Override training mode if specified

        Returns:
            Training: scalar loss value (NN-DPD)
            Inference: dict with 'pa_input', 'pa_output_no_dpd', 'pa_output_with_dpd'
        """
        is_training = training if training is not None else self._training

        # Determine if input is batch_size or pre-generated signal
        if isinstance(batch_size_or_signal, int):
            # Python int - generate signal
            x = self.generate_signal(batch_size_or_signal)
        elif isinstance(batch_size_or_signal, tf.Tensor):
            if len(batch_size_or_signal.shape) == 0:
                # Scalar tensor - treat as batch_size
                x = self.generate_signal(batch_size_or_signal)
            else:
                # Multi-dimensional tensor - treat as pre-generated signal
                x = batch_size_or_signal
        else:
            raise ValueError(
                f"Expected int, scalar tensor, or signal tensor, "
                f"got {type(batch_size_or_signal)}"
            )

        if is_training:
            return self._training_forward(x)
        else:
            return self._inference_forward(x)

    def _normalize_to_unit_power(self, x):
        """Normalize signal to unit power, return (normalized, scale_factor)."""
        power = tf.reduce_mean(tf.abs(x) ** 2)
        scale = tf.sqrt(power + 1e-12)
        return x / tf.cast(scale, x.dtype), scale

    def _forward_signal_path(self, x):
        """
        Forward signal through predistorter and PA (steps 1-3 of indirect learning).

        This is the shared signal path for both NN-DPD and LS-DPD:
            Step 1: Apply predistorter: u = DPD(x)
            Step 2: Pass through PA: y = PA(u)
            Step 3: Compensate for PA gain: y_comp = y / G

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            dict with:
                - u: predistorted signal (original scale)
                - u_norm: predistorted signal (normalized scale, for NN loss)
                - y_comp: gain-compensated PA output
                - x_scale: input normalization scale factor

        Note: Subclasses may override this for different normalization behavior.
        """
        raise NotImplementedError("Subclasses must implement _forward_signal_path()")

    def _training_forward(self, x):
        """
        Training forward pass.

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            Loss value or training result (subclass-specific)
        """
        raise NotImplementedError("Subclasses must implement _training_forward()")

    def _inference_forward(self, x):
        """
        Inference forward pass.

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            dict with PA input and outputs
        """
        raise NotImplementedError("Subclasses must implement _inference_forward()")
