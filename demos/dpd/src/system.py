# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Base DPD system providing the common infrastructure for DPD training and
inference.

This base class implements the shared signal processing pipeline that both DPD
methods (neural network, least-squares, etc.) build upon. The base class
handles signal generation, upsampling, PA modeling, and gain normalization,
while subclasses implement the specific predistortion algorithms.

The system follows the indirect learning architecture where the predistorter
is trained by learning the PA's inverse from its output.
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
    Base class for DPD systems implementing indirect learning architecture.

    Provides the common infrastructure for all DPD methods:

    - OFDM signal generation via Sionna
    - Upsampling from baseband to PA sample rate
    - Power amplifier model with memory effects
    - PA gain estimation and normalization
    - Receiver chain for inference-time EVM measurement

    Subclasses must implement:

    - ``_forward_signal_path()``: Signal flow through DPD and PA
    - ``_training_forward()``: Training loss computation
    - ``_inference_forward()``: Inference output generation

    Parameters
    ----------
    training : bool
        Operating mode. True for training (receiver not instantiated),
        False for inference (receiver instantiated).
    config : Config
        Configuration object with RF and OFDM parameters.
    rms_input_dbm : float, optional
        Target RMS power for PA input in dBm. Default is 0.5 dBm,
        which drives the considered PA into compression.
    pa_sample_rate : float, optional
        PA operating sample rate in Hz. Default is 122.88 MHz
        (8x the 15.36 MHz baseband rate for 1024-FFT, 15 kHz spacing).
    **kwargs
        Additional keyword arguments passed to Keras Layer.

    Attributes
    ----------
    dpd : Layer or None
        The predistorter layer (set by subclass).
    minimal_ofdm_receiver : Rx or None
        Minimal OFDM receiver for EVM measurement (inference mode only).
    signal_fs : float
        Baseband signal sample rate in Hz.
    pa_sample_rate : float
        PA operating sample rate in Hz.
    fft_size : int
        OFDM FFT size.
    cp_length : int
        Cyclic prefix length in samples.
    num_ofdm_symbols : int
        Number of OFDM symbols per slot.

    Notes
    -----
    **Indirect Learning Architecture:**

    The key insight is that at convergence, the predistorter output ``u``
    equals the ideal PA input that would produce linear output. By training
    a postdistorter on the PA output to reproduce ``u``, it learns the PA
    inverse. The trained postdistorter weights are then copied to the
    predistorter.

    The training loop:

    1. Generate baseband signal ``x``
    2. Apply predistorter: ``u = DPD(x)``
    3. Pass through PA: ``y = PA(u)``
    4. Normalize by PA gain: ``y_norm = y / G``
    5. Compute loss: ``L = ||DPD(y_norm) - u||^2``
    6. Update DPD weights via backpropagation

    **Why Gain Normalization?**

    The PA has linear gain G in addition to nonlinear distortion. Without
    normalization, the postdistorter would learn to invert both the gain
    and nonlinearity. By dividing by G, the nonlinear component is
    isolated for the postdistorter to learn.

    - ``estimate_pa_gain()`` should be called before training
    - PA gain is estimated once and remains fixed during training

    Example
    -------
    Subclasses use this base class as follows:

    >>> class MyDPDSystem(DPDSystem):
    ...     def __init__(self, training, config, **kwargs):
    ...         super().__init__(training, config, **kwargs)
    ...         self._dpd = MyDPDLayer()  # Set predistorter
    ...
    ...     def _forward_signal_path(self, x):
    ...         # Implement signal flow
    ...         ...
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

        # Baseband sample rate derived from config (fft_size * subcarrier_spacing).
        self._signal_fs = config.signal_sample_rate

        # Build transmitter once at init, not inside tf.function.
        # This avoids repeated graph construction overhead.
        self._tx = Tx(config)

        # Rational resampler for baseband -> PA rate conversion.
        self._interpolator = Interpolator(
            input_rate=self._signal_fs,
            output_rate=self._pa_sample_rate,
        )

        # Memory polynomial PA model (order=7, memory_depth=4 fixed).
        self._pa = PowerAmplifier()

        # DPD layer placeholder - subclass must set this.
        self._dpd = None

        # PA small-signal gain, estimated once before training.
        # Stored as tf.Variable for graph-mode compatibility.
        self._pa_gain = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self._pa_gain_initialized = False

        # Receiver only needed for inference (EVM measurement).
        # Not instantiated during training.
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

        # Cache config parameters for property access.
        self._fft_size = config.fft_size
        self._cp_length = config.cyclic_prefix_length
        self._num_ofdm_symbols = config.num_ofdm_symbols

        # Precompute subcarrier index ranges for frequency-domain symbol extraction.
        # Used by generate_signal() when return_extras=True.
        num_guard_lower = config.num_guard_carriers[0]
        num_guard_upper = config.num_guard_carriers[1]
        dc_null = config.dc_null
        self._lower_start = num_guard_lower
        self._lower_end = config.fft_size // 2
        self._upper_start = config.fft_size // 2 + (1 if dc_null else 0)
        self._upper_end = config.fft_size - num_guard_upper

    @property
    def dpd(self):
        """Layer: The predistorter layer (set by subclass)."""
        return self._dpd

    @property
    def minimal_ofdm_receiver(self):
        """Rx or None: OFDM receiver (only available in inference mode)."""
        return self._rx

    @property
    def signal_fs(self):
        """float: Baseband signal sample rate in Hz."""
        return self._signal_fs

    @property
    def pa_sample_rate(self):
        """float: PA operating sample rate in Hz."""
        return self._pa_sample_rate

    @property
    def fft_size(self):
        """int: OFDM FFT size."""
        return self._fft_size

    @property
    def cp_length(self):
        """int: Cyclic prefix length in samples."""
        return self._cp_length

    @property
    def num_ofdm_symbols(self):
        """int: Number of OFDM symbols per slot."""
        return self._num_ofdm_symbols

    def estimate_pa_gain(self, num_samples=10000):
        """
        Estimate PA small-signal (linear) gain.

        Measures the PA's voltage gain in its linear operating region using
        a low-amplitude test signal. This gain value is used to normalize
        the PA output before feeding it to the postdistorter.

        Parameters
        ----------
        num_samples : int, optional
            Number of random samples for gain estimation. More samples
            reduce variance. Default is 10000.

        Returns
        -------
        float
            Estimated voltage gain (linear scale, not dB).

        Notes
        -----
        This method should be called once before training begins. The
        estimated gain is stored internally and used by the forward pass
        to normalize PA output. Calling this multiple times will update
        the stored gain value.

        The gain is estimated in the linear region (low amplitude) where
        higher-order polynomial terms are negligible, giving just the
        first-order (linear) response.
        """
        gain = self._pa.estimate_gain(num_samples)

        self._pa_gain.assign(gain)
        self._pa_gain_initialized = True

        return float(gain.numpy())

    @staticmethod
    def normalize_to_rms(data, target_rms):
        """
        Normalize signal to target RMS power level.

        Scales the input signal so its RMS power matches the target,
        specified in dBm (assuming 50 ohm impedance).

        Parameters
        ----------
        data : tf.Tensor
            Input signal, shape ``[batch, num_samples]``, complex dtype.
        target_rms : float
            Target RMS power in dBm.

        Returns
        -------
        normalized : tf.Tensor
            Scaled signal with target RMS power, same shape as input.
        scale_factor : tf.Tensor
            Multiplicative scale factor applied (useful for inverse scaling).

        Notes
        -----
        The normalization treats all batches as one concatenated signal
        (global statistics), ensuring consistent power across the batch.

        Power conversion: ``P_watts = 10^((P_dBm - 30) / 10)``

        For 50 ohm systems: ``V_rms = sqrt(50 * P_watts)``
        """
        # Compute total signal energy using magnitude (avoids complex dtype issues).
        abs_data = tf.abs(data)
        sum_sq = tf.reduce_sum(abs_data * abs_data)
        norm = tf.sqrt(sum_sq)

        n = tf.cast(tf.size(data), tf.float32)

        # Convert dBm to watts: P = 10^((dBm - 30) / 10)
        target_power = tf.constant(10 ** ((target_rms - 30) / 10), dtype=tf.float32)

        # Scale factor to achieve target RMS voltage (50 ohm impedance assumed).
        scale_factor = tf.sqrt(50.0 * n * target_power) / norm

        normalized = data * tf.cast(scale_factor, data.dtype)

        return normalized, scale_factor

    def generate_signal(self, batch_size, return_extras=False):
        """
        Generate a batch of OFDM baseband signals upsampled to PA rate.

        Creates random transmit signals through the full Tx chain
        (bit generation -> LDPC -> QAM -> OFDM), normalizes power, and
        upsamples to the PA sample rate.

        Parameters
        ----------
        batch_size : int or tf.Tensor
            Number of independent signals to generate.
        return_extras : bool, optional
            If True, return additional data for constellation plotting
            and receiver processing. Default is False.

        Returns
        -------
        tf.Tensor or dict
            If ``return_extras=False``:
                Upsampled signal, shape ``[batch, num_samples]``.

            If ``return_extras=True``:
                Dictionary containing:

                - ``tx_upsampled`` : tf.Tensor
                    Upsampled signal at PA rate.
                - ``tx_baseband`` : tf.Tensor
                    Original baseband signal (flattened) for sync reference.
                - ``x_rg`` : tf.Tensor
                    Resource grid (frequency-domain, all subcarriers).
                - ``fd_symbols`` : tf.Tensor
                    Data subcarrier symbols only, shape
                    ``[num_data_subcarriers, num_symbols]``.

        Notes
        -----
        The ``fd_symbols`` output excludes guard bands and DC null,
        containing only the data-bearing subcarriers in the same order
        as the receiver expects for equalization.
        """
        # Generate OFDM signal through pre-built transmitter.
        batch_size_tensor = tf.cast(batch_size, tf.int32)
        tx_out = self._tx(batch_size_tensor)
        x_time = tx_out["x_time"]  # [B, 1, 1, num_samples]
        x_rg = tx_out["x_rg"]  # [B, 1, 1, num_symbols, fft_size]

        # Remove singleton dimensions: [B, 1, 1, samples] -> [B, samples].
        tx = tf.squeeze(x_time, axis=(1, 2))

        # Keep flattened baseband copy for receiver time synchronization.
        tx_baseband = tf.reshape(x_time, [-1])

        # Normalize to target PA input power level.
        tx_normalized, _ = self.normalize_to_rms(tx, self._rms_input_dbm)

        # Upsample from baseband rate to PA rate.
        tx_upsampled, _ = self._interpolator(tx_normalized)

        if not return_extras:
            return tx_upsampled

        # Extract frequency-domain data symbols for constellation comparison.
        # Take first batch element, remove singleton dims.
        x_rg_squeezed = tf.squeeze(x_rg[0], axis=(0, 1))  # [num_sym, fft_size]

        # Extract data subcarriers (exclude guards and DC).
        fd_lower = tf.transpose(x_rg_squeezed[:, self._lower_start : self._lower_end])
        fd_upper = tf.transpose(x_rg_squeezed[:, self._upper_start : self._upper_end])
        fd_symbols = tf.concat([fd_lower, fd_upper], axis=0)

        return {
            "tx_upsampled": tx_upsampled,
            "tx_baseband": tx_baseband,
            "x_rg": x_rg,
            "fd_symbols": fd_symbols,
        }

    def call(self, batch_size_or_signal, training=None):
        """
        Forward pass through the DPD system.

        Handles both signal generation (from batch size) and processing
        of pre-generated signals. Dispatches to training or inference
        path based on mode.

        Parameters
        ----------
        batch_size_or_signal : int, tf.Tensor (scalar), or tf.Tensor (2D)
            Either:

            - Python int or scalar tensor: interpreted as batch size,
              signal will be generated internally.
            - 2D tensor ``[batch, samples]``: pre-generated signal to process.

        training : bool or None, optional
            Override the instance's training mode. If None, uses the
            mode specified at construction.

        Returns
        -------
        tf.Tensor or dict
            **Training mode:**
                Scalar loss value (for gradient-based optimization).

            **Inference mode:**
                Dictionary with keys:

                - ``pa_input`` : PA input signal (upsampled, before DPD)
                - ``pa_output_no_dpd`` : PA output without predistortion
                - ``pa_output_with_dpd`` : PA output with predistortion

        Raises
        ------
        ValueError
            If ``batch_size_or_signal`` is not a valid type.

        Notes
        -----
        The ability to accept pre-generated signals allows for consistent
        evaluation: generate once, then compare with/without DPD on the
        exact same input.
        """
        is_training = training if training is not None else self._training

        # Determine if input is batch_size or pre-generated signal.
        if isinstance(batch_size_or_signal, int):
            x = self.generate_signal(batch_size_or_signal)
        elif isinstance(batch_size_or_signal, tf.Tensor):
            if len(batch_size_or_signal.shape) == 0:
                # Scalar tensor -> treat as batch_size.
                x = self.generate_signal(batch_size_or_signal)
            else:
                # Multi-dimensional tensor -> pre-generated signal.
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
        """
        Normalize signal to unit average power.

        Parameters
        ----------
        x : tf.Tensor
            Input signal (any shape).

        Returns
        -------
        normalized : tf.Tensor
            Signal with unit average power.
        scale : tf.Tensor
            RMS value of original signal (for rescaling).

        Notes
        -----
        Unit power normalization ensures consistent DPD behavior
        regardless of input signal amplitude. The scale factor can
        be used to restore original amplitude after processing.
        """
        power = tf.reduce_mean(tf.abs(x) ** 2)
        scale = tf.sqrt(power + 1e-12)  # Small epsilon prevents division by zero.
        return x / tf.cast(scale, x.dtype), scale

    def _forward_signal_path(self, x):
        """
        Forward signal through predistorter and PA.

        This is the shared signal path implementing steps 1-3 of indirect
        learning. Subclasses must implement this to define their specific
        normalization and signal flow behavior.

        Parameters
        ----------
        x : tf.Tensor
            Input signal at PA rate, shape ``[batch, num_samples]``.

        Returns
        -------
        dict
            Signal path outputs (subclass-specific), typically including:

            - ``u`` : Predistorted signal (original scale)
            - ``u_norm`` : Predistorted signal (normalized, for loss)
            - ``y_comp`` : Gain-compensated PA output
            - ``x_scale`` : Input normalization scale factor

        Raises
        ------
        NotImplementedError
            Base class does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement _forward_signal_path()")

    def _training_forward(self, x):
        """
        Execute training forward pass and compute loss.

        Parameters
        ----------
        x : tf.Tensor
            Input signal at PA rate, shape ``[batch, num_samples]``.

        Returns
        -------
        tf.Tensor or dict
            Loss value or training results (subclass-specific).

        Raises
        ------
        NotImplementedError
            Base class does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement _training_forward()")

    def _inference_forward(self, x):
        """
        Execute inference forward pass.

        Parameters
        ----------
        x : tf.Tensor
            Input signal at PA rate, shape ``[batch, num_samples]``.

        Returns
        -------
        dict
            Inference outputs including PA input and outputs with/without DPD.

        Raises
        ------
        NotImplementedError
            Base class does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement _inference_forward()")
