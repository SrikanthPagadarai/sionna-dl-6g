"""DPD System for training and inference.

Wraps signal generation, upsampling, DPD, and PA into a single differentiable system.
Supports both Neural Network (nn) and Least-Squares (ls) DPD methods.
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.phy.ofdm import OFDMDemodulator

from .nn_dpd import NeuralNetworkDPD
from .ls_dpd import LeastSquaresDPD
from .power_amplifier import PowerAmplifier
from .interpolator import Interpolator
from .tx import Tx
from .utilities import normalize_to_rms


class DPDSystem(Layer):
    """
    Complete DPD system for training and inference.

    Wraps signal generation (Sionna Tx), upsampling, DPD, and PA.
    Supports both Neural Network and Least-Squares DPD methods.

    For NN-DPD:
        Returns indirect learning loss: ||DPD(PA_output/G) - predistorter_output||²
    For LS-DPD:
        Uses perform_ls_learning() for closed-form coefficient estimation.

    The indirect learning approach:
    1. Generate baseband signal x
    2. Upsample to PA sample rate
    3. Apply predistorter: u = DPD(x)
    4. Pass through PA: y = PA(u)
    5. Normalize by PA gain: y_norm = y / G
    6. Train postdistorter: loss = ||DPD(y_norm) - u||²

    Args:
        training: Whether in training mode
        dpd_method: DPD method - "nn" (Neural Network) or "ls" (Least-Squares)
        tx_config_path: Path to transmitter configuration JSON
        pa_order: PA polynomial order (default: 7)
        pa_memory_depth: PA memory depth (default: 4)
        dpd_order: DPD polynomial order for LS-DPD (default: 7)
        dpd_memory_depth: DPD memory depth (default: 4)
        dpd_num_filters: DPD hidden layer size for NN-DPD (default: 64)
        dpd_num_layers_per_block: Layers per residual block for NN-DPD (default: 2)
        dpd_num_res_blocks: Number of residual blocks for NN-DPD (default: 3)
        ls_nIterations: Number of LS iterations (default: 3)
        ls_learning_rate: LS learning rate (default: 0.75)
        ls_learning_method: LS method - 'newton' or 'ema' (default: 'newton')
        rms_input_dbm: Target input RMS power in dBm (default: 0.5)
        pa_sample_rate: PA sample rate in Hz (default: 122.88e6)
        use_tf_interpolator: Use TF graph-mode compatible interpolator (default: True)
    """

    def __init__(
        self,
        training: bool,
        dpd_method: str = "nn",
        tx_config_path: str = "src/tx_config.json",
        pa_order: int = 7,
        pa_memory_depth: int = 4,
        dpd_order: int = 7,
        dpd_memory_depth: int = 4,
        dpd_num_filters: int = 64,
        dpd_num_layers_per_block: int = 2,
        dpd_num_res_blocks: int = 3,
        ls_nIterations: int = 3,
        ls_learning_rate: float = 0.75,
        ls_learning_method: str = "newton",
        rms_input_dbm: float = 0.5,
        pa_sample_rate: float = 122.88e6,
        use_tf_interpolator: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if dpd_method not in ("nn", "ls"):
            raise ValueError(f"dpd_method must be 'nn' or 'ls', got '{dpd_method}'")

        self._training = training
        self._dpd_method = dpd_method
        self._tx_config_path = tx_config_path
        self._rms_input_dbm = rms_input_dbm
        self._pa_sample_rate = pa_sample_rate
        self._use_tf_interpolator = use_tf_interpolator

        # Load configuration
        self._tx_config = json.loads(Path(tx_config_path).read_text())

        # Compute signal sample rate from FFT size and subcarrier spacing
        fft_size = float(self._tx_config["rg"]["fft_size"])
        subcarrier_spacing = float(self._tx_config["rg"]["subcarrier_spacing"])
        self._signal_fs = fft_size * subcarrier_spacing

        # Build transmitter once (not inside tf.function)
        self._tx = Tx(tx_config_path)

        # Interpolator for upsampling
        self._interpolator = Interpolator(
            input_rate=self._signal_fs,
            output_rate=self._pa_sample_rate,
        )

        # Power Amplifier
        self._pa = PowerAmplifier(order=pa_order, memory_depth=pa_memory_depth)

        # DPD layer - either Neural Network or Least-Squares
        if dpd_method == "nn":
            self._dpd = NeuralNetworkDPD(
                memory_depth=dpd_memory_depth,
                num_filters=dpd_num_filters,
                num_layers_per_block=dpd_num_layers_per_block,
                num_res_blocks=dpd_num_res_blocks,
            )
        else:  # "ls"
            self._dpd = LeastSquaresDPD(
                params={
                    "order": dpd_order,
                    "memory_depth": dpd_memory_depth,
                    "nIterations": ls_nIterations,
                    "learning_rate": ls_learning_rate,
                    "learning_method": ls_learning_method,
                }
            )

        # Loss function with scaling for better gradient flow (NN-DPD only)
        self._loss_fn = tf.keras.losses.MeanSquaredError()
        self._loss_scale = 1000.0  # Scale up loss for better monitoring

        # PA gain (estimated during first forward pass)
        self._pa_gain = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self._pa_gain_initialized = False

        # Demodulation setup for constellation plotting
        self._fft_size = int(self._tx_config["rg"]["fft_size"])
        self._cp_length = int(self._tx_config["rg"]["cyclic_prefix_length"])
        self._num_ofdm_symbols = int(self._tx_config["rg"]["num_ofdm_symbols"])
        self._num_guard_lower = int(self._tx_config["rg"]["num_guard_carriers"][0])
        self._num_guard_upper = int(self._tx_config["rg"]["num_guard_carriers"][1])
        self._dc_null = bool(self._tx_config["rg"]["dc_null"])

        # Sionna demodulator
        self._ofdm_demod = OFDMDemodulator(
            fft_size=self._fft_size,
            l_min=0,
            cyclic_prefix_length=self._cp_length,
        )

        # Subcarrier indices for data extraction
        self._lower_start = self._num_guard_lower
        self._lower_end = self._fft_size // 2
        self._upper_start = self._fft_size // 2 + (1 if self._dc_null else 0)
        self._upper_end = self._fft_size - self._num_guard_upper

    @property
    def dpd(self):
        """Access the DPD layer."""
        return self._dpd

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
        # Generate a test signal with low amplitude (linear region)
        test_input = tf.complex(
            tf.random.normal([num_samples], stddev=0.1),
            tf.random.normal([num_samples], stddev=0.1),
        )

        # Pass through PA
        test_output = self._pa(test_input)

        # Compute gain as sqrt(output_power / input_power)
        input_power = tf.reduce_mean(tf.abs(test_input) ** 2)
        output_power = tf.reduce_mean(tf.abs(test_output) ** 2)
        gain = tf.sqrt(output_power / (input_power + 1e-12))

        self._pa_gain.assign(gain)
        self._pa_gain_initialized = True

        return float(gain.numpy())

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
        tx_normalized, _ = normalize_to_rms(tx, self._rms_input_dbm)

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

        For NN-DPD: Input is normalized for better NN conditioning.
        For LS-DPD: Input is used directly (no normalization).

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            dict with:
                - u: predistorted signal (original scale)
                - u_norm: predistorted signal (normalized scale, for NN loss)
                - y_comp: gain-compensated PA output
                - x_scale: input normalization scale factor
        """
        if self._dpd_method == "nn":
            # NN-DPD: Normalize input for better conditioning
            x_norm, x_scale = self._normalize_to_unit_power(x)
            u_norm = self._dpd(x_norm, training=False)
            u = u_norm * tf.cast(x_scale, u_norm.dtype)
        else:
            # LS-DPD: No normalization needed
            u = self._dpd(x, training=False)
            u_norm = u
            x_scale = tf.constant(1.0, dtype=tf.float32)

        # Step 2: Pass through PA
        y = self._pa(u)

        # Step 3: Compensate for PA gain
        y_comp = y / tf.cast(self._pa_gain, y.dtype)

        return {
            "u": u,
            "u_norm": u_norm,
            "y_comp": y_comp,
            "x_scale": x_scale,
        }

    def _training_forward(self, x):
        """
        Training forward pass with indirect learning (NN-DPD only).

        Complete indirect learning architecture:
            Step 1: Apply predistorter: u = DPD(x)
            Step 2: Pass through PA: y = PA(u)
            Step 3: Compensate for PA gain: y_comp = y / G
            Step 4: Apply postdistorter: u_hat = DPD(y_comp)
            Step 5: Compute loss: ||u - u_hat||²

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            scalar MSE loss

        Raises:
            ValueError: If called with dpd_method="ls"
        """
        if self._dpd_method != "nn":
            raise ValueError(
                "_training_forward() is for NN-DPD only. "
                "Use perform_ls_learning() for LS-DPD."
            )

        # Steps 1-3: Forward through predistorter and PA
        signals = self._forward_signal_path(x)
        u_norm = signals["u_norm"]
        y_comp = signals["y_comp"]

        # Stop gradient on target (predistorter output)
        u_target = tf.stop_gradient(u_norm)

        # Normalize PA output for postdistorter
        y_norm, _ = self._normalize_to_unit_power(y_comp)

        # Step 4: Apply postdistorter (this is what we're training)
        u_hat_norm = self._dpd(y_norm, training=True)

        # Step 5: Compute loss in normalized domain
        u_target_ri = tf.stack(
            [tf.math.real(u_target), tf.math.imag(u_target)], axis=-1
        )
        u_hat_ri = tf.stack(
            [tf.math.real(u_hat_norm), tf.math.imag(u_hat_norm)], axis=-1
        )

        loss = self._loss_fn(u_target_ri, u_hat_ri) * self._loss_scale
        return loss

    def _ls_training_iteration(self, x):
        """
        Single LS-DPD training iteration using indirect learning architecture.

        Complete indirect learning architecture:
            Step 1: Apply predistorter: u = DPD(x)
            Step 2: Pass through PA: y = PA(u)
            Step 3: Compensate for PA gain: y_comp = y / G
            Step 4: Apply postdistorter: u_hat = DPD(y_comp)  [Newton method]
            Step 5: LS coefficient update

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            dict with iteration results (y_power for monitoring)
        """
        # Steps 1-3: Forward through predistorter and PA
        signals = self._forward_signal_path(x)
        u = signals["u"]
        y_comp = signals["y_comp"]

        # Flatten for LS operations
        u_flat = tf.reshape(u, [-1])
        y_flat = tf.reshape(y_comp, [-1])

        # Build basis matrix from gain-compensated PA output
        Y = self._dpd.setup_basis_matrix(y_flat)

        # Get current coefficients
        current_coeffs = self._dpd.coeffs

        # Step 4 & 5: Apply postdistorter and compute LS update
        if self._dpd._learning_method == "newton":
            # Newton method: postdistorter is explicitly applied
            # u_hat = DPD(y_comp) - the postdistorter output
            u_hat = self._dpd.predistort(y_flat)
            # Error signal for LS estimation
            error = u_flat - u_hat
            # Coefficient update: c_new = c + lr * LS_solve(Y, error)
            new_coeffs = (
                current_coeffs
                + self._dpd._learning_rate * self._dpd._ls_estimation(Y, error)
            )
        else:
            # EMA method: direct LS estimation
            # c_new = (1-lr) * c + lr * LS_solve(Y, u)
            new_coeffs = (
                1 - self._dpd._learning_rate
            ) * current_coeffs + self._dpd._learning_rate * self._dpd._ls_estimation(
                Y, u_flat
            )

        # Update coefficients
        self._dpd.coeffs = new_coeffs

        # Return monitoring info
        y_power = 10 * tf.experimental.numpy.log10(
            tf.reduce_mean(tf.abs(y_flat) ** 2) + 1e-12
        )
        return {"y_power": float(y_power.numpy())}

    def perform_ls_learning(self, batch_size, nIterations=None, verbose=False):
        """
        Perform LS-DPD learning using indirect learning architecture.

        Iteratively calls _ls_training_iteration() to update DPD coefficients.

        Args:
            batch_size: Batch size for signal generation
            nIterations: Override number of iterations (uses DPD default if None)
            verbose: Print progress

        Returns:
            Dictionary with learning results:
                - coeffs: Final DPD coefficients
                - coeff_history: Coefficient history across iterations

        Raises:
            ValueError: If dpd_method is not "ls"
        """
        if self._dpd_method != "ls":
            raise ValueError(
                f"perform_ls_learning() requires dpd_method='ls', "
                f"got '{self._dpd_method}'"
            )

        # Generate signal
        x = self.generate_signal(batch_size)

        # Determine number of iterations
        n_iters = nIterations if nIterations is not None else self._dpd._nIterations

        # Initialize coefficient history
        coeff_history = self._dpd.coeffs.numpy().copy()

        if verbose:
            print(
                f"Starting LS-DPD learning: {n_iters} iterations, "
                f"order={self._dpd._order}, memory={self._dpd._memory_depth}"
            )

        # Iterative learning
        for iteration in range(n_iters):
            result = self._ls_training_iteration(x)

            # Record coefficient history
            coeff_history = np.hstack([coeff_history, self._dpd.coeffs.numpy()])

            if verbose:
                print(
                    f"  Iteration {iteration + 1}/{n_iters}: "
                    f"PA output power = {result['y_power']:.2f} dB"
                )

        if verbose:
            print("LS-DPD learning complete.")

        # Store history in DPD layer for plotting
        self._dpd.coeff_history = coeff_history

        return {
            "coeffs": self._dpd.coeffs.numpy(),
            "coeff_history": coeff_history,
        }

    def _inference_forward(self, x):
        """
        Inference forward pass.

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            dict with PA input and outputs
        """
        # PA output without DPD
        pa_output_no_dpd = self._pa(x)

        if self._dpd_method == "nn":
            # NN-DPD: Normalize for DPD, apply DPD, scale back
            x_norm, x_scale = self._normalize_to_unit_power(x)
            x_predistorted_norm = self._dpd(x_norm, training=False)
            x_predistorted = x_predistorted_norm * tf.cast(
                x_scale, x_predistorted_norm.dtype
            )
        else:
            # LS-DPD: Apply DPD directly (no normalization)
            x_predistorted = self._dpd(x, training=False)

        # Pass through PA
        pa_output_with_dpd = self._pa(x_predistorted)

        return {
            "pa_input": x,
            "pa_output_no_dpd": pa_output_no_dpd,
            "pa_output_with_dpd": pa_output_with_dpd,
            "predistorted": x_predistorted,
        }

    def demod(self, signal):
        """
        Demodulate OFDM signal to extract frequency-domain symbols.

        Args:
            signal: [num_samples] complex tensor at baseband sample rate

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
