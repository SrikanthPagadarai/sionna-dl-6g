"""Least-Squares DPD System for training and inference.

Extends base DPDSystem with Least-Squares-based Digital Pre-Distortion.
Uses closed-form coefficient estimation with indirect learning architecture.
"""

import numpy as np
import tensorflow as tf

from .config import Config
from .system import DPDSystem
from .ls_dpd import LeastSquaresDPD


class LS_DPDSystem(DPDSystem):
    """
    Least-Squares DPD system for training and inference.

    Extends DPDSystem with LS-based predistortion using indirect learning.

    For LS-DPD:
        Uses perform_ls_learning() for closed-form coefficient estimation.

    The indirect learning approach:
    1. Generate baseband signal x
    2. Upsample to PA sample rate
    3. Apply predistorter: u = DPD(x)
    4. Pass through PA: y = PA(u)
    5. Normalize by PA gain: y_norm = y / G
    6. LS coefficient update based on error

    Args:
        training: Whether in training mode
        config: Config instance with system parameters
        dpd_order: DPD polynomial order (default: 7)
        dpd_memory_depth: DPD memory depth (default: 4)
        ls_nIterations: Number of LS iterations (default: 3)
        ls_learning_rate: LS learning rate (default: 0.75)
        ls_learning_method: LS method - 'newton' or 'ema' (default: 'newton')
        rms_input_dbm: Target input RMS power in dBm (default: 0.5)
        pa_sample_rate: PA sample rate in Hz (default: 122.88e6)
    """

    def __init__(
        self,
        training: bool,
        config: Config,
        dpd_order: int = 7,
        dpd_memory_depth: int = 4,
        ls_nIterations: int = 3,
        ls_learning_rate: float = 0.75,
        ls_learning_method: str = "newton",
        rms_input_dbm: float = 0.5,
        pa_sample_rate: float = 122.88e6,
        **kwargs,
    ):
        super().__init__(
            training=training,
            config=config,
            rms_input_dbm=rms_input_dbm,
            pa_sample_rate=pa_sample_rate,
            **kwargs,
        )

        # Least-Squares DPD layer
        self._dpd = LeastSquaresDPD(
            params={
                "order": dpd_order,
                "memory_depth": dpd_memory_depth,
                "nIterations": ls_nIterations,
                "learning_rate": ls_learning_rate,
                "learning_method": ls_learning_method,
            }
        )

    def _forward_signal_path(self, x):
        """
        Forward signal through predistorter and PA (steps 1-3 of indirect learning).

        For LS-DPD: No normalization needed.

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            dict with:
                - u: predistorted signal (original scale)
                - u_norm: predistorted signal (same as u for LS)
                - y_comp: gain-compensated PA output
                - x_scale: input normalization scale factor (1.0 for LS)
        """
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
        Training forward pass for LS-DPD.

        LS-DPD does not use gradient-based training. Use perform_ls_learning() instead.

        Args:
            x: [B, num_samples] input signal at PA rate

        Raises:
            ValueError: Always, as LS-DPD uses perform_ls_learning() instead
        """
        raise ValueError(
            "_training_forward() is for NN-DPD only. "
            "Use perform_ls_learning() for LS-DPD."
        )

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
        """
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
