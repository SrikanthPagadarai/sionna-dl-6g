# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Least-squares DPD system integrating the Memory-Polynomial predistorter.

This subclass provides the complete LS-DPD system that combines the base
DPDSystem infrastructure with closed-form least-squares coefficient
estimation.

The system implements the indirect learning architecture where:

1. A predistorter processes the input signal
2. The PA distorts the predistorted signal
3. A postdistorter (same structure as predistorter) learns to invert the PA
4. Postdistorter coefficients are copied to the predistorter

The key insight is that the postdistorter sees PA output as input and
predistorter output as target - this is a standard supervised learning
problem solvable by least squares.
"""

import numpy as np
import tensorflow as tf

from .config import Config
from .system import DPDSystem
from .ls_dpd import LeastSquaresDPD


class LS_DPDSystem(DPDSystem):
    """
    Complete LS-DPD system with closed-form coefficient estimation.

    Extends the base DPDSystem with a Memory Polynomial predistorter
    trained via least-squares regression. The training uses indirect
    learning architecture with either Newton or EMA update methods.

    Parameters
    ----------
    training : bool
        Operating mode. True for training, False for inference.
    config : Config
        Frozen configuration object with RF and OFDM parameters.
    dpd_order : int, optional
        Maximum polynomial order (must be odd). Default: 7.
    dpd_memory_depth : int, optional
        Number of memory taps per polynomial branch. Default: 4.
    ls_nIterations : int, optional
        Number of indirect learning iterations. Default: 3.
    ls_learning_rate : float, optional
        Coefficient update rate (0-1). Higher values give faster but
        potentially unstable convergence. Default: 0.75.
    ls_learning_method : str, optional
        Coefficient update method:

        - ``'newton'``: Update based on error between predistorter output
          and postdistorter output. More stable.
        - ``'ema'``: Exponential moving average of direct LS solutions.
          Faster convergence but can overshoot.

        Default: ``'newton'``.
    rms_input_dbm : float, optional
        Target RMS power for PA input in dBm. Default: 0.5.
    pa_sample_rate : float, optional
        PA operating sample rate in Hz. Default: 122.88 MHz.
    **kwargs
        Additional keyword arguments passed to base DPDSystem.

    Attributes
    ----------
    dpd : LeastSquaresDPD
        The Memory Polynomial predistorter layer.

    Notes
    -----
    **Newton vs EMA Update Methods:**

    *Newton method* (default):

    .. math::

        \\mathbf{c}_{new} = \\mathbf{c} + \\mu \\cdot \\text{LS}(\\mathbf{Y}, \\mathbf{u} - \\hat{\\mathbf{u}})

    where :math:`\\hat{\\mathbf{u}} = \\text{DPD}(\\mathbf{y}/G)` is the
    postdistorter output. This computes an incremental correction.

    *EMA method*:

    .. math::

        \\mathbf{c}_{new} = (1-\\mu) \\mathbf{c} + \\mu \\cdot \\text{LS}(\\mathbf{Y}, \\mathbf{u})

    This directly averages between old and new LS solutions.

    **Typical Convergence:**

    LS-DPD typically converges in 2-4 iterations. More iterations may be
    needed for highly nonlinear PAs or when starting far from optimal.

    **miscellaneous:**

    - ``estimate_pa_gain()`` must be called before ``perform_ls_learning()``
    - After ``perform_ls_learning()``, DPD coefficients are optimized
    - Coefficient history is stored in ``dpd.coeff_history``

    Example
    -------
    >>> config = Config()
    >>> system = LS_DPDSystem(training=True, config=config)
    >>> system.estimate_pa_gain()
    >>> results = system.perform_ls_learning(batch_size=16, verbose=True)
    >>> print(f"Final coefficients shape: {results['coeffs'].shape}")

    See Also
    --------
    NN_DPDSystem : Neural network-based DPD system.
    LeastSquaresDPD : The underlying predistorter layer.
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

        # Instantiate the Memory Polynomial predistorter
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
        Forward signal through predistorter and PA.

        Parameters
        ----------
        x : tf.Tensor
            Input signal at PA rate, shape ``[batch, num_samples]``.

        Returns
        -------
        dict
            Signal path outputs:

            - ``u`` : Predistorted signal
            - ``u_norm`` : Same as ``u`` (no normalization for LS)
            - ``y_comp`` : Gain-compensated PA output
            - ``x_scale`` : Always 1.0 (no scaling for LS)
        """
        # Apply predistorter (no input normalization needed for LS).
        u = self._dpd(x, training=False)
        u_norm = u  # No normalization difference for LS.
        x_scale = tf.constant(1.0, dtype=tf.float32)

        # Pass predistorted signal through PA.
        y = self._pa(u)

        # Divide by PA gain to isolate nonlinear distortion.
        # This makes postdistorter learn only the inverse nonlinearity.
        y_comp = y / tf.cast(self._pa_gain, y.dtype)

        return {
            "u": u,
            "u_norm": u_norm,
            "y_comp": y_comp,
            "x_scale": x_scale,
        }

    def _training_forward(self, x):
        """
        Training forward pass - not used for LS-DPD.

        LS-DPD uses closed-form coefficient estimation via
        ``perform_ls_learning()``. This method exists only
        to satisfy the base class interface.

        Parameters
        ----------
        x : tf.Tensor
            Input signal (unused).

        Raises
        ------
        ValueError
            Always raised - use ``perform_ls_learning()`` instead.
        """
        raise ValueError(
            "_training_forward() is for NN-DPD only. "
            "Use perform_ls_learning() for LS-DPD."
        )

    def _inference_forward(self, x):
        """
        Run inference to compare PA output with and without DPD.

        Parameters
        ----------
        x : tf.Tensor
            Input signal at PA rate, shape ``[batch, num_samples]``.

        Returns
        -------
        dict
            Inference outputs:

            - ``pa_input`` : Original input signal
            - ``pa_output_no_dpd`` : PA output without predistortion
            - ``pa_output_with_dpd`` : PA output with predistortion
            - ``predistorted`` : DPD output (before PA)
        """
        # Baseline: PA output without any predistortion.
        pa_output_no_dpd = self._pa(x)

        # Apply predistorter then PA.
        x_predistorted = self._dpd(x, training=False)
        pa_output_with_dpd = self._pa(x_predistorted)

        return {
            "pa_input": x,
            "pa_output_no_dpd": pa_output_no_dpd,
            "pa_output_with_dpd": pa_output_with_dpd,
            "predistorted": x_predistorted,
        }

    def _ls_training_iteration(self, x):
        """
        Execute one iteration of indirect learning coefficient update.

        Implements the complete indirect learning loop:

        1. Apply current predistorter: ``u = DPD(x)``
        2. Pass through PA: ``y = PA(u)``
        3. Normalize by gain: ``y_comp = y / G``
        4. Build basis matrix from ``y_comp``
        5. Update coefficients via Newton or EMA method

        Parameters
        ----------
        x : tf.Tensor
            Input signal at PA rate, shape ``[batch, num_samples]``.

        Returns
        -------
        dict
            Iteration results:

            - ``y_power`` : PA output power in dB (for monitoring convergence)

        Notes
        -----
        **Newton Method:**

        Computes the error between predistorter output ``u`` and postdistorter
        output ``u_hat = DPD(y_comp)``, then finds the coefficient correction
        that minimizes this error in a least-squares sense.

        **EMA Method:**

        Directly computes the optimal coefficients for the current data,
        then blends with previous coefficients using exponential moving average.
        """
        # Forward through predistorter and PA (steps 1-3).
        signals = self._forward_signal_path(x)
        u = signals["u"]
        y_comp = signals["y_comp"]

        # Flatten for LS operations (basis matrix expects 1D input).
        u_flat = tf.reshape(u, [-1])
        y_flat = tf.reshape(y_comp, [-1])

        # Build polynomial basis matrix from gain-compensated PA output.
        # Each column is a basis function evaluated on y_flat.
        Y = self._dpd.setup_basis_matrix(y_flat)

        current_coeffs = self._dpd.coeffs

        # Coefficient update depends on learning method.
        if self._dpd._learning_method == "newton":
            # Newton: compute error and find correction.
            # u_hat is what the postdistorter produces from PA output.
            u_hat = self._dpd.predistort(y_flat)
            # Error: how far postdistorter output is from predistorter output.
            error = u_flat - u_hat
            # LS solution gives correction to reduce this error.
            new_coeffs = (
                current_coeffs
                + self._dpd._learning_rate * self._dpd._ls_estimation(Y, error)
            )
        else:
            # EMA: blend old coefficients with new LS solution.
            # New LS solution minimizes ||Y @ c - u||^2.
            new_coeffs = (
                1 - self._dpd._learning_rate
            ) * current_coeffs + self._dpd._learning_rate * self._dpd._ls_estimation(
                Y, u_flat
            )

        # Update predistorter coefficients.
        self._dpd.coeffs = new_coeffs

        # Return PA output power for convergence monitoring.
        y_power = 10 * tf.experimental.numpy.log10(
            tf.reduce_mean(tf.abs(y_flat) ** 2) + 1e-12
        )
        return {"y_power": float(y_power.numpy())}

    def perform_ls_learning(self, batch_size, nIterations=None, verbose=False):
        """
        Train the LS-DPD predistorter using indirect learning.

        Generates a batch of OFDM signals and iteratively refines the
        predistorter coefficients using closed-form least-squares updates.

        Parameters
        ----------
        batch_size : int
            Number of OFDM frames to generate for training. Larger batches
            give more stable coefficient estimates but require more memory.
        nIterations : int, optional
            Number of indirect learning iterations. If None, uses the
            value specified at construction. Default: None.
        verbose : bool, optional
            If True, print progress information. Default: False.

        Returns
        -------
        dict
            Training results:

            - ``coeffs`` : np.ndarray
                Final optimized coefficients, shape ``[n_coeffs, 1]``.
            - ``coeff_history`` : np.ndarray
                Coefficient values at each iteration, shape
                ``[n_coeffs, n_iterations+1]``. First column is initial
                (identity) coefficients.

        Notes
        -----
        **Convergence Monitoring:**

        The ``y_power`` printed in verbose mode should stabilize as training
        progresses. Increasing power may indicate instability (try reducing
        ``ls_learning_rate``).

        **Pre-conditions:**

        - ``estimate_pa_gain()`` must be called first
        - System must be in training mode

        **Post-conditions:**

        - DPD coefficients are updated in place
        - Coefficient history is stored in ``self.dpd.coeff_history``

        Example
        -------
        >>> system = LS_DPDSystem(training=True, config=config)
        >>> system.estimate_pa_gain()
        >>> results = system.perform_ls_learning(
        ...     batch_size=16,
        ...     nIterations=5,
        ...     verbose=True
        ... )
        Starting LS-DPD learning: 5 iterations, order=7, memory=4
          Iteration 1/5: PA output power = -12.34 dB
          ...
        LS-DPD learning complete.
        """
        # Generate training signal (same signal used for all iterations).
        x = self.generate_signal(batch_size)

        # Use constructor value if not overridden.
        n_iters = nIterations if nIterations is not None else self._dpd._nIterations

        # Initialize history with identity coefficients (before any training).
        coeff_history = self._dpd.coeffs.numpy().copy()

        if verbose:
            print(
                f"Starting LS-DPD learning: {n_iters} iterations, "
                f"order={self._dpd._order}, memory={self._dpd._memory_depth}"
            )

        # Iterative coefficient refinement.
        for iteration in range(n_iters):
            result = self._ls_training_iteration(x)

            # Append current coefficients to history.
            coeff_history = np.hstack([coeff_history, self._dpd.coeffs.numpy()])

            if verbose:
                print(
                    f"  Iteration {iteration + 1}/{n_iters}: "
                    f"PA output power = {result['y_power']:.2f} dB"
                )

        if verbose:
            print("LS-DPD learning complete.")

        # Store history in DPD layer for external access (e.g., plotting).
        self._dpd.coeff_history = coeff_history

        return {
            "coeffs": self._dpd.coeffs.numpy(),
            "coeff_history": coeff_history,
        }
