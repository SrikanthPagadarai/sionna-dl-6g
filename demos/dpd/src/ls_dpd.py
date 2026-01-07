# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Least-squares digital predistortion using Memory Polynomial model.

Implements the conventional DPD approach where the predistorter
coefficients are computed via least-squares approach.

The default model is a **Memory Polynomial (MP)**: multiple parallel
branches, each consisting of a static nonlinearity followed by an FIR filter,
with outputs summed together. When ``lag_depth > 0``, the model extends to
a **Generalized Memory Polynomial (GMP)** which adds cross-terms between
the signal and lagged/leading envelope.

The least-squares approach has several advantages:

- Closed-form solution (no iterative optimization)
- Guaranteed convergence to optimal coefficients (for the chosen model)
- Lower computational cost for small models
- Interpretable coefficients that relate to PA physics

However, it is limited to the expressiveness of the polynomial model,
whereas neural networks can learn arbitrary nonlinear mappings.

The implementation follows the indirect learning architecture where
the postdistorter coefficients are computed from PA input/output data,
then copied to the predistorter.
"""

import numpy as np
import tensorflow as tf


class LeastSquaresDPD(tf.keras.layers.Layer):
    """
    Memory Polynomial predistorter with least-squares training.

    Implements a polynomial-based predistorter where the output is a weighted
    sum of nonlinear basis functions of the input signal. The default model
    is a Memory Polynomial (MP). Setting ``lag_depth > 0`` extends this to a
    Generalized Memory Polynomial (GMP).

    **Memory Polynomial:**

    .. math::

        y[n] = \\sum_{k} \\sum_{m} a_{k,m} \\cdot x[n-m] \\cdot |x[n-m]|^{k-1}

    The structure contains parallel branches, each with a static nonlinearity
    ``x * |x|^(k-1)`` followed by an FIR filter (the memory taps).

    **Generalized Memory Polynomial (lag_depth > 0):**

    Adds cross-terms where the envelope is sampled at different time instants:

    .. math::

        y[n] = \\text{MP terms} + \\sum_{k} \\sum_{m} \\sum_{l} b_{k,m,l} \\cdot x[n-m] \\cdot |x[n-m-l]|^{k-1}

    Parameters
    ----------
    params : dict, optional
        Configuration dictionary. Supported keys:

        - ``order`` : int
            Maximum polynomial order (must be odd). Default: 7.
        - ``memory_depth`` : int
            Number of memory taps per branch. Default: 4.
        - ``lag_depth`` : int
            Lag/lead depth for GMP cross-terms. Default: 0 (standard MP).
        - ``nIterations`` : int
            Number of indirect learning iterations. Default: 3.
        - ``learning_rate`` : float
            Coefficient update rate (0-1). Default: 0.75.
        - ``learning_method`` : str
            Update method: 'newton' or 'ema'. Default: 'newton'.
        - ``use_even`` : bool
            Include even-order terms. Default: False.
        - ``use_conj`` : bool
            Include conjugate branch (for IQ imbalance). Default: False.
        - ``use_dc_term`` : bool
            Include DC offset term. Default: False.

    **kwargs
        Additional keyword arguments passed to Keras Layer.

    Attributes
    ----------
    n_coeffs : int
        Total number of DPD coefficients.
    coeffs : tf.Tensor
        Complex coefficient vector, shape ``[n_coeffs, 1]``.
    coeff_history : list or None
        History of coefficients across learning iterations (set externally).

    Notes
    -----
    **Why Odd-Order Only (Default)?**

    PA nonlinearity is predominantly odd-symmetric. Even-order terms model
    asymmetric distortion which is typically small. Using odd-only terms
    halves the coefficient count with minimal accuracy loss.

    **Parallel Hammerstein Interpretation:**

    The MP model can be viewed as parallel branches:

    - Branch 1: ``x`` -> FIR filter -> (linear path)
    - Branch 2: ``x * |x|²`` -> FIR filter -> (cubic nonlinearity)
    - Branch 3: ``x * |x|⁴`` -> FIR filter -> (5th order)
    - ...

    Each branch's FIR filter has ``memory_depth`` taps. All branch outputs
    are summed to produce the predistorted signal.

    **When to Use GMP (lag_depth > 0):**

    GMP cross-terms help when the PA exhibits strong interaction between
    the signal and its delayed envelope (e.g., thermal effects with
    time constants comparable to the signal bandwidth). For most PAs,
    standard MP (lag_depth=0) is sufficient.

    **Coefficient Initialization:**

    Coefficients are initialized to identity (first coefficient = 1, rest = 0),
    meaning the initial predistorter is a pass-through. Training then adjusts
    coefficients to learn the PA inverse.

    **miscellaneous:**

    - Input signal should be complex baseband at PA sample rate
    - For batched input, all batches are concatenated for basis matrix
    - Output has same shape as input
    - Predistortion is differentiable w.r.t. coefficients

    Example
    -------
    >>> # Standard Memory Polynomial
    >>> params = {"order": 7, "memory_depth": 4, "lag_depth": 0}
    >>> dpd = LeastSquaresDPD(params)
    >>> x = tf.complex(tf.random.normal([1024]), tf.random.normal([1024]))
    >>> y = dpd(x)  # Apply predistortion
    >>> y.shape
    TensorShape([1024])

    >>> # Generalized Memory Polynomial (with cross-terms)
    >>> params_gmp = {"order": 7, "memory_depth": 4, "lag_depth": 2}
    >>> dpd_gmp = LeastSquaresDPD(params_gmp)
    """

    DEFAULT_PARAMS = {
        "order": 7,
        "memory_depth": 4,
        "lag_depth": 0,
        "nIterations": 3,
        "use_conj": False,
        "use_dc_term": False,
        "learning_rate": 0.75,
        "use_even": False,
        "learning_method": "newton",
    }

    def __init__(self, params=None, **kwargs):
        super().__init__(**kwargs)
        p = {**self.DEFAULT_PARAMS, **(params or {})}

        if p["order"] % 2 == 0:
            raise ValueError("Order of the DPD must be odd.")

        self._order, self._memory_depth, self._lag_depth = (
            p["order"],
            p["memory_depth"],
            p["lag_depth"],
        )
        self._nIterations, self._learning_rate = p["nIterations"], p["learning_rate"]
        self._learning_method = p["learning_method"]
        self._use_even, self._use_conj, self._use_dc_term = (
            p["use_even"],
            p["use_conj"],
            p["use_dc_term"],
        )

        # GMP cross-terms with even orders not implemented.
        if self._use_even:
            assert (
                self._lag_depth == 0
            ), "GMP not supported for even terms. Set lag_depth=0"

        self._n_coeffs = self._compute_n_coeffs()

        # Coefficient history is managed externally by DPDSystem.perform_ls_learning().
        self.coeff_history = None

    def build(self, input_shape):
        """
        Create trainable coefficient weights.

        Initializes coefficients to identity predistorter (pass-through):
        first coefficient is 1+0j, all others are 0+0j.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of input tensor (used by Keras, not directly needed here).
        """
        # Initialize to identity: [1, 0, 0, ..., 0]
        # This makes initial predistorter a pass-through.
        init_real = np.zeros((self._n_coeffs, 1), dtype=np.float32)
        init_real[0, 0] = 1.0  # First coefficient = 1 (linear term, no delay)
        init_imag = np.zeros((self._n_coeffs, 1), dtype=np.float32)

        # Store real and imaginary parts separately for TF compatibility.
        # Complex weights have limited support in some TF/Keras versions.
        self._coeffs_real = self.add_weight(
            name="dpd_coeffs_real",
            shape=(self._n_coeffs, 1),
            initializer=tf.keras.initializers.Constant(init_real),
            trainable=True,
            dtype=tf.float32,
        )
        self._coeffs_imag = self.add_weight(
            name="dpd_coeffs_imag",
            shape=(self._n_coeffs, 1),
            initializer=tf.keras.initializers.Constant(init_imag),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    @property
    def n_coeffs(self):
        """int: Total number of DPD coefficients."""
        return self._n_coeffs

    @property
    def coeffs(self):
        """
        tf.Tensor: Complex coefficient vector, shape ``[n_coeffs, 1]``.

        Raises
        ------
        RuntimeError
            If layer has not been built yet.
        """
        if not self.built:
            raise RuntimeError(
                "Layer not built. Call the layer on input first, or call build()."
            )
        return tf.complex(self._coeffs_real, self._coeffs_imag)

    @coeffs.setter
    def coeffs(self, value):
        """
        Set coefficients from complex tensor.

        Parameters
        ----------
        value : tf.Tensor
            Complex coefficient tensor, shape ``[n_coeffs, 1]``.

        Raises
        ------
        RuntimeError
            If layer has not been built yet.
        """
        if not self.built:
            raise RuntimeError(
                "Layer not built. Call the layer on input first, or call build()."
            )
        self._coeffs_real.assign(tf.math.real(value))
        self._coeffs_imag.assign(tf.math.imag(value))

    def _compute_n_coeffs(self):
        """
        Compute total number of DPD coefficients for current configuration.

        Returns
        -------
        int
            Total coefficient count.

        Notes
        -----
        Coefficient count depends on:

        - Polynomial orders: (order+1)/2 for odd-only, or order for even+odd
        - Memory depth: each order has memory_depth taps
        - Lag depth (GMP only): adds 2 * (n_orders-1) * memory_depth * lag_depth cross-terms
        - Conjugate: doubles coefficient count
        - DC term: adds 1 coefficient
        """
        # Number of polynomial orders used.
        n_order = self._order if self._use_even else (self._order + 1) // 2

        # Main memory polynomial: n_order orders * memory_depth taps.
        n = n_order * self._memory_depth

        # GMP cross-terms (lagging and leading), only for orders >= 3.
        # These are only added when lag_depth > 0.
        if not self._use_even:
            n += 2 * (n_order - 1) * self._memory_depth * self._lag_depth

        # Conjugate branch doubles the count (for IQ imbalance correction).
        if self._use_conj:
            n *= 2

        # DC term adds one coefficient.
        if self._use_dc_term:
            n += 1

        return n

    def _delay_signal(self, signal, delay):
        """
        Apply sample delay to signal by prepending zeros.

        Parameters
        ----------
        signal : tf.Tensor
            1D signal to delay.
        delay : int
            Number of samples to delay.

        Returns
        -------
        tf.Tensor
            Delayed signal (same length, with zeros prepended).
        """
        if delay == 0:
            return signal
        padding = tf.zeros(delay, dtype=signal.dtype)
        return tf.concat([padding, signal[:-delay]], axis=0)

    def _add_memory_columns(self, columns, branch):
        """
        Add delayed versions of a branch signal for all memory depths.

        Parameters
        ----------
        columns : list
            List to append columns to (modified in place).
        branch : tf.Tensor
            Branch signal to add with delays.
        """
        for delay in range(self._memory_depth):
            columns.append(self._delay_signal(branch, delay))

    def setup_basis_matrix(self, x):
        """
        Construct the MP/GMP basis matrix for least-squares fitting.

        Each column of the basis matrix corresponds to one coefficient
        in the model. The predistorter output is ``y = X @ coeffs``.

        Parameters
        ----------
        x : tf.Tensor
            Input signal, shape ``[num_samples]``, complex dtype.

        Returns
        -------
        tf.Tensor
            Basis matrix, shape ``[num_samples, n_coeffs]``, complex dtype.

        Notes
        -----
        Column ordering follows this pattern:

        1. Main MP terms: orders 1,3,5,7 * memory delays 0,1,2,3
        2. Lagging cross-terms (GMP, if lag_depth > 0): orders 3,5,7 * lags * delays
        3. Leading cross-terms (GMP, if lag_depth > 0): orders 3,5,7 * leads * delays
        4. Conjugate terms (if enabled): same structure with conj(x)
        5. DC term (if enabled): column of ones

        This method is fully differentiable, enabling gradient-based
        fine-tuning if desired.
        """
        x = tf.cast(tf.reshape(x, [-1]), tf.complex64)
        n_samples = tf.shape(x)[0]
        abs_x = tf.abs(x)
        step = 1 if self._use_even else 2  # Step 2 for odd-only orders.
        columns = []

        # Main memory polynomial branch: x[n-m] * |x[n-m]|^(k-1)
        # This is the core MP/Parallel Hammerstein structure.
        for order in range(1, self._order + 1, step):
            branch = x * tf.cast(tf.pow(abs_x, order - 1), tf.complex64)
            self._add_memory_columns(columns, branch)

        # Lagging cross-terms (GMP extension): x[n-m] * |x[n-m-l]|^(k-1)
        # Signal multiplied by lagged envelope. Only added if lag_depth > 0.
        for order in range(3, self._order + 1, step):
            abs_base = tf.pow(abs_x, order - 1)
            for lag in range(1, self._lag_depth + 1):
                lagged_abs = tf.concat(
                    [tf.zeros(lag, dtype=tf.float32), abs_base[:-lag]], axis=0
                )
                branch = x * tf.cast(lagged_abs, tf.complex64)
                self._add_memory_columns(columns, branch)

        # Leading cross-terms (GMP extension): x[n-m] * |x[n-m+l]|^(k-1)
        # Signal multiplied by leading envelope. Only added if lag_depth > 0.
        for order in range(3, self._order + 1, step):
            abs_base = tf.pow(abs_x, order - 1)
            for lead in range(1, self._lag_depth + 1):
                lead_abs = tf.concat(
                    [abs_base[lead:], tf.zeros(lead, dtype=tf.float32)], axis=0
                )
                branch = x * tf.cast(lead_abs, tf.complex64)
                self._add_memory_columns(columns, branch)

        # Conjugate branch for IQ imbalance correction.
        # Same structure as main branch but with conj(x) instead of x.
        if self._use_conj:
            for order in range(1, self._order + 1, step):
                branch = tf.math.conj(x) * tf.cast(
                    tf.pow(abs_x, order - 1), tf.complex64
                )
                self._add_memory_columns(columns, branch)

        # DC term: constant offset (rarely needed for DPD).
        if self._use_dc_term:
            columns.append(tf.ones(n_samples, dtype=tf.complex64))

        return tf.stack(columns, axis=1)

    def predistort(self, x):
        """
        Apply predistortion to input signal.

        Computes ``y = X @ coeffs`` where X is the GMP basis matrix.
        This method is fully differentiable.

        Parameters
        ----------
        x : tf.Tensor
            Input signal, shape ``[num_samples]`` or ``[batch, num_samples]``.

        Returns
        -------
        tf.Tensor
            Predistorted signal, same shape as input.

        Raises
        ------
        ValueError
            If input has more than 2 dimensions.

        Notes
        -----
        For batched input, all batches are concatenated before building
        the basis matrix, then the output is reshaped back to batch form.
        This ensures consistent processing across the batch.
        """
        if not self.built:
            self.build(x.shape)

        input_shape, input_ndims = tf.shape(x), len(x.shape)
        coeffs = self.coeffs

        if input_ndims == 1:
            # Single signal: straightforward matrix multiply.
            X = self.setup_basis_matrix(x)
            return tf.reshape(tf.linalg.matmul(X, coeffs), [-1])
        elif input_ndims == 2:
            # Batched input: flatten, process, reshape back.
            batch_size, samples_per_batch = input_shape[0], input_shape[1]
            X = self.setup_basis_matrix(tf.reshape(x, [-1]))
            y_flat = tf.reshape(tf.linalg.matmul(X, coeffs), [-1])
            return tf.reshape(y_flat, [batch_size, samples_per_batch])
        else:
            raise ValueError(f"Input must be 1D or 2D, got shape {x.shape}")

    def call(self, x, training=None):
        """
        Keras layer call interface.

        Parameters
        ----------
        x : tf.Tensor
            Input signal.
        training : bool or None, optional
            Training mode flag (unused, included for Keras compatibility).

        Returns
        -------
        tf.Tensor
            Predistorted signal.
        """
        return self.predistort(x)

    # [ls-estimation-start]
    def _ls_estimation(self, X, y):
        """
        Compute least-squares coefficient estimate.

        Solves the regularized least-squares problem:
        ``min ||X @ coeffs - y||² + lambda*||coeffs||²``

        Parameters
        ----------
        X : tf.Tensor
            Basis matrix, shape ``[num_samples, n_coeffs]``.
        y : tf.Tensor
            Target signal, shape ``[num_samples]``.

        Returns
        -------
        tf.Tensor
            Estimated coefficients, shape ``[n_coeffs, 1]``.

        Notes
        -----
        The first (memory_depth + lag_depth - 1) and last lag_depth samples
        are excluded from the fit to avoid edge effects from the delay
        operations in the basis matrix construction.

        L2 regularization (lambda=1e-3) prevents ill-conditioning when the
        basis matrix has near-collinear columns.
        """
        # Exclude edge samples affected by delays.
        start = self._memory_depth + self._lag_depth - 1
        end = -self._lag_depth if self._lag_depth > 0 else None
        X_slice, y_slice = X[start:end], tf.reshape(y[start:end], [-1, 1])

        # Regularized least-squares via TensorFlow's lstsq.
        return tf.linalg.lstsq(X_slice, y_slice, l2_regularizer=1e-3)

    # [ls-estimation-end]
