"""Batched Least-Squares DPD using Indirect Learning Architecture."""

import numpy as np
import tensorflow as tf


class LeastSquaresDPD(tf.keras.layers.Layer):
    """
    Least-Squares Digital Pre-Distortion using Indirect Learning Architecture.

    Inherits from tf.keras.layers.Layer for Sionna compatibility and differentiability.
    Supports batched input [B, num_samples] for Sionna-style processing.
    Uses the GMP (Generalized Memory Polynomial) model.

    The predistort() method is fully differentiable. For LS-based learning,
    use DPDSystem.perform_ls_learning() which orchestrates the training loop.

    Args:
        order: Polynomial order (must be odd, default: 7)
        memory_depth: Memory depth in samples (default: 4)
        lag_depth: Lag/lead depth for cross-terms (default: 0)
        nIterations: Number of learning iterations (default: 3)
        learning_rate: Learning rate (default: 0.75)
        learning_method: 'newton' or 'ema' (default: 'newton')
        use_even: Include even-order terms (default: False)
        use_conj: Include conjugate terms (default: False)
        use_dc_term: Include DC term (default: False)
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

        if self._use_even:
            assert (
                self._lag_depth == 0
            ), "GMP not yet supported for even terms. Set lag_depth=0"

        self._n_coeffs = self._compute_n_coeffs()
        # coeff_history is managed by DPDSystem.perform_ls_learning()
        self.coeff_history = None

    def build(self, input_shape):
        """Build layer - create trainable weights."""
        init_real = np.zeros((self._n_coeffs, 1), dtype=np.float32)
        init_real[0, 0] = 1.0
        init_imag = np.zeros((self._n_coeffs, 1), dtype=np.float32)

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
        return self._n_coeffs

    @property
    def coeffs(self):
        """Return complex coefficients from real/imag parts."""
        if not self.built:
            raise RuntimeError(
                "Layer not built. Call the layer on input first, or call build()."
            )
        return tf.complex(self._coeffs_real, self._coeffs_imag)

    @coeffs.setter
    def coeffs(self, value):
        """Set coefficients from complex tensor."""
        if not self.built:
            raise RuntimeError(
                "Layer not built. Call the layer on input first, or call build()."
            )
        self._coeffs_real.assign(tf.math.real(value))
        self._coeffs_imag.assign(tf.math.imag(value))

    def _compute_n_coeffs(self):
        """Compute total number of DPD coefficients."""
        n_order = self._order if self._use_even else (self._order + 1) // 2
        n = n_order * self._memory_depth
        if not self._use_even:
            n += 2 * (n_order - 1) * self._memory_depth * self._lag_depth
        if self._use_conj:
            n *= 2
        if self._use_dc_term:
            n += 1
        return n

    def _delay_signal(self, signal, delay):
        """Apply delay to signal by prepending zeros."""
        if delay == 0:
            return signal
        padding = tf.zeros(delay, dtype=signal.dtype)
        return tf.concat([padding, signal[:-delay]], axis=0)

    def _add_memory_columns(self, columns, branch):
        """Add delayed versions of branch for all memory depths."""
        for delay in range(self._memory_depth):
            columns.append(self._delay_signal(branch, delay))

    def setup_basis_matrix(self, x):
        """
        Build GMP basis matrix for 1D input. Fully differentiable.

        Args:
            x: [num_samples] complex tensor
        Returns:
            [num_samples, n_coeffs] complex tensor
        """
        x = tf.cast(tf.reshape(x, [-1]), tf.complex64)
        n_samples = tf.shape(x)[0]
        abs_x = tf.abs(x)
        step = 1 if self._use_even else 2
        columns = []

        # Main memory polynomial branch
        for order in range(1, self._order + 1, step):
            branch = x * tf.cast(tf.pow(abs_x, order - 1), tf.complex64)
            self._add_memory_columns(columns, branch)

        # Lagging cross-terms
        for order in range(3, self._order + 1, step):
            abs_base = tf.pow(abs_x, order - 1)
            for lag in range(1, self._lag_depth + 1):
                lagged_abs = tf.concat(
                    [tf.zeros(lag, dtype=tf.float32), abs_base[:-lag]], axis=0
                )
                branch = x * tf.cast(lagged_abs, tf.complex64)
                self._add_memory_columns(columns, branch)

        # Leading cross-terms
        for order in range(3, self._order + 1, step):
            abs_base = tf.pow(abs_x, order - 1)
            for lead in range(1, self._lag_depth + 1):
                lead_abs = tf.concat(
                    [abs_base[lead:], tf.zeros(lead, dtype=tf.float32)], axis=0
                )
                branch = x * tf.cast(lead_abs, tf.complex64)
                self._add_memory_columns(columns, branch)

        # Conjugate branch
        if self._use_conj:
            for order in range(1, self._order + 1, step):
                branch = tf.math.conj(x) * tf.cast(
                    tf.pow(abs_x, order - 1), tf.complex64
                )
                self._add_memory_columns(columns, branch)

        # DC term
        if self._use_dc_term:
            columns.append(tf.ones(n_samples, dtype=tf.complex64))

        return tf.stack(columns, axis=1)

    def predistort(self, x):
        """
        Apply predistortion to input signal. Fully differentiable.

        Args:
            x: [num_samples] or [B, num_samples] tensor
        Returns:
            Same shape as input - predistorted signal
        """
        if not self.built:
            self.build(x.shape)

        input_shape, input_ndims = tf.shape(x), len(x.shape)
        coeffs = self.coeffs

        if input_ndims == 1:
            X = self.setup_basis_matrix(x)
            return tf.reshape(tf.linalg.matmul(X, coeffs), [-1])
        elif input_ndims == 2:
            batch_size, samples_per_batch = input_shape[0], input_shape[1]
            X = self.setup_basis_matrix(tf.reshape(x, [-1]))
            y_flat = tf.reshape(tf.linalg.matmul(X, coeffs), [-1])
            return tf.reshape(y_flat, [batch_size, samples_per_batch])
        else:
            raise ValueError(f"Input must be 1D or 2D, got shape {x.shape}")

    def call(self, x, training=None):
        """Keras layer call - applies predistortion."""
        return self.predistort(x)

    def _ls_estimation(self, X, y):
        """Regularized least-squares estimation."""
        start = self._memory_depth + self._lag_depth - 1
        end = -self._lag_depth if self._lag_depth > 0 else None
        X_slice, y_slice = X[start:end], tf.reshape(y[start:end], [-1, 1])

        lam = tf.constant(0.001, dtype=tf.float32)
        XH = tf.linalg.adjoint(X_slice)
        XHX = tf.linalg.matmul(XH, X_slice)
        reg = tf.cast(lam * tf.eye(tf.shape(XHX)[0]), dtype=tf.complex64)
        return tf.linalg.solve(XHX + reg, tf.linalg.matmul(XH, y_slice))
