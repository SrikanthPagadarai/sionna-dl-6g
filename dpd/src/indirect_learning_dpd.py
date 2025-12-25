"""Batched Indirect Learning Architecture DPD for Sionna pipeline."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class IndirectLearningDPD(tf.keras.layers.Layer):
    """
    Indirect Learning Architecture Digital Pre-Distortion for batched signals.

    Inherits from tf.keras.layers.Layer for Sionna compatibility and differentiability.
    Supports batched input [B, num_samples] for Sionna-style processing.
    Uses the same GMP (Generalized Memory Polynomial) model as the original.

    The predistort() method is fully differentiable and can be used in end-to-end
    training pipelines. For LS-based learning, use perform_learning().

    Args:
        order: Polynomial order (must be odd, default: 7)
        memory_depth: Memory depth in samples (default: 3)
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

        self._order = p["order"]
        self._memory_depth = p["memory_depth"]
        self._lag_depth = p["lag_depth"]
        self._nIterations = p["nIterations"]
        self._learning_rate = p["learning_rate"]
        self._learning_method = p["learning_method"]
        self._use_even = p["use_even"]
        self._use_conj = p["use_conj"]
        self._use_dc_term = p["use_dc_term"]

        if self._use_even:
            assert (
                self._lag_depth == 0
            ), "GMP not yet supported for even terms. Set lag_depth=0"

        self._n_coeffs = self._compute_n_coeffs()

        # Will be initialized in build()
        self._coeffs = None
        self.coeff_history = None
        self.result_history = None

    def build(self, input_shape):
        """Build layer - create trainable weights."""
        # Initialize coefficients using add_weight for proper Keras tracking
        # First coefficient is 1, rest are 0
        # We store real and imaginary parts separately for proper gradient flow
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
    def order(self):
        return self._order

    @property
    def memory_depth(self):
        return self._memory_depth

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

    def setup_basis_matrix(self, x):
        """
        Build GMP basis matrix for 1D input.

        This method is fully differentiable.

        Args:
            x: [num_samples] complex tensor

        Returns:
            [num_samples, n_coeffs] complex tensor
        """
        x = tf.reshape(x, [-1])
        x = tf.cast(x, tf.complex64)

        n_samples = tf.shape(x)[0]
        step = 1 if self._use_even else 2
        columns = []
        abs_x = tf.abs(x)

        # Main memory polynomial branch
        for order in range(1, self._order + 1, step):
            abs_power = tf.pow(abs_x, order - 1)
            branch = x * tf.cast(abs_power, tf.complex64)
            for delay in range(self._memory_depth):
                if delay == 0:
                    delayed = branch
                else:
                    padding = tf.zeros(delay, dtype=tf.complex64)
                    delayed = tf.concat([padding, branch[:-delay]], axis=0)
                columns.append(delayed)

        # Lagging cross-terms
        for order in range(3, self._order + 1, step):
            abs_base = tf.pow(abs_x, order - 1)
            for lag in range(1, self._lag_depth + 1):
                lagged_abs = tf.concat(
                    [tf.zeros(lag, dtype=tf.float32), abs_base[:-lag]], axis=0
                )
                branch = x * tf.cast(lagged_abs, tf.complex64)
                for delay in range(self._memory_depth):
                    if delay == 0:
                        delayed = branch
                    else:
                        padding = tf.zeros(delay, dtype=tf.complex64)
                        delayed = tf.concat([padding, branch[:-delay]], axis=0)
                    columns.append(delayed)

        # Leading cross-terms
        for order in range(3, self._order + 1, step):
            abs_base = tf.pow(abs_x, order - 1)
            for lead in range(1, self._lag_depth + 1):
                lead_abs = tf.concat(
                    [abs_base[lead:], tf.zeros(lead, dtype=tf.float32)], axis=0
                )
                branch = x * tf.cast(lead_abs, tf.complex64)
                for delay in range(self._memory_depth):
                    if delay == 0:
                        delayed = branch
                    else:
                        padding = tf.zeros(delay, dtype=tf.complex64)
                        delayed = tf.concat([padding, branch[:-delay]], axis=0)
                    columns.append(delayed)

        # Conjugate branch
        if self._use_conj:
            for order in range(1, self._order + 1, step):
                abs_power = tf.pow(abs_x, order - 1)
                branch = tf.math.conj(x) * tf.cast(abs_power, tf.complex64)
                for delay in range(self._memory_depth):
                    if delay == 0:
                        delayed = branch
                    else:
                        padding = tf.zeros(delay, dtype=tf.complex64)
                        delayed = tf.concat([padding, branch[:-delay]], axis=0)
                    columns.append(delayed)

        # DC term
        if self._use_dc_term:
            columns.append(tf.ones(n_samples, dtype=tf.complex64))

        X = tf.stack(columns, axis=1)
        return X

    def predistort(self, x):
        """
        Apply predistortion to input signal.

        This method is fully differentiable.

        Args:
            x: [num_samples] or [B, num_samples] tensor

        Returns:
            Same shape as input - predistorted signal
        """
        # Ensure layer is built
        if not self.built:
            self.build(x.shape)

        input_shape = tf.shape(x)
        input_ndims = len(x.shape)

        # Get complex coefficients
        coeffs = self.coeffs

        if input_ndims == 1:
            # 1D input
            X = self.setup_basis_matrix(x)
            return tf.reshape(tf.linalg.matmul(X, coeffs), [-1])
        elif input_ndims == 2:
            # Batched input [B, N] - flatten, process, reshape
            batch_size = input_shape[0]
            samples_per_batch = input_shape[1]

            x_flat = tf.reshape(x, [-1])
            X = self.setup_basis_matrix(x_flat)
            y_flat = tf.reshape(tf.linalg.matmul(X, coeffs), [-1])

            return tf.reshape(y_flat, [batch_size, samples_per_batch])
        else:
            raise ValueError(f"Input must be 1D or 2D, got shape {x.shape}")

    def call(self, x, training=None):
        """
        Keras layer call - applies predistortion.

        This makes the layer usable in Keras Sequential/Functional models
        and compatible with Sionna pipelines.

        Args:
            x: Input tensor [num_samples] or [B, num_samples]
            training: Boolean or None. Whether the layer is in training mode.
                     Not used in this layer but included for Keras/Sionna compatibility.

        Returns:
            Predistorted signal with same shape as input
        """
        return self.predistort(x)

    def _ls_estimation(self, X, y):
        """Regularized least-squares estimation."""
        start = self._memory_depth + self._lag_depth - 1
        end = -self._lag_depth if self._lag_depth > 0 else None

        if end is None:
            X_slice = X[start:]
            y_slice = y[start:]
        else:
            X_slice = X[start:end]
            y_slice = y[start:end]

        y_slice = tf.reshape(y_slice, [-1, 1])

        lam = tf.constant(0.001, dtype=tf.float32)
        XH = tf.linalg.adjoint(X_slice)
        XHX = tf.linalg.matmul(XH, X_slice)
        reg = tf.cast(lam * tf.eye(tf.shape(XHX)[0]), dtype=tf.complex64)
        XHy = tf.linalg.matmul(XH, y_slice)

        beta = tf.linalg.solve(XHX + reg, XHy)
        return beta

    def perform_learning(self, x, pa, verbose=True):
        """
        Perform iterative DPD learning using indirect learning architecture.

        This method uses LS estimation to learn DPD coefficients.
        For batched input, signal is flattened for learning to ensure
        consistent coefficient estimation across all batches.

        Args:
            x: [num_samples] or [B, num_samples] input tensor (at PA sample rate)
            pa: PowerAmplifier instance
            verbose: Print progress (default: True)

        Returns:
            Dictionary with learning results
        """
        # Ensure layer is built
        if not self.built:
            self.build(x.shape)

        # Flatten if batched for consistent learning
        input_ndims = len(x.shape)
        input_shape = tf.shape(x)

        if input_ndims == 2:
            batch_size = input_shape[0]
            samples_per_batch = input_shape[1]
            x_flat = tf.reshape(x, [-1])
        else:
            batch_size = 1
            x_flat = x

        self.coeff_history = self.coeffs.numpy().copy()
        self.result_history = []

        if verbose:
            print(
                f"Starting DPD learning: {self._nIterations} iterations, "
                f"order={self._order}, memory={self._memory_depth}"
            )

        for iteration in range(self._nIterations):
            # Apply current predistortion
            u = self.predistort(x_flat)

            # Pass through PA (need to handle batched PA)
            if input_ndims == 2:
                u_batched = tf.reshape(u, [batch_size, samples_per_batch])
                y_batched = pa(u_batched)
                y = tf.reshape(y_batched, [-1])
            else:
                y = pa(tf.expand_dims(u, 0))
                y = tf.reshape(y, [-1])

            # Build basis matrix from PA output
            Y = self.setup_basis_matrix(y)

            # Update coefficients based on learning method
            current_coeffs = self.coeffs
            if self._learning_method == "newton":
                error = u - self.predistort(y)
                update = self._ls_estimation(Y, error)
                new_coeffs = current_coeffs + self._learning_rate * update
            else:  # 'ema'
                update = self._ls_estimation(Y, u)
                new_coeffs = (
                    1 - self._learning_rate
                ) * current_coeffs + self._learning_rate * update

            # Update coefficients using the setter
            self.coeffs = new_coeffs

            # Record history
            self.coeff_history = np.hstack([self.coeff_history, self.coeffs.numpy()])

            if verbose:
                # Compute power metrics
                y_power = 10 * np.log10(np.mean(np.abs(y.numpy()) ** 2) + 1e-12)
                print(
                    f"  Iteration {iteration + 1}/{self._nIterations}: "
                    f"PA output power = {y_power:.2f} dB"
                )

        if verbose:
            print("DPD learning complete.")

        return {
            "coeffs": self.coeffs.numpy(),
            "coeff_history": self.coeff_history,
        }

    def plot_coeff_history(self, save_path=None):
        """Plot coefficient learning history."""
        if self.coeff_history is None:
            print("No learning history available. Run perform_learning() first.")
            return

        iterations = np.arange(self.coeff_history.shape[1])

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, np.abs(self.coeff_history.T))
        plt.title("DPD Coefficient Learning History")
        plt.xlabel("Iteration")
        plt.ylabel("|coeffs|")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "order": self._order,
                "memory_depth": self._memory_depth,
                "lag_depth": self._lag_depth,
                "nIterations": self._nIterations,
                "learning_rate": self._learning_rate,
                "learning_method": self._learning_method,
                "use_even": self._use_even,
                "use_conj": self._use_conj,
                "use_dc_term": self._use_dc_term,
            }
        )
        return config
