"""Batched memory polynomial power amplifier model for Sionna pipeline."""

import tensorflow as tf


class PowerAmplifier(tf.keras.layers.Layer):
    """
    Memory polynomial power amplifier model as a differentiable Keras layer.

    Supports batched input [B, num_samples] for Sionna-style processing.
    Uses the same memory polynomial model as the original PowerAmplifier.

    The PA model is: y[n] = sum_{k=1,3,5,...}^{order} sum_{m=0}^{memory_depth-1}
                            a_{k,m} * x[n-m] * |x[n-m]|^{k-1}

    Args:
        order: Polynomial order (must be odd, default: 7)
        memory_depth: Memory depth in samples (default: 4)
        coefficients: Optional custom coefficients [n_coeffs, memory_depth]
                     If None, uses default WARP board coefficients
    """

    # Default coefficients derived from a WARP board
    DEFAULT_COEFFS = tf.constant(
        [
            [
                0.9295 - 0.0001j,
                0.2939 + 0.0005j,
                -0.1270 + 0.0034j,
                0.0741 - 0.0018j,
            ],  # 1st order
            [
                0.1419 - 0.0008j,
                -0.0735 + 0.0833j,
                -0.0535 + 0.0004j,
                0.0908 - 0.0473j,
            ],  # 3rd order
            [
                0.0084 - 0.0569j,
                -0.4610 + 0.0274j,
                -0.3011 - 0.1403j,
                -0.0623 - 0.0269j,
            ],  # 5th order
            [
                0.1774 + 0.0265j,
                0.0848 + 0.0613j,
                -0.0362 - 0.0307j,
                0.0415 + 0.0429j,
            ],  # 7th order
        ],
        dtype=tf.complex64,
    )

    def __init__(
        self, order: int = 7, memory_depth: int = 4, coefficients=None, **kwargs
    ):
        super().__init__(**kwargs)

        if order % 2 == 0:
            raise ValueError("Order must be odd.")

        self._order = order
        self._memory_depth = memory_depth
        self._n_coeffs = (order + 1) // 2

        # Use provided coefficients or prune defaults
        if coefficients is not None:
            self._poly_coeffs = tf.cast(coefficients, tf.complex64)
        else:
            self._poly_coeffs = self.DEFAULT_COEFFS[: self._n_coeffs, :memory_depth]

    def call(self, x):
        """
        Apply PA model to input signal.

        Args:
            x: [..., num_samples] complex tensor (supports batched input)

        Returns:
            [..., num_samples] complex tensor - PA output
        """

        # Build basis matrix for batched input
        X = self._setup_basis_matrix(x)

        # Flatten coefficients for matmul:
        # [n_coeffs, memory_depth] -> [n_coeffs * memory_depth, 1]
        coeffs = tf.reshape(tf.transpose(self._poly_coeffs), [-1, 1])

        # Apply PA model: X @ coeffs
        # X shape: [..., num_samples, n_coeffs * memory_depth]
        # coeffs shape: [n_coeffs * memory_depth, 1]
        pa_output = tf.linalg.matmul(X, coeffs)

        # Remove last dimension: [..., num_samples, 1] -> [..., num_samples]
        pa_output = tf.squeeze(pa_output, axis=-1)

        return pa_output

    def _setup_basis_matrix(self, x):
        """
        Build memory polynomial basis matrix for batched input.

        Args:
            x: [..., num_samples] complex tensor

        Returns:
            [..., num_samples, n_coeffs * memory_depth] complex tensor
        """
        x = tf.cast(x, tf.complex64)
        abs_x = tf.abs(x)  # [..., num_samples], float32

        columns = []

        for order in range(1, self._order + 1, 2):
            # x * |x|^(order-1)
            abs_power = tf.pow(abs_x, order - 1)  # float32
            branch = x * tf.cast(abs_power, tf.complex64)  # [..., num_samples]

            for delay in range(self._memory_depth):
                if delay == 0:
                    delayed = branch
                else:
                    # Pad with zeros at the beginning, shift right
                    # Use tf.pad for batched operation
                    paddings = [[0, 0]] * (len(branch.shape) - 1) + [[delay, 0]]
                    delayed = tf.pad(branch[..., :-delay], paddings)

                columns.append(delayed)

        # Stack columns: list of [..., num_samples] -> [..., num_samples, n_cols]
        X = tf.stack(columns, axis=-1)
        return X
