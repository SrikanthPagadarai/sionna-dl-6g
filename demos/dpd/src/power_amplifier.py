# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Memory polynomial power amplifier model for DPD system simulation.

Implements a PA behavioral model based on measured coefficients
from a WARP (Wireless Open-Access Research Platform) board. The memory
polynomial structure captures both static nonlinearity (AM/AM, AM/PM)
and dynamic memory effects that arise from thermal and electrical time
constants.

The model is implemented as a differentiable Keras layer, enabling gradient-based
DPD training where the PA acts as a fixed (non-trainable) component in the
signal path.
"""

import tensorflow as tf


class PowerAmplifier(tf.keras.layers.Layer):
    """
    Memory polynomial power amplifier model.

    Implements a memory polynomial (MP) PA model with fixed coefficients
    derived from WARP board measurements. The model captures amplitude-
    dependent gain compression (AM/AM) and phase distortion (AM/PM),
    as well as memory effects from previous samples.

    The PA transfer function is:

    .. math::

        y[n] = \\sum_{k \\in \\{1,3,5,7\\}} \\sum_{m=0}^{3} a_{k,m} \\cdot x[n-m] \\cdot |x[n-m]|^{k-1}

    where :math:`a_{k,m}` are complex coefficients, :math:`k` is the polynomial
    order (odd-only to model symmetric nonlinearity), and :math:`m` is the
    memory tap index.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed to the Keras Layer base class.

    Attributes
    ----------
    order : int
        Polynomial order (fixed at 7). Only odd orders are used because
        PA nonlinearity is predominantly odd-symmetric around zero.
    memory_depth : int
        Memory depth in samples (fixed at 4). Captures short-term memory
        effects from bias network and thermal dynamics.

    Notes
    -----
    **miscellaneous:**

    - Input signal should be normalized to appropriate power level
      (the model expects inputs with RMS around 0.1-1.0)
    - Output has same shape as input
    - Output exhibits AM/AM compression and AM/PM distortion

    Example
    -------
    >>> pa = PowerAmplifier()
    >>> x = tf.complex(tf.random.normal([16, 1024]), tf.random.normal([16, 1024]))
    >>> x = x * 0.3  # Scale to reasonable PA input level
    >>> y = pa(x)
    >>> y.shape
    TensorShape([16, 1024])
    """

    # =========================================================================
    # Default coefficients from WARP board characterization
    # =========================================================================
    # These coefficients were extracted using from a WARP v3 board PA.
    # The structure is [order_index, memory_tap] where order_index maps
    # to polynomial orders {1, 3, 5, 7}.
    #
    # Physical interpretation:
    # - Row 0 (1st order): Linear gain and memory (dominates at low power)
    # - Row 1, 2, 3 (3rd, 5th, 7th order): Cubic and higher-order compression
    _DEFAULT_COEFFS = tf.constant(
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._order = 7
        self._memory_depth = 4
        # Number of polynomial terms: orders {1,3,5,7} = 4 terms
        self._n_coeffs = 4  # (order + 1) // 2
        self._poly_coeffs = self._DEFAULT_COEFFS

    @property
    def order(self) -> int:
        """int: Polynomial order (fixed at 7, odd-only)."""
        return self._order

    @property
    def memory_depth(self) -> int:
        """int: Memory depth in samples (fixed at 4)."""
        return self._memory_depth

    def call(self, x):
        """
        Apply PA distortion to input signal.

        Parameters
        ----------
        x : tf.Tensor
            Input signal with shape ``[..., num_samples]`` and complex dtype.
            Supports arbitrary batch dimensions.

        Returns
        -------
        tf.Tensor
            Distorted output signal with same shape as input.

        Notes
        -----
        The computation is structured as a matrix-vector product for efficiency:
        ``y = X @ coeffs`` where X is the basis matrix containing all polynomial
        and memory terms, and coeffs is the flattened coefficient vector.
        """
        # Build basis matrix containing all x[n-m] * |x[n-m]|^(k-1) terms
        X = self._setup_basis_matrix(x)

        # Reshape coefficients for batched matmul:
        # [n_coeffs, memory_depth] -> [n_coeffs * memory_depth, 1]
        # Transpose before flatten to get memory-major ordering matching basis matrix
        coeffs = tf.reshape(tf.transpose(self._poly_coeffs), [-1, 1])

        # Apply PA model via matrix multiplication
        # X: [..., num_samples, n_coeffs * memory_depth]
        # coeffs: [n_coeffs * memory_depth, 1]
        # result: [..., num_samples, 1]
        pa_output = tf.linalg.matmul(X, coeffs)

        # Remove trailing dimension: [..., num_samples, 1] -> [..., num_samples]
        pa_output = tf.squeeze(pa_output, axis=-1)

        return pa_output

    def _setup_basis_matrix(self, x):
        """
        Construct the memory polynomial basis matrix.

        Parameters
        ----------
        x : tf.Tensor
            Input signal with shape ``[..., num_samples]``.

        Returns
        -------
        tf.Tensor
            Basis matrix with shape ``[..., num_samples, n_coeffs * memory_depth]``.
            Each column corresponds to one coefficient in the model.

        Notes
        -----
        The basis matrix columns are ordered as:
        ``[x|x|^0, x_{-1}|x_{-1}|^0, ..., x|x|^2, x_{-1}|x_{-1}|^2, ...]``

        where ``x_{-m}`` denotes ``x[n-m]`` (delayed by m samples).

        Delayed samples at the signal start are zero-padded.
        """
        x = tf.cast(x, tf.complex64)
        abs_x = tf.abs(x)  # [..., num_samples], float32

        columns = []

        # Iterate over odd polynomial orders: 1, 3, 5, 7
        for order in range(1, self._order + 1, 2):
            # Compute x * |x|^(order-1) for the nonlinear branch
            # order=1: x (linear)
            # order=3: x * |x|^2 (cubic)
            # order=5: x * |x|^4 (quintic)
            # order=7: x * |x|^6 (septic)
            abs_power = tf.pow(abs_x, order - 1)  # float32
            branch = x * tf.cast(abs_power, tf.complex64)  # [..., num_samples]

            # Add delayed versions for memory taps
            for delay in range(self._memory_depth):
                if delay == 0:
                    # No delay - use signal as-is
                    delayed = branch
                else:
                    # Shift signal right by 'delay' samples, zero-pad at start.
                    # This models causal memory: output depends on past inputs.
                    paddings = [[0, 0]] * (len(branch.shape) - 1) + [[delay, 0]]
                    delayed = tf.pad(branch[..., :-delay], paddings)

                columns.append(delayed)

        # Stack all basis columns: [..., num_samples, n_cols]
        X = tf.stack(columns, axis=-1)
        return X

    def estimate_gain(self, num_samples=10000):
        """
        Estimate small-signal (linear) gain of the PA.

        Uses a low-amplitude test signal to measure gain in the linear region
        where higher-order polynomial terms are negligible. This gain value
        is needed for proper normalization in the indirect learning DPD
        architecture.

        Parameters
        ----------
        num_samples : int
            Number of random samples for gain estimation. More samples
            reduce variance. Defaults to 10000.

        Returns
        -------
        tf.Tensor
            Estimated voltage gain (linear scale, not dB). Typically close
            to the magnitude of the first-order, zero-memory coefficient
            (~0.93 for default coefficients).

        Notes
        -----
        **Why estimate gain?**

        In indirect learning DPD, the PA output is divided by the gain G
        before being fed to the postdistorter. Without this normalization,
        the postdistorter would learn to invert both the PA nonlinearity
        AND its linear gain, which is undesirable.

        **Why low amplitude?**

        At low input amplitudes, the PA operates linearly and higher-order
        terms (|x|^2, |x|^4, etc.) become negligible. The measured gain
        then reflects only the first-order (linear) coefficient.
        """
        # Generate low-amplitude test signal (stddev=0.1 keeps PA in linear region)
        test_input = tf.complex(
            tf.random.normal([num_samples], stddev=0.1),
            tf.random.normal([num_samples], stddev=0.1),
        )

        # Pass through PA
        test_output = self(test_input)

        # Compute gain as sqrt(output_power / input_power)
        # This is the RMS voltage gain
        input_power = tf.reduce_mean(tf.abs(test_input) ** 2)
        output_power = tf.reduce_mean(tf.abs(test_output) ** 2)
        # Small epsilon prevents division by zero if input is all zeros
        gain = tf.sqrt(output_power / (input_power + 1e-12))

        return gain
