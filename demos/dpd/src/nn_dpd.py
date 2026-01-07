# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Neural network digital predistortion using fully-connected feedforward architecture.

Implements a neural network-based DPD that learns the PA inverse
through gradient-based optimization. Unlike the conventional LS-DPD,
a neural network-based DPD can learn arbitrary nonlinear mappings, potentially
capturing PA behaviors that don't fit within the memory polynomial model.

The architecture uses:

- **Sliding window** input to capture memory effects (similar concept to MP)
- **Amplitude features** (|x|^2, |x|^4, |x|^6) matching memory polynomial basis functions
- **Residual blocks** for stable deep network training
- **Skip connection** to ensure identity initialization
- **Real-valued processing** of complex signals (split into I/Q)

Neural Network vs Memory Polynomial Trade-offs:

+-------------------+------------------+-------------------+
| Aspect            | NN-DPD           | MP/LS-DPD         |
+-------------------+------------------+-------------------+
| Expressiveness    | Arbitrary        | Polynomial only   |
| Training          | Iterative (slow) | Closed-form (fast)|
| Interpretability  | Black box        | Physical meaning  |
| Hyperparameters   | Many             | Few               |
| Generalization    | Risk of overfit  | Well-understood   |
+-------------------+------------------+-------------------+

The implementation follows TensorFlow/Keras Layer conventions for seamless
integration with Sionna and standard training loops.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, PReLU


class ResidualBlock(Layer):
    """
    Fully-connected residual block with layer normalization and PReLU.

    Implements the pre-activation residual block pattern where normalization
    and activation precede each linear transformation. This ordering improves
    gradient flow and training stability compared to post-activation.

    Parameters
    ----------
    units : int, optional
        Number of units in each dense layer. Default: 64.
    num_layers : int, optional
        Number of dense layers in the block. Default: 2.
    **kwargs
        Additional keyword arguments passed to Keras Layer.

    Attributes
    ----------
    units : int
        Number of units per layer.
    num_layers : int
        Number of layers in block.

    Notes
    -----
    **Why PReLU?**

    DPD corrections can be both positive and negative. PReLU (Parametric
    ReLU) has a learnable slope for negative inputs, allowing the network
    to optimize how negative corrections are handled. Standard ReLU would
    zero out negative values, limiting the correction space.

    **Why Layer Normalization?**

    Layer normalization (vs batch normalization) normalizes across features
    rather than batch dimension. This is important for DPD where:

    - Batch sizes may be small or variable
    - Each sample should be processed consistently
    - Inference behavior should match training exactly

    **Skip Connection:**

    The residual connection ``output = F(x) + x`` ensures:

    - Gradient can flow directly through the skip path
    - Block can learn to output zero (identity mapping)
    - Deeper networks remain trainable

    Example
    -------
    >>> block = ResidualBlock(units=64, num_layers=2)
    >>> x = tf.random.normal([16, 100, 64])  # [batch, time, features]
    >>> y = block(x)
    >>> y.shape
    TensorShape([16, 100, 64])
    """

    def __init__(self, units: int = 64, num_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.units = int(units)
        self.num_layers = int(num_layers)

        # Pre-activation pattern: norm -> activation -> dense
        self._layer_norms = [
            LayerNormalization(axis=-1) for _ in range(self.num_layers)
        ]
        # PReLU with shared slope across time dimension (axis=1).
        # Each feature can have its own slope.
        self._activations = [PReLU(shared_axes=[1]) for _ in range(self.num_layers)]
        self._dense_layers = [
            Dense(self.units, activation=None, kernel_initializer="glorot_uniform")
            for _ in range(self.num_layers)
        ]

    def call(self, inputs):
        """
        Forward pass through residual block.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor, shape ``[batch, time, units]``.

        Returns
        -------
        tf.Tensor
            Output tensor with residual connection, same shape as input.
        """
        z = inputs
        for ln, act, dense in zip(
            self._layer_norms, self._activations, self._dense_layers
        ):
            z = ln(z)
            z = act(z)
            z = dense(z)
        # Residual connection: add input to transformed output.
        return z + inputs


class NeuralNetworkDPD(Layer):
    """
    Fully-connected feedforward neural network predistorter with memory.

    Implements a neural network that learns the PA inverse function through
    gradient-descent and backpropagation. Memory effects are captured via
    a sliding window over input samples, similar in concept to the
    memory polynomial approach.

    The network receives explicit amplitude features (|x|^2, |x|^4, |x|^6) in
    addition to I/Q components. This is critical because PA distortion is
    fundamentally amplitude-driven (AM-AM and AM-PM conversion). Without
    these features, the network must learn from scratch that amplitude
    matters, which is inefficient and leads to suboptimal spectral behavior.

    Parameters
    ----------
    memory_depth : int, optional
        Number of samples in sliding window. Larger values capture longer
        memory effects but increase computation. Default: 4.
    num_filters : int, optional
        Number of units in hidden layers. Controls model capacity.
        Default: 64.
    num_layers_per_block : int, optional
        Number of dense layers per residual block. Default: 2.
    num_res_blocks : int, optional
        Number of residual blocks. More blocks increase depth and capacity.
        Default: 3.
    **kwargs
        Additional keyword arguments passed to Keras Layer.

    Attributes
    ----------
    _memory_depth : int
        Sliding window size.
    _num_filters : int
        Hidden layer width.
    _num_res_blocks : int
        Number of residual blocks.

    Notes
    -----
    **Amplitude Features:**

    PA nonlinearity is driven by instantaneous signal amplitude. The memory
    polynomial explicitly models terms like ``|x[n-m]|^p · x[n-m]`` where p
    is even (0, 2, 4, 6...) corresponding to odd-order nonlinearities. By
    providing |x|^2, |x|^4, |x|^6 as explicit input features, the network is
    given direct access to the same basis functions that make memory
    polynomials effective for PA linearization.

    Feature layout per sample (5 * memory_depth total features):
    ``[real[n-M+1:n], imag[n-M+1:n], |x|^2[n-M+1:n], |x|^4[n-M+1:n], |x|^6[n-M+1:n]]``

    **Identity Initialization:**

    The output layer is initialized to zeros, so the initial network output
    is just the skip connection (identity function). This ensures:

    - Initial predistorter is pass-through (no distortion)
    - Training starts from a reasonable point
    - Network learns corrections relative to identity

    **Complex Signal Handling:**

    Complex signals are split into real and imaginary parts for processing.
    This is necessary because standard neural network layers operate on
    real-valued tensors. The network learns to process I and Q jointly,
    capturing their correlations.

    **Causal Processing:**

    The sliding window only includes current and past samples (causal).
    This is appropriate for real-time DPD where future samples are
    unavailable. Zero-padding handles the initial transient.

    **miscellaneous:**

    - Input must be complex64 tensor
    - Batch dimension is optional (will be added if missing)
    - Output has same shape as input
    - Initial (untrained) output equals input (identity)

    Example
    -------
    >>> dpd = NeuralNetworkDPD(memory_depth=4, num_filters=64)
    >>> x = tf.complex(tf.random.normal([16, 1024]), tf.random.normal([16, 1024]))
    >>> y = dpd(x)
    >>> y.shape
    TensorShape([16, 1024])

    See Also
    --------
    LeastSquaresDPD : Polynomial-based DPD with closed-form training.
    NN_DPDSystem : System wrapper for NN-DPD training and inference.
    """

    def __init__(
        self,
        memory_depth: int = 4,
        num_filters: int = 64,
        num_layers_per_block: int = 2,
        num_res_blocks: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._memory_depth = int(memory_depth)
        self._num_filters = int(num_filters)
        self._num_layers_per_block = int(num_layers_per_block)
        self._num_res_blocks = int(num_res_blocks)

        # Input feature size: memory_depth samples × 5 features each.
        # Features per sample: real, imag, |x|^2, |x|^4, |x|^6
        self._num_features_per_sample = 5
        self._input_size = self._num_features_per_sample * self._memory_depth

        # Project input features to hidden dimension.
        self._input_dense = Dense(
            self._num_filters,
            activation=None,
            kernel_initializer="glorot_uniform",
            name="input_projection",
        )

        # Stack of residual blocks for deep nonlinear processing.
        self._res_blocks = [
            ResidualBlock(
                units=self._num_filters,
                num_layers=self._num_layers_per_block,
            )
            for _ in range(self._num_res_blocks)
        ]

        # Output projection: hidden -> [real, imag].
        # Zero initialization ensures initial output is identity (via skip).
        self._output_dense = Dense(
            2,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="output",
        )

    def _create_sliding_windows_batched(self, signal):
        """
        Extract sliding window features from batched complex signal.

        Creates a causal sliding window where each output position sees
        the current sample and (memory_depth - 1) past samples. Includes
        amplitude features (|x|^2, |x|^4, |x|^6) to capture PA nonlinearity.

        Parameters
        ----------
        signal : tf.Tensor
            Complex input signal, shape ``[batch, num_samples]``.

        Returns
        -------
        tf.Tensor
            Feature tensor, shape ``[batch, num_samples, 5 * memory_depth]``.
            Layout per sample:
            ``[real[n-M+1:n], imag[n-M+1:n], |x|^2[n-M+1:n], |x|^4[n-M+1:n], |x|^6[n-M+1:n]]``

        Notes
        -----
        Zero-padding at the start ensures output length equals input length.

        The amplitude features align with memory polynomial basis functions:

        - |x|^2 corresponds to 3rd-order nonlinearity (|x|^2 · x)
        - |x|^4 corresponds to 5th-order nonlinearity (|x|^4 · x)
        - |x|^6 corresponds to 7th-order nonlinearity (|x|^6 · x)

        This mirrors the structure of memory polynomial models where terms
        like ``|x[n-m]|^p · x[n-m]`` explicitly encode amplitude dependence.
        """
        batch_size = tf.shape(signal)[0]

        # Zero-pad at start for causal processing.
        # After padding: [B, memory_depth - 1 + num_samples]
        padding = self._memory_depth - 1
        pad_zeros = tf.zeros([batch_size, padding], dtype=signal.dtype)
        padded_signal = tf.concat([pad_zeros, signal], axis=1)

        # Extract sliding windows using tf.signal.frame.
        # Output: [B, num_samples, memory_depth]
        windows = tf.signal.frame(padded_signal, self._memory_depth, 1, axis=1)

        # Split complex into real/imag and convert to float32.
        real_part = tf.cast(tf.math.real(windows), tf.float32)
        imag_part = tf.cast(tf.math.imag(windows), tf.float32)

        # Compute amplitude features for AM-AM/AM-PM modeling.
        # These align with memory polynomial basis functions where |x|^p · x
        # represents (p+1)-order nonlinearity:
        #   |x|^2 · x = 3rd order intermodulation
        #   |x|^4 · x = 5th order intermodulation
        #   |x|^6 · x = 7th order intermodulation
        amplitude = tf.abs(windows)
        amplitude = tf.cast(amplitude, tf.float32)
        amp_squared = amplitude**2  # |x|^2 (3rd order basis)
        amp_power4 = amplitude**4  # |x|^4 (5th order basis)
        amp_power6 = amplitude**6  # |x|^6 (7th order basis)

        # Normalize amplitude features to prevent gradient explosion.
        # Due to high PAPR of OFDM, scale by the theoretical
        # expected values.
        amp_power4 = amp_power4 / 2.0
        amp_power6 = amp_power6 / 6.0

        # Concatenate all features: [B, num_samples, 5 * memory_depth]
        # Layout: [real, imag, |x|^2, |x|^4, |x|^6] for each time tap
        features = tf.concat(
            [real_part, imag_part, amp_squared, amp_power4, amp_power6], axis=-1
        )

        return features

    def _output_to_complex(self, output):
        """
        Convert real-valued network output to complex signal.

        Parameters
        ----------
        output : tf.Tensor
            Network output with real/imag channels, shape ``[..., 2]``.

        Returns
        -------
        tf.Tensor
            Complex signal, shape ``[...]`` (last dimension removed).
        """
        return tf.complex(output[..., 0], output[..., 1])

    def call(self, x, training=None):
        """
        Apply neural network predistortion to input signal.

        Parameters
        ----------
        x : tf.Tensor
            Input signal, shape ``[batch, num_samples]`` or ``[num_samples]``.
            Must be complex dtype.
        training : bool or None, optional
            Training mode flag. Affects dropout/batch norm if present.
            Currently unused but included for Keras compatibility.

        Returns
        -------
        tf.Tensor
            Predistorted signal, same shape as input.

        Notes
        -----
        The network computes ``y = NN(x) + x`` where the skip connection
        ensures identity behavior when NN outputs zeros (initial state).
        """
        # Handle unbatched input by temporarily adding batch dimension.
        input_ndims = len(x.shape)
        if input_ndims == 1:
            x = tf.expand_dims(x, axis=0)

        x = tf.cast(x, tf.complex64)

        # Extract sliding window features: [B, N, 5*M]
        # Includes real, imag, |x|^2, |x|^4, |x|^6 for each memory tap
        features = self._create_sliding_windows_batched(x)

        # Extract current sample for skip connection.
        # In the feature layout:
        #   real[n-M+1:n] at indices [0, M-1]
        #   imag[n-M+1:n] at indices [M, 2M-1]
        # Current sample: real at index (M-1), imag at index (2M-1)
        M = self._memory_depth
        skip_real = features[..., M - 1]
        skip_imag = features[..., 2 * M - 1]
        skip = tf.stack([skip_real, skip_imag], axis=-1)  # [B, N, 2]

        # Forward through network.
        z = self._input_dense(features)  # [B, N, num_filters]

        for block in self._res_blocks:
            z = block(z)

        z = self._output_dense(z)  # [B, N, 2]

        # Skip connection: network learns correction relative to identity.
        z = z + skip

        # Convert back to complex.
        y = self._output_to_complex(z)

        # Remove batch dimension if input was unbatched.
        if input_ndims == 1:
            y = tf.squeeze(y, axis=0)

        return y
