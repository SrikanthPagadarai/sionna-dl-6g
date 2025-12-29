"""Neural Network based Digital Pre-Distortion using Indirect Learning Architecture.

This module implements a feedforward neural network DPD following the standard
TensorFlow Layer pattern. Supports batched operation [B, num_samples] where
each batch element is processed independently.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, PReLU


class ResidualBlock(Layer):
    """
    Fully-connected residual block with configurable depth.

    Uses PReLU activation with learnable negative slope, allowing the network
    to optimize how negative values are handled. This is important for DPD
    corrections that can be both positive and negative.

    Args:
        units: Number of units in each dense layer
        num_layers: Number of dense layers in the block (default: 2)
    """

    def __init__(self, units: int = 64, num_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.units = int(units)
        self.num_layers = int(num_layers)

        self._layer_norms = [
            LayerNormalization(axis=-1) for _ in range(self.num_layers)
        ]
        self._activations = [PReLU(shared_axes=[1]) for _ in range(self.num_layers)]
        self._dense_layers = [
            Dense(self.units, activation=None, kernel_initializer="glorot_uniform")
            for _ in range(self.num_layers)
        ]

    def call(self, inputs):
        z = inputs
        for ln, act, dense in zip(
            self._layer_norms, self._activations, self._dense_layers
        ):
            z = ln(z)
            z = act(z)
            z = dense(z)
        return z + inputs


class NeuralNetworkDPD(Layer):
    """
    Neural Network based Digital Pre-Distortion using Indirect Learning Architecture.

    Inherits from tf.keras.layers.Layer for Sionna compatibility and differentiability.
    Supports batched operation where each batch element is processed independently.

    The neural network uses a sliding window to capture memory effects.
    Complex signals are handled by splitting into real/imaginary components.
    A skip connection ensures initial behavior is close to identity.

    Args:
        memory_depth: Number of samples in sliding window (default: 4)
        num_filters: Number of units in hidden layers (default: 64)
        num_layers_per_block: Layers per residual block (default: 2)
        num_res_blocks: Number of residual blocks (default: 3)
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

        # Input size: memory_depth samples × 2 (real + imag)
        self._input_size = 2 * self._memory_depth

        # Input projection layer
        self._input_dense = Dense(
            self._num_filters,
            activation=None,
            kernel_initializer="glorot_uniform",
            name="input_projection",
        )

        # Residual stack
        self._res_blocks = [
            ResidualBlock(
                units=self._num_filters,
                num_layers=self._num_layers_per_block,
            )
            for _ in range(self._num_res_blocks)
        ]

        # Output layer: 2 outputs (real and imaginary parts)
        # Initialize to zeros so initial output is just the skip connection
        self._output_dense = Dense(
            2,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="output",
        )

    def _create_sliding_windows_batched(self, signal):
        """
        Create sliding window features from batched complex signal.

        Args:
            signal: [B, num_samples] complex tensor

        Returns:
            features: [B, num_samples, 2 * memory_depth] real tensor (float32)
        """
        # signal: [B, num_samples]
        batch_size = tf.shape(signal)[0]

        # Pad signal at the beginning for causal processing:
        # [B, memory_depth-1 + num_samples]
        padding = self._memory_depth - 1
        pad_zeros = tf.zeros([batch_size, padding], dtype=signal.dtype)
        padded_signal = tf.concat([pad_zeros, signal], axis=1)

        # Create sliding windows using tf.signal.frame
        # Input: [B, padded_length], Output: [B, num_samples, memory_depth]
        windows = tf.signal.frame(padded_signal, self._memory_depth, 1, axis=1)

        # Split into real and imaginary parts (as float32)
        # [B, num_samples, memory_depth] each
        real_part = tf.cast(tf.math.real(windows), tf.float32)
        imag_part = tf.cast(tf.math.imag(windows), tf.float32)

        # Concatenate along last axis: [B, num_samples, 2 * memory_depth]
        features = tf.concat([real_part, imag_part], axis=-1)

        return features

    def _output_to_complex(self, output):
        """
        Convert network output to complex signal.

        Args:
            output: [B, num_samples, 2] or [num_samples, 2] real tensor

        Returns:
            [B, num_samples] or [num_samples] complex tensor
        """
        return tf.complex(output[..., 0], output[..., 1])

    def call(self, x, training=None):
        """
        Apply predistortion to input signal.

        Args:
            x: [B, num_samples] complex tensor (batched)
               or [num_samples] complex tensor (unbatched)

        Returns:
            Same shape as input - predistorted signal
        """
        # Handle unbatched input by adding batch dimension
        input_ndims = len(x.shape)
        if input_ndims == 1:
            x = tf.expand_dims(x, axis=0)  # [1, num_samples]

        x = tf.cast(x, tf.complex64)

        # Create sliding window features: [B, num_samples, 2 * memory_depth]
        features = self._create_sliding_windows_batched(x)

        # Extract current sample for skip connection (last sample in each window)
        # features layout per sample: [real(t-M+1),...,real(t), imag(t-M+1),...,imag(t)]
        skip_real = features[..., self._memory_depth - 1]  # [B, num_samples]
        skip_imag = features[..., 2 * self._memory_depth - 1]  # [B, num_samples]
        skip = tf.stack([skip_real, skip_imag], axis=-1)  # [B, num_samples, 2]

        # Forward pass through network
        # Dense layers operate on last dimension, so [B, num_samples, input_size] works
        z = self._input_dense(features)  # [B, num_samples, num_filters]

        for block in self._res_blocks:
            z = block(z)

        z = self._output_dense(z)  # [B, num_samples, 2]

        # Add skip connection: output = NN(x) + x
        # This ensures initial behavior is identity (when NN outputs zeros)
        z = z + skip

        # Convert to complex: [B, num_samples]
        y = self._output_to_complex(z)

        # Remove batch dimension if input was unbatched
        if input_ndims == 1:
            y = tf.squeeze(y, axis=0)

        return y
