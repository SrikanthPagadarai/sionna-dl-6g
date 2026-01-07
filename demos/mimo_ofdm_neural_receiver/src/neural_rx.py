# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Neural receiver for MIMO-OFDM simulation.

Implements a convolutional neural network-based receiver that
learns to map received OFDM signals directly to log-likelihood ratios (LLRs),
bypassing traditional channel estimation and equalization stages:

    Received Signal (y) + Noise Power (no) -> CNN -> LLRs -> [LDPC Decoder]

The architecture uses a ResNet-style design with:
- Input convolution to expand channel dimension
- Stack of residual blocks with layer normalization
- Output convolution producing per-bit LLR predictions

Key design decisions:

1. **End-to-end learning**: The network jointly learns channel estimation,
   equalization, and demapping in a single differentiable pipeline.

2. **Noise power as input**: Feeding log10(no) helps the network adapt its
   behavior across different SNR operating points.

3. **Training mode**: When ``channel_coding_off=True``, LDPC decoding is
   skipped and raw LLRs are returned for BCE loss computation.
"""

import tensorflow as tf
import sionna as sn
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu

from .config import Config


class ResidualBlock(Layer):
    """
    Residual block with convolutions and layer normalization.

    Implements a pre-activation residual block where normalization and
    activation precede each convolution. The skip connection enables
    gradient flow through deep networks and allows the block to learn
    residual refinements rather than full transformations.

    Architecture per layer:
        LayerNorm -> ReLU -> Conv2D(3x3)

    The block applies ``num_resnet_layers`` such layers sequentially,
    then adds the input via skip connection.

    Parameters
    ----------
    num_conv2d_filters : int, (default 128)
        Number of output channels for each convolution. All convolutions
        in the block use the same filter count.

    num_resnet_layers : int, (default 2)
        Number of normalization-activation-convolution sequences in the
        block. Must be at least 1.

    Raises
    ------
    ValueError
        If ``num_resnet_layers < 1``.

    Note
    ----
    Layer normalization is applied over spatial and channel dimensions
    (axes -1, -2, -3) rather than batch normalization. This provides
    more stable training with small batch sizes and varying SNR conditions.

    The 3x3 kernel with 'same' padding preserves spatial dimensions,
    allowing the skip connection to work without dimension adjustment.
    """

    def __init__(self, num_conv2d_filters: int = 128, num_resnet_layers: int = 2):
        """
        Initialize residual block layers.

        Parameters
        ----------
        num_conv2d_filters : int, (default 128)
            Output channels per convolution.

        num_resnet_layers : int, (default 2)
            Depth of the residual block (number of conv layers).

        Post-conditions
        ---------------
        - ``_layer_norms`` contains ``num_resnet_layers`` LayerNorm instances.
        - ``_convs`` contains ``num_resnet_layers`` Conv2D instances.
        - All convolutions use 3x3 kernels with 'same' padding.
        """
        super().__init__()
        if num_resnet_layers < 1:
            raise ValueError("num_resnet_layers must be >= 1")
        self.num_conv2d_filters = int(num_conv2d_filters)
        self.num_resnet_layers = int(num_resnet_layers)

        # Pre-activation design: LayerNorm -> ReLU -> Conv for each layer
        self._layer_norms = [
            LayerNormalization(axis=(-1, -2, -3)) for _ in range(self.num_resnet_layers)
        ]
        self._convs = [
            Conv2D(
                filters=self.num_conv2d_filters,
                kernel_size=(3, 3),
                padding="same",
                activation=None,
            )
            for _ in range(self.num_resnet_layers)
        ]

    # [resblock-call-start]
    def call(self, inputs):
        """
        Apply residual transformation to input tensor.

        Parameters
        ----------
        inputs : tf.Tensor, float32, [batch, height, width, channels]
            Input feature maps. Channel dimension must match
            ``num_conv2d_filters`` for the skip connection to work.

        Returns
        -------
        tf.Tensor, float32, [batch, height, width, channels]
            Output feature maps with same shape as input.

        Pre-conditions
        --------------
        - Input must be float32 (assertion checks this for debugging).
        - Input channels should equal ``num_conv2d_filters``.

        Post-conditions
        ---------------
        - Output shape equals input shape.
        - Output = transform(input) + input (residual connection).

        Invariants
        ----------
        - Spatial dimensions are preserved (3x3 conv with 'same' padding).
        """
        z = inputs
        for ln, conv in zip(self._layer_norms, self._convs):
            # Debug assertion: catch dtype issues early in development
            tf.debugging.assert_type(z, tf.float32)
            z = ln(z)
            z = relu(z)
            z = conv(z)
        # Skip connection: enables gradient flow and residual learning
        return z + inputs

    # [resblock-call-end]


class NeuralRx(Layer):
    """
    Convolutional neural receiver mapping received signals to LLRs.

    This network replaces the traditional channel estimation, equalization,
    and demapping stages with a learned CNN that directly produces
    log-likelihood ratios for each coded bit. The architecture processes
    the received signal across a time-frequency dimensional grid.

    Architecture:
        1. **Input preparation**: Concatenate [Re(y), Im(y), log10(no)]
        2. **Input convolution**: Expand to ``num_conv2d_filters`` channels
        3. **Residual stack**: ``num_res_blocks`` residual blocks
        4. **Output convolution**: Reduce to ``num_streams x bits_per_symbol``
        5. **Reshape**: Reorganize to per-stream, per-bit LLR format
        6. **Resource grid demapper**: Extract data symbol positions
        7. **LDPC decoder** (optional): Decode to information bits

    Parameters
    ----------
    cfg : Config
        Configuration containing resource grid, modulation, and code params.

    channel_coding_off : bool, (default False)
        If True, skip LDPC decoding and return raw LLRs. Used during
        training to compute BCE loss against transmitted coded bits.

    num_conv2d_filters : int, (default 128)
        Channel dimension throughout the residual stack.

    num_resnet_layers : int, (default 2)
        Number of conv layers per residual block.

    num_res_blocks : int, (default 4)
        Number of residual blocks in the network.

    Attributes
    ----------
    _cfg : Config
        Reference to configuration object.

    _channel_coding_off : bool
        Whether to skip LDPC decoding.

    Note
    ----
    The noise power is fed in log10 scale because:
    1. SNR varies over orders of magnitude during training
    2. Log scale provides more uniform gradient behavior
    3. Empirically improves convergence and final performance

    Example
    -------
    >>> cfg = Config(num_bits_per_symbol=BitsPerSym.QPSK)
    >>> neural_rx = NeuralRx(cfg, channel_coding_off=True)
    >>> out = neural_rx(y, no, batch_size)
    >>> llrs = out["llr"]  # Shape: [batch, 1, num_streams, n]
    """

    def __init__(
        self,
        cfg: Config,
        channel_coding_off: bool = False,
        num_conv2d_filters: int = 256,
        num_resnet_layers: int = 2,
        num_res_blocks: int = 4,
    ):
        """
        Initialize neural receiver architecture.

        Parameters
        ----------
        cfg : Config
            Configuration specifying resource grid and modulation.

        channel_coding_off : bool, (default False)
            Skip LDPC decoding if True (training mode).

        num_conv2d_filters : int, (default 256)
            Width of the residual network.

        num_resnet_layers : int, (default 2)
            Depth of each residual block.

        num_res_blocks : int, (default 4)
            Number of residual blocks.

        """
        super().__init__()
        self._cfg = cfg
        self._channel_coding_off = bool(channel_coding_off)
        self.num_conv2d_filters = int(num_conv2d_filters)
        self.num_resnet_layers = int(num_resnet_layers)
        self.num_res_blocks = int(num_res_blocks)

        # Input conv: expand from (2*num_rx_ant + 1) to num_conv2d_filters channels
        self._input_conv = Conv2D(
            filters=self.num_conv2d_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=None,
        )

        # Residual stack for feature extraction
        self._res_blocks = [
            ResidualBlock(
                num_conv2d_filters=self.num_conv2d_filters,
                num_resnet_layers=self.num_resnet_layers,
            )
            for _ in range(self.num_res_blocks)
        ]

        # Output conv: contract to (num_streams x bits_per_symbol) for LLR output
        # Each output channel corresponds to one bit position in the constellation
        self._output_conv = Conv2D(
            filters=int(
                self._cfg.rg.num_streams_per_tx * self._cfg.num_bits_per_symbol
            ),
            kernel_size=(3, 3),
            padding="same",
            activation=None,
        )

        # Resource grid demapper extracts LLRs at data symbol positions
        self._rg_demapper = sn.phy.ofdm.ResourceGridDemapper(self._cfg.rg, self._cfg.sm)

        # LDPC decoder (used only during inference when channel_coding_off=False)
        self._decoder = LDPC5GDecoder(
            LDPC5GEncoder(self._cfg.k, self._cfg.n), hard_out=True
        )

    def call(self, y: tf.Tensor, no: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        """
        Process received signal to produce LLRs and optionally decoded bits.

        Parameters
        ----------
        y : tf.Tensor, complex64, [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
            Received OFDM signal after channel and noise.

        no : tf.Tensor, float32, [batch] or scalar
            Noise power spectral density.

        batch_size : tf.Tensor, int32, scalar
            Batch dimension size (needed for reshape operations in graph mode).

        Returns
        -------
        Dict[str, tf.Tensor]
            Dictionary containing:

            - ``"llr"``: Predicted log-likelihood ratios, shape
              [batch, 1, num_ut_ant, n].

            - ``"b_hat"``: Decoded information bits, shape
              [batch, 1, num_ut_ant, k]. None if ``channel_coding_off=True``.

        Note
        ----
        The tensor transformations in this method follow a specific sequence:

        1. Remove num_rx dimension (assuming single receiver)
        2. Transpose to [batch, ofdm_symbols, subcarriers, antennas]
        3. Split complex to real channels: 2xnum_rx_ant + 1 (noise) channels
        4. Process through CNN
        5. Reshape output to match ResourceGridDemapper expectations
        6. Extract data positions and reshape for decoder input
        """
        # =====================================================================
        # Input Preparation
        # =====================================================================
        # Remove num_rx dimension (single receiver assumed in this demo)
        y = tf.squeeze(y, axis=1)

        # Convert noise power to log scale for better neural network conditioning
        # SNR ranges span orders of magnitude; log scale normalizes the input
        no = sn.phy.utils.log10(no)

        # Transpose to image format: [batch, OFDM_symbols, subcarriers, antennas]
        # CNN expects spatial dims in the middle, channels last
        y = tf.transpose(y, [0, 2, 3, 1])

        # Broadcast noise power to spatial dimensions for concatenation
        # Shape: [batch] -> [batch, 1, 1, 1] -> [batch, H, W, 1]
        no = tf.reshape(no, [-1])
        no = no + tf.zeros(
            [tf.shape(y)[0]], dtype=no.dtype
        )  # ensure length: tf.shape(y)[0]
        no = tf.reshape(no, [tf.shape(y)[0], 1, 1, 1])  # [tf.shape(y)[0],1,1,1]

        # Broadcast to match spatial dimensions of y
        no = tf.broadcast_to(no, [tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], 1])

        # Concatenate real, imaginary, and noise channels
        # Input channels: 2 * num_rx_ant (real + imag) + 1 (noise) = 17 for 8 antennas
        z = tf.concat([tf.math.real(y), tf.math.imag(y), no], axis=-1)

        # =====================================================================
        # Neural Network Forward Pass
        # =====================================================================
        # Input convolution: expand channel dimension
        z = self._input_conv(z)

        # Residual stack: extract hierarchical features
        for block in self._res_blocks:
            z = block(z)

        # Output convolution: produce per-bit LLR predictions
        z = self._output_conv(z)

        # =====================================================================
        # Output Reshaping for Decoder Compatibility
        # =====================================================================
        # Reshape from [batch, H, W, streams*bits] to [batch, H, W, streams, bits]
        z = tf.reshape(
            z,
            [
                tf.shape(z)[0],
                tf.shape(z)[1],
                tf.shape(z)[2],
                self._cfg.rg.num_streams_per_tx,
                self._cfg.num_bits_per_symbol,
            ],
        )

        # Transpose to ResourceGridDemapper expected format
        # From [batch, ofdm, subcarrier, stream, bits] to [batch, stream, ofdm, subcarrier, bits]
        z = tf.transpose(z, [0, 3, 1, 2, 4])

        # Add num_tx dimension (required by ResourceGridDemapper)
        z = tf.expand_dims(z, axis=1)

        # Extract LLRs at data symbol positions (removes pilots)
        llr = self._rg_demapper(z)

        # Reshape to decoder input format: [batch, 1, num_ut_ant, n]
        llr = tf.reshape(llr, [batch_size, 1, self._cfg.num_ut_ant, self._cfg.n])

        # =====================================================================
        # Optional LDPC Decoding
        # =====================================================================
        b_hat = None
        if not self._channel_coding_off:
            # Decode LLRs to hard bit decisions
            b_hat = self._decoder(llr)

        return {"llr": llr, "b_hat": b_hat}
