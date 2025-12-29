import tensorflow as tf
import sionna as sn
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu

from .config import Config


class ResidualBlock(Layer):
    """
    Conv-Residual block with configurable depth (num_resnet_layers).
    Kernel size is fixed at (3,3).
    """

    def __init__(self, num_conv2d_filters: int = 128, num_resnet_layers: int = 2):
        super().__init__()
        if num_resnet_layers < 1:
            raise ValueError("num_resnet_layers must be >= 1")
        self.num_conv2d_filters = int(num_conv2d_filters)
        self.num_resnet_layers = int(num_resnet_layers)

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

    def call(self, inputs):
        z = inputs
        for ln, conv in zip(self._layer_norms, self._convs):
            tf.debugging.assert_type(z, tf.float32)
            z = ln(z)
            z = relu(z)
            z = conv(z)
        return z + inputs


class NeuralRx(Layer):
    """
    Convolutional neural receiver that maps (y, no) -> per-RG LLRs.
    """

    def __init__(
        self,
        cfg: Config,
        channel_coding_off: bool = False,
        num_conv2d_filters: int = 128,
        num_resnet_layers: int = 2,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self._cfg = cfg
        self._channel_coding_off = bool(channel_coding_off)
        self.num_conv2d_filters = int(num_conv2d_filters)
        self.num_resnet_layers = int(num_resnet_layers)
        self.num_res_blocks = int(num_res_blocks)

        self._input_conv = Conv2D(
            filters=self.num_conv2d_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=None,
        )

        # Residual stack (parametrized by CSI/Config)
        self._res_blocks = [
            ResidualBlock(
                num_conv2d_filters=self.num_conv2d_filters,
                num_resnet_layers=self.num_resnet_layers,
            )
            for _ in range(self.num_res_blocks)
        ]

        # Output conv yields one channel per bit (LLR per bit)
        self._output_conv = Conv2D(
            filters=int(
                self._cfg.rg.num_streams_per_tx * self._cfg.num_bits_per_symbol
            ),
            kernel_size=(3, 3),
            padding="same",
            activation=None,
        )

        self._rg_demapper = sn.phy.ofdm.ResourceGridDemapper(self._cfg.rg, self._cfg.sm)
        self._decoder = LDPC5GDecoder(
            LDPC5GEncoder(self._cfg.k, self._cfg.n), hard_out=True
        )

    def call(self, y: tf.Tensor, no: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        # assuming a single receiver, remove the num_rx dimension
        y = tf.squeeze(y, axis=1)

        # feeding the noise power in log10 scale helps with the performance
        no = sn.phy.utils.log10(no)

        # put antenna dim last: y -> [B, N_sym, N_sc, N_ant]
        y = tf.transpose(y, [0, 2, 3, 1])

        # make `no` a [tf.shape(y)[0], 1, 1, 1] tensor
        #       (works for scalar or [tf.shape(y)[0]])
        no = tf.reshape(no, [-1])
        no = no + tf.zeros(
            [tf.shape(y)[0]], dtype=no.dtype
        )  # ensure length: tf.shape(y)[0]
        no = tf.reshape(no, [tf.shape(y)[0], 1, 1, 1])  # [tf.shape(y)[0],1,1,1]

        # Broadcast to one channel alongside features
        no = tf.broadcast_to(no, [tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], 1])

        # stack:
        # z dimensions
        #   - [batch_size, num ofdm symbols, num subcarriers, 2 * num rx antennas + 1]
        z = tf.concat([tf.math.real(y), tf.math.imag(y), no], axis=-1)

        # Input conv
        z = self._input_conv(z)

        # Residual stack
        for block in self._res_blocks:
            z = block(z)

        # Output conv
        z = self._output_conv(z)

        # Split channels into [S, bits]
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

        # transpose to a form expected by ResourceGridDemapper
        z = tf.transpose(z, [0, 3, 1, 2, 4])
        z = tf.expand_dims(z, axis=1)

        # resource-grid demapper
        llr = self._rg_demapper(z)

        # reshape to fit the dimensions expected at the input of channel decoder
        llr = tf.reshape(llr, [batch_size, 1, self._cfg.num_ut_ant, self._cfg.n])

        # channel decoding
        b_hat = None
        if not self._channel_coding_off:
            b_hat = self._decoder(llr)

        return {"llr": llr, "b_hat": b_hat}
