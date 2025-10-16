import tensorflow as tf
import sionna as sn
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
from typing import Optional

from .config import Config
from .csi import CSI

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

        self._layer_norms = [LayerNormalization(axis=(-1, -2, -3)) for _ in range(self.num_resnet_layers)]
        self._convs = [Conv2D(filters=self.num_conv2d_filters,kernel_size=(3, 3),padding="same",activation=None)
                       for _ in range(self.num_resnet_layers)]

    def call(self, inputs):
        z = inputs
        for ln, conv in zip(self._layer_norms, self._convs):
            z = ln(z)
            z = tf.nn.relu(z)
            z = conv(z)
        return z + inputs


class NeuralRx(Layer):
    """
    Convolutional neural receiver that maps (y, no) -> per-RG LLRs.
    """
    def __init__(self, cfg: Config, channel_coding_off: bool = False, num_conv2d_filters: int = 128, num_resnet_layers: int = 2, num_res_blocks: int = 4):
        super().__init__()
        self.cfg = cfg
        self._channel_coding_off = bool(channel_coding_off)
        self.num_conv2d_filters = int(num_conv2d_filters)
        self.num_resnet_layers = int(num_resnet_layers)
        self.num_res_blocks = int(num_res_blocks)

        self._input_conv = Conv2D(filters=self.num_conv2d_filters,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation=None)

        # Residual stack (parametrized by CSI/Config)
        self._res_blocks = [ResidualBlock(num_conv2d_filters=self.num_conv2d_filters,num_resnet_layers=self.num_resnet_layers) for _ in range(self.num_res_blocks)]

        # Output conv yields one channel per bit (LLR per bit)
        self._output_conv = Conv2D(filters=int(self.cfg.rg.num_streams_per_tx)*int(self.cfg.num_bits_per_symbol),
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation=None)
        
        self._rg_demapper = sn.phy.ofdm.ResourceGridDemapper(self.cfg.rg, self.cfg.sm)
        self._decoder = LDPC5GDecoder(LDPC5GEncoder(self.cfg.k, self.cfg.n), hard_out=True)

    def call(self, y: tf.Tensor, no: tf.Tensor) -> tf.Tensor:
        # Remove num_rx dim and put antenna dim last: [B, S, Ns, Nsc] -> [B, Ns, Nsc, S]
        y = tf.transpose(tf.squeeze(y, axis=1), [0, 2, 3, 1])

        # Noise power (log10), broadcast without explicit shapes
        no_log = sn.phy.utils.log10(no)
        no_map = tf.ones_like(y[..., :1], dtype=y.dtype.real_dtype) * tf.cast(no_log, y.dtype.real_dtype)

        # Stack real/imag per-antenna plus noise channel: [B, Ns, Nsc, 2*S+1]
        z = tf.concat([tf.math.real(y), tf.math.imag(y), no_map], axis=-1)

        # Input conv
        z = self._input_conv(z)

        # Residual stack
        for block in self._res_blocks:
            z = block(z)

        # Output conv -> channels = num_streams * bits
        z = self._output_conv(z)

        # Prepare shape for resource-grid demapping: [B, 1, S, Ns, Nsc, bits]
        B, Ns, Nsc = tf.unstack(tf.shape(z)[:3])
        S    = self.cfg.rg.num_streams_per_tx
        bits = int(self.cfg.num_bits_per_symbol)

        # resource-grid demapping
        z = tf.reshape(z, [B, Ns, Nsc, S, bits])      # [B, Ns, Nsc, S, bits]
        z = tf.transpose(z, [0, 3, 1, 2, 4])          # [B, S, Ns, Nsc, bits]
        llr = self._rg_demapper(z[:, tf.newaxis, ...]) # [B, 1, S, Ns, Nsc, bits] -> demap

        # Flatten to [B, 1, S, n]
        llr = tf.reshape(llr, [tf.shape(llr)[0], tf.shape(llr)[1], tf.shape(llr)[2], -1])

        # channel decoding
        b_hat = None
        if not self._channel_coding_off:
            b_hat = self._decoder(llr)
        
        return {"llr": llr, "b_hat": b_hat}


if __name__ == "__main__":
    # Minimal shape sanity check for NeuralRx
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import tensorflow as tf
    from sionna.phy.utils import ebnodb2no

    # Choose scenario
    DIRECTION = "downlink"   # "uplink" or "downlink"
    PERFECT_CSI = True
    BATCH_SIZE = 4
    EBN0_DB = 15.0

    # Build config/CSI
    cfg = Config(direction=DIRECTION, perfect_csi=PERFECT_CSI)
    B = tf.constant(BATCH_SIZE, dtype=tf.int32)
    csi = CSI(cfg)
    csi.build(B)

    # Pull some handy dims from the resource grid / config
    n_sym = cfg.rg.num_ofdm_symbols
    n_sc  = cfg.rg.fft_size
    num_streams = cfg.rg.num_streams_per_tx  # number of layers/streams
    bits_per_sym = cfg.num_bits_per_symbol

    # Dummy received signal 'y' (frequency domain, post-OFDM)
    # Shape convention for NeuralRx input in this repo:
    #   y: [B, 1, N_streams, N_sym, N_sc] (complex)
    y_shape = (BATCH_SIZE, 1, num_streams, n_sym, n_sc)
    y_real = tf.random.normal(y_shape, dtype=tf.float32)
    y_imag = tf.random.normal(y_shape, dtype=tf.float32)
    y = tf.complex(y_real, y_imag)

    # Noise power 'no' (scalar). If you prefer per-sample, change to shape [BATCH_SIZE].
    no = ebnodb2no(tf.constant(EBN0_DB, tf.float32),
                   cfg.num_bits_per_symbol,
                   cfg.coderate,
                   cfg.rg)

    # Create and run NeuralRx
    nrx = NeuralRx(cfg)

    # Forward pass
    rx_out = nrx(y, no)  # expected: [B, 1, N_streams, N_sym, N_sc, bits_per_sym]

    # Print a quick summary of shapes
    print("===== NeuralRx Shape Check =====")
    print(f"Direction                 : {DIRECTION}, Perfect CSI: {PERFECT_CSI}")
    print(f"BATCH_SIZE                : {BATCH_SIZE}")
    print(f"RG: N_sym={n_sym}, N_sc={n_sc}, N_streams={num_streams}, bits/sym={int(bits_per_sym)}")
    print(f"y shape                   : {y.shape} (expect [B, 1, N_streams, N_sym, N_sc])")
    print(f"no shape/value            : {no.shape} | {float(tf.reshape(no, [-1])[0]) if tf.size(no)>0 else 'scalar'}")
    print(f"NeuralRx output (LLR, RG) : {rx_out["llr"].shape} (expect [B, 1, N_streams, (N_sym-n_pilots)*(N_sc-nguard-1)])")

    # If you want to sanity-peek a slice:
    print("Sample LLR vector at [0, 0, 0, :]:")
    print(rx_out["llr"][0, 0, 0, :])

