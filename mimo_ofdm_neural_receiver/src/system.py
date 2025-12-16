import tensorflow as tf
from typing import Dict, Any
from sionna.phy.utils import ebnodb2no, compute_ber
from tensorflow.keras import Model

from .config import Config, BitsPerSym, CDLModel
from .csi import CSI
from .tx import Tx
from .channel import Channel
from .rx import Rx
from .neural_rx import NeuralRx


class System(Model):
    def __init__(self,
                 *,
                 training: bool = False,
                 perfect_csi: bool = False,
                 cdl_model: CDLModel = "D",
                 delay_spread: float = 300e-9,
                 carrier_frequency: float = 2.6e9,
                 speed: float = 0.0,
                 num_bits_per_symbol: BitsPerSym = BitsPerSym.QPSK,
                 use_neural_rx: bool = False,
                 num_conv2d_filters: int = 128,
                 num_resnet_layers: int = 2,
                 num_res_blocks: int = 4,
                 name: str = "system"):
        super().__init__(name=name)

        self._training = training

        self._use_neural_rx = bool(use_neural_rx)
        self._num_conv2d_filters = num_conv2d_filters
        self._num_resnet_layers = num_resnet_layers
        self._num_res_blocks = num_res_blocks

        self._cfg = Config(
            perfect_csi=perfect_csi,
            cdl_model=cdl_model,
            delay_spread=delay_spread,
            carrier_frequency=carrier_frequency,
            speed=speed,
            num_bits_per_symbol=num_bits_per_symbol,
        )

        # CSI/Tx/Channel/Rx
        self._csi = CSI(self._cfg)
        self._tx  = Tx(self._cfg, self._training) # in training, channel coding is off in Tx
        self._ch  = Channel()
        self._rx  = Rx(self._cfg, self._csi) # baseline Rx (not trained)
        self._neural_rx = NeuralRx(self._cfg, self._training, self._num_conv2d_filters, 
                                   self._num_resnet_layers, self._num_res_blocks) # in training, correspondigly, channel decoding is off in NeuralRx

        # Loss function
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=[], dtype=tf.int32),    # scalar batch size
            tf.TensorSpec(shape=[], dtype=tf.float32),  # scalar ebno_db
        ]
    )
    def call_scalar(self, batch_size, ebno_db_scalar):
        """Scalar-SNR entry point for Sionna PlotBER.simulate()"""
        ebno_vec = tf.fill([batch_size], ebno_db_scalar)  # expand to (B,)
        return self.__call__(batch_size, ebno_vec)        # reuse your vector path

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=[], dtype=tf.int32),      # batch_size (scalar)
            tf.TensorSpec(shape=[None], dtype=tf.float32) # ebno_db vector (len == batch_size)
        ]
    )
    def __call__(self, batch_size: tf.Tensor, ebno_db: tf.Tensor):
        # Build CSI for this batch_size once per call
        h_freq = self._csi.build(batch_size)

        no = ebnodb2no(ebno_db, self._cfg.num_bits_per_symbol, self._cfg.coderate, self._cfg.rg)

        tx_out = self._tx(batch_size, h_freq)
        y_out  = self._ch(tx_out["x_rg_tx"], h_freq, no)

        rx_to_use = self._neural_rx if self._use_neural_rx else self._rx
        rx_args_to_pass = (y_out["y"], no, batch_size) if self._use_neural_rx else (y_out["y"], h_freq, no, tx_out["g"])
        rx_out = rx_to_use(*rx_args_to_pass)

        if self._use_neural_rx and self._training:
            loss = self.bce(tx_out["c"], rx_out["llr"])
            return loss

        return tx_out["b"], rx_out["b_hat"]



if __name__ == "__main__":
    import os, tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    system = System(
        perfect_csi=True,
        cdl_model="D",
        delay_spread=300e-9,
        carrier_frequency=2.6e9,
        speed=0.0,
        use_neural_rx=False,
        name="system",
    )

    B = tf.constant(4, tf.int32)
    EbNo = tf.constant(40.0, tf.float32)
    b, b_hat = system(B, EbNo)
    tf.print("BER:", compute_ber(b, b_hat))
