import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
import tensorflow as tf
from typing import Dict, Any, Optional
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.ofdm import ResourceGridMapper, RZFPrecoder
from .config import Config
from .csi import CSI

class Tx:
    """
    Uses a shared CSI instance (composition) so Tx, Channel, Rx all see the SAME h_freq.
    Pipeline:
      BinarySource -> LDPC5GEncoder -> Mapper -> ResourceGridMapper -> (optional) RZFPrecoder
    """
    def __init__(self, cfg: Config, channel_coding_off: bool = False):
        self._cfg = cfg
        self._channel_coding_off = channel_coding_off

        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._cfg.k, self._cfg.n)
        self._mapper = Mapper(self._cfg.modulation, self._cfg.num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._cfg.rg)

        self._precoder: Optional[RZFPrecoder] = None
        if self._cfg.direction == "downlink":
            self._precoder = RZFPrecoder(self._cfg.rg, self._cfg.sm, return_effective_channel=True)

        self._num_streams_per_tx = self._cfg.num_streams_per_tx

    @tf.function
    def __call__(self, batch_size: tf.Tensor, h_freq: tf.Tensor) -> Dict[str, Any]:

        # Bits -> code -> symbols -> RG
        b = None
        if self._channel_coding_off:
            c = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._cfg.n])
        else:
            b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._cfg.k])
            c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        # If downlink, do precoding using the *shared* h_freq
        x_rg_tx = x_rg
        g = None
        if self._precoder is not None:
            x_rg_tx, g = self._precoder(x_rg, h_freq)

        return {"b": b, "c": c, "x": x, "x_rg": x_rg, "x_rg_tx": x_rg_tx, "g": g}


if __name__ == "__main__":
    """
    Example usage for standalone TX stage.
    Creates CSI once, then runs the TX pipeline.
    """
    from .csi import CSI

    cfg = Config(direction="uplink")
    B = tf.constant(4, dtype=tf.int32)

    csi = CSI(cfg)
    h_freq = csi.build(B)
    tx = Tx(cfg)
    out = tx(B, h_freq)

    print("\n[TX] Outputs:")
    for k, v in out.items():
        if v is not None:
            print(f"{k:10s}: shape={v.shape}")
        else:
            print(f"{k:10s}: None")