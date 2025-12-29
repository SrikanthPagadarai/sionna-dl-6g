import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import tensorflow as tf  # noqa: E402
from typing import Dict, Any  # noqa: E402
from sionna.phy.mapping import BinarySource, Mapper  # noqa: E402
from sionna.phy.fec.ldpc import LDPC5GEncoder  # noqa: E402
from sionna.phy.ofdm import ResourceGridMapper  # noqa: E402
from .config import Config  # noqa: E402


class Tx:
    """
    Uses a shared CSI instance (composition) so Tx, Channel, Rx all see the SAME h_freq.
    Pipeline:
      BinarySource -> LDPC5GEncoder -> Mapper -> ResourceGridMapper
    """

    def __init__(self, cfg: Config, channel_coding_off: bool = False):
        self._cfg = cfg
        self._channel_coding_off = channel_coding_off

        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._cfg.k, self._cfg.n)
        self._mapper = Mapper(self._cfg.modulation, self._cfg.num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._cfg.rg)
        self._num_streams_per_tx = self._cfg.num_streams_per_tx

    @tf.function
    def __call__(self, batch_size: tf.Tensor, h_freq: tf.Tensor) -> Dict[str, Any]:

        # Bits -> code -> symbols -> RG
        b = None
        if self._channel_coding_off:
            c = self._binary_source(
                [batch_size, 1, self._num_streams_per_tx, self._cfg.n]
            )
        else:
            b = self._binary_source(
                [batch_size, 1, self._num_streams_per_tx, self._cfg.k]
            )
            c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        return {"b": b, "c": c, "x": x, "x_rg": x_rg}
