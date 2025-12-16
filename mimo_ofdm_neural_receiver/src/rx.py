import tensorflow as tf
from typing import Dict, Any
from sionna.phy.ofdm import LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mapping import Demapper
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from .config import Config
from .csi import CSI


class Rx:
    """
    Full receiver chain; uses the *shared* CSI.h_freq for UL perfect-CSI path.
    """

    def __init__(self, cfg: Config, csi: CSI):
        self._cfg = cfg
        self._csi = csi

        self._ce = LSChannelEstimator(self._cfg.rg, interpolation_type="nn")
        self._eq = LMMSEEqualizer(self._cfg.rg, self._cfg.sm)
        self._demapper = Demapper(
            "app", self._cfg.modulation, self._cfg.num_bits_per_symbol
        )
        self._decoder = LDPC5GDecoder(
            LDPC5GEncoder(self._cfg.k, self._cfg.n), hard_out=True
        )

    @tf.function
    def __call__(
        self,
        y: tf.Tensor,
        h_freq: tf.Tensor,
        no: tf.Tensor,
    ) -> Dict[str, Any]:

        # Perfect vs estimated CSI
        if self._cfg.perfect_csi:
            h_hat = self._csi.remove_nulled_scs(h_freq)
            err_var = tf.cast(0.0, tf.float32)
        else:
            h_hat, err_var = self._ce(y, no)

        # Equalize, demap, decode
        x_hat, no_eff = self._eq(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        b_hat = self._decoder(llr)

        return {
            "h_hat": h_hat,
            "err_var": err_var,
            "x_hat": x_hat,
            "no_eff": no_eff,
            "llr": llr,
            "b_hat": b_hat,
        }
