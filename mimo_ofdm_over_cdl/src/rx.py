import tensorflow as tf
from typing import Dict, Any, Optional
from sionna.phy.ofdm import LSChannelEstimator, LMMSEEqualizer, ResourceGridDemapper
from sionna.phy.mapping import Demapper
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from .config import Config
from .csi import CSI

class Rx:
    """
    Full receiver chain; uses the *shared* CSI.h_freq for UL perfect-CSI path.
    """
    def __init__(self, cfg: Config, csi: CSI):
        self.cfg = cfg.build()
        self.csi = csi
        self.perfect_csi = cfg.perfect_csi
        self.rg = self.csi.rg

        self._ce = LSChannelEstimator(self.rg, interpolation_type="nn")
        self._eq = LMMSEEqualizer(self.rg, self.cfg.sm)
        self._demapper = Demapper("app", self.cfg.modulation, self.cfg.num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(LDPC5GEncoder(self.cfg.k, self.cfg.n), hard_out=True)

    @tf.function
    def __call__(self, batch_size: tf.Tensor, y: tf.Tensor, no: tf.Tensor, g: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        self.csi.assert_batch(batch_size)

        # Perfect vs estimated CSI
        if self.perfect_csi:
            h_hat = self.csi.remove_nulled_scs(self.csi.h_freq)
            if self.cfg.direction == "downlink":
                if g is None:
                    raise ValueError("perfect_csi=True (downlink) requires Tx-provided 'g'.")
                h_hat = g
            '''
            if self.cfg.direction == "uplink":
                h_hat = self.csi.remove_nulled_scs(self.csi.h_freq)
            else:
                if g is None:
                    raise ValueError("perfect_csi=True (downlink) requires Tx-provided 'g'.")
                h_hat = g
            '''
            err_var = 0.0
        else:
            h_hat, err_var = self._ce(y, no)

        # Equalize, demap, decode
        x_hat, no_eff = self._eq(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        b_hat = self._decoder(llr)

        return {"h_hat": h_hat, "err_var": err_var, "x_hat": x_hat, "no_eff": no_eff, "llr": llr, "b_hat": b_hat}


if __name__ == "__main__":
    """
    Example standalone test for RX stage.
    Uses dummy y and no for demonstration.
    """
    from .csi import CSI
    from sionna.phy.utils import ebnodb2no

    cfg = Config(direction="downlink", perfect_csi=True)
    B = tf.constant(4, dtype=tf.int32)
    EbNo_dB = tf.constant(10.0)

    csi = CSI(cfg, batch_size=B)
    rx = Rx(cfg, csi)

    # Dummy input signal
    n_sym = csi.rg.num_ofdm_symbols
    n_sc  = csi.rg.fft_size
    num_tx = csi.rg.num_tx
    num_streams_per_tx = csi.rg.num_streams_per_tx
    n_guard_left, n_guard_right = csi.rg.num_guard_carriers
    y_shape = (B, num_tx, num_streams_per_tx, n_sym, n_sc)
    g_shape = (B, num_tx, num_streams_per_tx, num_tx, num_streams_per_tx, n_sym, n_sc-n_guard_left-n_guard_right-1)
    y = tf.complex(tf.random.normal(y_shape, dtype=tf.float32),tf.random.normal(y_shape, dtype=tf.float32))
    g = tf.complex(tf.random.normal(g_shape, dtype=tf.float32),tf.random.normal(g_shape, dtype=tf.float32))
    no = ebnodb2no(EbNo_dB, cfg.num_bits_per_symbol, cfg.coderate, csi.rg)

    out = rx(B, y, no, g)
    print("\n[RX] Output shapes:")
    for k, v in out.items():
        print(f"{k:10s}: shape={v.shape}")
