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
    def __init__(self, cfg: Config, csi: CSI, channel_coding_off: bool = False):
        self.cfg = cfg
        self.csi = csi
        self._channel_coding_off = channel_coding_off

        self._ce = LSChannelEstimator(self.cfg.rg, interpolation_type="nn")
        self._eq = LMMSEEqualizer(self.cfg.rg, self.cfg.sm)
        self._demapper = Demapper("app", self.cfg.modulation, self.cfg.num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(LDPC5GEncoder(self.cfg.k, self.cfg.n), hard_out=True)

    @tf.function
    def __call__(self, batch_size: tf.Tensor, y: tf.Tensor, no: tf.Tensor, g: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        self.csi.assert_batch(batch_size)

        # Perfect vs estimated CSI
        if self.cfg.perfect_csi:
            if self.cfg.direction == "uplink":
                h_hat = self.csi.remove_nulled_scs(self.csi.h_freq)
            else:
                if g is None:
                    raise ValueError("perfect_csi=True (downlink) requires Tx-provided 'g'.")
                h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = self._ce(y, no)

        # Equalize, demap, decode
        x_hat, no_eff = self._eq(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        b_hat = None
        if not self._channel_coding_off:
            b_hat = self._decoder(llr)

        return {"h_hat": h_hat, "err_var": err_var, "x_hat": x_hat, "no_eff": no_eff, "llr": llr, "b_hat": b_hat}


if __name__ == "__main__":
    # Minimal standalone smoke test for Rx
    import tensorflow as tf
    from sionna.phy.utils import ebnodb2no
    from .config import Config
    from .csi import CSI
    from .rx import Rx

    def rand_cplx(shape, dtype=tf.float32):
        return tf.complex(tf.random.normal(shape, dtype=dtype),tf.random.normal(shape, dtype=dtype))

    # Setup
    cfg = Config(direction="downlink", perfect_csi=False)
    B = tf.constant(4, tf.int32)
    EbNo_dB = tf.constant(10.0, tf.float32)

    csi = CSI(cfg)
    csi.build(B)
    rx = Rx(cfg, csi)

    # dummy inputs
    if cfg.direction == "uplink":
        y = rand_cplx((B, 1, cfg.num_bs_ant, cfg.rg.num_ofdm_symbols, cfg.rg.fft_size))
        g = None
    else:  # downlink
        y = rand_cplx((B, cfg.rg.num_tx, cfg.rg.num_streams_per_tx, cfg.rg.num_ofdm_symbols, cfg.rg.fft_size))
        gl, gr = cfg.rg.num_guard_carriers
        n_sc_eff = cfg.rg.fft_size - gl - gr - 1# if cfg.rg.dc
        g = rand_cplx((B, cfg.rg.num_tx, cfg.rg.num_streams_per_tx, cfg.rg.num_tx,
                       cfg.rg.num_streams_per_tx, cfg.rg.num_ofdm_symbols, n_sc_eff))

    no = ebnodb2no(EbNo_dB, cfg.num_bits_per_symbol, cfg.coderate, cfg.rg)

    # Run & report
    out = rx(B, y, no, g)
    print("\n[RX] Output shapes:")
    for k, v in out.items():
        print(f"{k:10s}: {v.shape}")

