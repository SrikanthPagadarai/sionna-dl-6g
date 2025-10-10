import tensorflow as tf
from typing import Dict, Any
from sionna.phy.channel import ApplyOFDMChannel
from .config import Config
from .csi import CSI

class Channel:
    """
    Apply the frequency-domain channel with AWGN.
    Uses the *shared* CSI.h_freq (same tensor seen by Tx/Rx).
    """
    def __init__(self, cfg: Config, csi: CSI):
        self.cfg = cfg.build()
        self.csi = csi
        self.rg = self.csi.rg
        self._apply = ApplyOFDMChannel(add_awgn=True)

    @tf.function
    def __call__(self, batch_size: tf.Tensor, x_rg_tx: tf.Tensor, no: tf.Tensor) -> Dict[str, Any]:
        self.csi.assert_batch(batch_size)
        y = self._apply(x_rg_tx, self.csi.h_freq, no)
        return {"y": y}


if __name__ == "__main__":
    """
    Example standalone test for Channel stage.
    Simulates channel application on random input with cached CSI.h_freq.
    """
    from .csi import CSI
    from sionna.phy.utils import ebnodb2no

    cfg = Config(direction="downlink")
    B = tf.constant(4, dtype=tf.int32)
    EbNo_dB = tf.constant(10.0)

    csi = CSI(cfg, batch_size=B)
    n_tx  = csi.rg.num_tx
    n_sym = csi.rg.num_ofdm_symbols
    n_sc  = csi.rg.fft_size

    channel = Channel(cfg, csi)

    # Generate dummy transmitted resource grid
    x_shape = (B, n_tx, n_sym, n_sc)
    x_rg_tx = tf.complex(tf.random.normal(x_shape, dtype=tf.float32),tf.random.normal(x_shape, dtype=tf.float32))
    no = ebnodb2no(EbNo_dB, cfg.num_bits_per_symbol, cfg.coderate, cfg.rg)

    y = channel(B, x_rg_tx, no)
    print("\n[CHANNEL] Output shapes:")
    for k, v in y.items():
        print(f"{k:10s}: shape={v.shape}")