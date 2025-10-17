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
    def __init__(self):
        self._apply = ApplyOFDMChannel(add_awgn=True)

    @tf.function
    def __call__(self, x_rg_tx: tf.Tensor, h_freq: tf.Tensor, no: tf.Tensor) -> Dict[str, Any]:
        y = self._apply(x_rg_tx, h_freq, no)
        return {"y": y}


if __name__ == "__main__":
    """
    Example standalone test for Channel stage.
    Simulates channel application on random input with cached CSI.h_freq.
    """
    from .csi import CSI
    from sionna.phy.utils import ebnodb2no

    cfg = Config(direction="uplink")
    B = tf.constant(4, dtype=tf.int32)
    EbNo_dB = tf.constant(10.0)

    csi = CSI(cfg)
    h_freq = csi.build(B)

    channel = Channel()

    # Generate dummy transmitted resource grid
    x_shape = (B, cfg.rg.num_tx, cfg.rg.num_ofdm_symbols, cfg.rg.fft_size)
    x_rg_tx = tf.complex(tf.random.normal(x_shape, dtype=tf.float32),tf.random.normal(x_shape, dtype=tf.float32))
    no = ebnodb2no(EbNo_dB, cfg.num_bits_per_symbol, cfg.coderate, cfg.rg)

    y = channel(x_rg_tx, h_freq, no)
    print("\n[CHANNEL] Output shapes:")
    for k, v in y.items():
        print(f"{k:10s}: shape={v.shape}")