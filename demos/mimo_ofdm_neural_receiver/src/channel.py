import tensorflow as tf
from typing import Dict, Any
from sionna.phy.channel import ApplyOFDMChannel


class Channel:
    """
    Apply the frequency-domain channel with AWGN.
    """

    def __init__(self):
        self._apply = ApplyOFDMChannel(add_awgn=True)

    @tf.function
    def __call__(
        self, x_rg_tx: tf.Tensor, h_freq: tf.Tensor, no: tf.Tensor
    ) -> Dict[str, Any]:
        y = self._apply(x_rg_tx, h_freq, no)
        return {"y": y}
