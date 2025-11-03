"""
Single-iteration simulation using shared CSI:
Tx -> Channel -> Rx
"""

import tensorflow as tf
from sionna.phy.utils import ebnodb2no, compute_ber
from src import Config, CSI, Tx, Channel, Rx

# config
cfg = Config(direction="downlink", perfect_csi=False)

# constants
B = tf.constant(8, dtype=tf.int32)
EbNo_dB = tf.constant(40.0)
no = ebnodb2no(EbNo_dB, cfg.num_bits_per_symbol, cfg.coderate, cfg.rg)

# CSI object
csi = CSI(cfg, batch_size=B)

# create tx, channel and rx objects
tx = Tx(cfg, csi)
ch = Channel(cfg, csi)
rx = Rx(cfg, csi)

# invoke tx, channel, and rx call() methods
tx_out = tx(B)
y_out = ch(B, tx_out["x_rg_tx"], no)
rx_out = rx(B, y_out["y"], no, g=tx_out["g"])

# BER
ber = compute_ber(tx_out['b'], rx_out['b_hat'])
print("BER: {}".format(ber))
