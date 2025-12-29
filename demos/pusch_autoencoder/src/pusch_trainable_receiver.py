import tensorflow as tf
from sionna.phy.channel import time_to_ofdm_channel
from sionna.phy.nr import PUSCHReceiver


class PUSCHTrainableReceiver(PUSCHReceiver):
    """
    PUSCHReceiver variant that:
    1. Returns LLRs before TB decoding in training mode
    2. Passes trainable constellation to the neural detector for proper demapping
    """

    def __init__(self, *args, training=False, pusch_transmitter=None, **kwargs):
        self._training = training
        self._pusch_transmitter = pusch_transmitter

        # Pass pusch_transmitter to parent
        super().__init__(*args, pusch_transmitter=pusch_transmitter, **kwargs)

    @property
    def trainable_variables(self):
        if hasattr(self._mimo_detector, "trainable_variables"):
            return self._mimo_detector.trainable_variables
        return []

    def _get_normalized_constellation(self):
        """Get normalized constellation points from transmitter."""
        if self._pusch_transmitter is not None:
            return self._pusch_transmitter.get_normalized_constellation()
        return None

    def call(self, y, no, h=None):
        # (Optional) OFDM Demodulation
        if self._input_domain == "time":
            y = self._ofdm_demodulator(y)

        # Channel estimation
        if self._perfect_csi:
            if self._input_domain == "time":
                h = time_to_ofdm_channel(h, self.resource_grid, self._l_min)

            if self._w is not None:
                h = tf.transpose(h, perm=[0, 1, 3, 5, 6, 2, 4])
                h = tf.matmul(h, self._w)
                h = tf.transpose(h, perm=[0, 1, 5, 2, 6, 3, 4])
            h_hat = h
            err_var = tf.zeros_like(tf.math.real(h_hat[:, :1, :1, :, :, :, :]))
        else:
            h_hat, err_var = self._channel_estimator(y, no)

        # MIMO Detection - pass constellation for proper demapping
        constellation = self._get_normalized_constellation()
        if constellation is not None:
            llr = self._mimo_detector(
                y, h_hat, err_var, no, constellation=constellation
            )
        else:
            llr = self._mimo_detector(y, h_hat, err_var, no)

        # Layer demapping
        llr = self._layer_demapper(llr)

        if self._training:
            # Return LLRs for training
            if len(llr.shape) == 4 and llr.shape[2] == 1:
                llr = tf.squeeze(llr, axis=2)
            return llr
        else:
            # TB Decoding for inference
            b_hat, tb_crc_status = self._tb_decoder(llr)
            if self._return_tb_crc_status:
                return b_hat, tb_crc_status
            else:
                return b_hat
