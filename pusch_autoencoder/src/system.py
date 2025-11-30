import numpy as np
import tensorflow as tf

from sionna.phy.channel import OFDMChannel, subcarrier_frequencies, cir_to_ofdm_channel, ApplyOFDMChannel
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.utils import ebnodb2no
from sionna.phy.ofdm import LinearDetector
from sionna.phy.mimo import StreamManagement

from .config import Config
from .pusch_trainable_transmitter import PUSCHTrainableTransmitter
from .pusch_neural_detector import PUSCHNeuralDetector
from .pusch_trainable_receiver import PUSCHTrainableReceiver

class PUSCHLinkE2E(tf.keras.Model):
    def __init__(self, channel_model, perfect_csi, use_autoencoder=False, training=False, training_mode="conventional"):
        super().__init__()

        self._training = training
        self._training_mode = training_mode
        self._rl_perturbation_variance = 0.01

        self._channel_model = channel_model
        self._perfect_csi = perfect_csi
        self._use_autoencoder = use_autoencoder

        # Central config (all hard-coded system parameters live in Config)
        self._cfg = Config()

        # System configuration from Config
        self._num_prb = self._cfg.num_prb
        self._mcs_index = self._cfg.mcs_index
        self._num_layers = self._cfg.num_layers
        self._mcs_table = self._cfg.mcs_table
        self._domain = self._cfg.domain

        # from Config
        self._num_ue_ant = self._cfg.num_ue_ant
        self._num_ue = self._cfg.num_ue
        self._subcarrier_spacing = self._cfg.subcarrier_spacing  # must be the same as used for Path2CIR

        # PUSCHConfig for the first transmitter
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = self._subcarrier_spacing / 1000.0
        pusch_config.carrier.n_size_grid = self._num_prb
        pusch_config.num_antenna_ports = self._num_ue_ant
        pusch_config.num_layers = self._num_layers
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 1
        pusch_config.dmrs.dmrs_port_set = list(range(self._num_layers))
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.tb.mcs_index = self._mcs_index
        pusch_config.tb.mcs_table = self._mcs_table

        # set Config's properties so that Neural-Detector can use them
        self._cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
        self._cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
        self._cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

        # Create PUSCHConfigs for the other transmitters
        pusch_configs = [pusch_config]
        for i in range(1, self._num_ue):
            pc = pusch_config.clone()
            pc.dmrs.dmrs_port_set = list(range(i * self._num_layers, (i + 1) * self._num_layers))
            pusch_configs.append(pc)

        # Create PUSCHTransmitter
        self._pusch_transmitter = (PUSCHTrainableTransmitter(pusch_configs, output_domain=self._domain, training=self._training, training_mode=self._training_mode)
                                   if self._use_autoencoder 
                                   else PUSCHTransmitter(pusch_configs, output_domain=self._domain))
        self._cfg.resource_grid = self._pusch_transmitter.resource_grid

        # Create PUSCHReceiver
        rx_tx_association = np.ones([1, self._num_ue], bool)
        stream_management = StreamManagement(rx_tx_association,self._num_layers)

        if self._use_autoencoder:
            detector = PUSCHNeuralDetector(self._cfg)
        else:
            detector = LinearDetector(
                equalizer="lmmse",
                output="bit",
                demapping_method="maxlog",
                resource_grid=self._pusch_transmitter.resource_grid,
                stream_management=stream_management,
                constellation_type="qam",
                num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol,
            )

        # configure receiver
        receiver = PUSCHTrainableReceiver if self._use_autoencoder else PUSCHReceiver
        receiver_kwargs = {"mimo_detector": detector,"input_domain": self._domain}

        # perfect/imperfect CSI
        if self._perfect_csi:
            receiver_kwargs["channel_estimator"] = "perfect"

        # enable/disable training
        if self._use_autoencoder:
            receiver_kwargs["training"] = training

        self._pusch_receiver = receiver(pusch_transmitter=self._pusch_transmitter, **receiver_kwargs)

        # configure differentiable channel for autoencoder, iterable channel for baseline
        if self._use_autoencoder:
            self._frequencies = subcarrier_frequencies(self._pusch_transmitter.resource_grid.fft_size, self._pusch_transmitter.resource_grid.subcarrier_spacing)
            self._channel = ApplyOFDMChannel(add_awgn=True)

            if self._training and self._training_mode == "conventional":
                self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self._channel = OFDMChannel(
                self._channel_model,
                self._pusch_transmitter.resource_grid,
                normalize_channel=True,
                return_channel=True,
            )

    @property
    def trainable_variables(self):
        vars_ = []
        if hasattr(self, "_pusch_transmitter"):
            vars_ += list(self._pusch_transmitter.trainable_variables)
        if hasattr(self, "_pusch_receiver"):
            vars_ += list(self._pusch_receiver.trainable_variables)
        return vars_
        
    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):

        if self._use_autoencoder:
            x_map, x_map_eps, x, b, c = self._pusch_transmitter(batch_size, perturbation_variance=self._rl_perturbation_variance)
        else:
            x, b = self._pusch_transmitter(batch_size)

        no = ebnodb2no(
            ebno_db,
            self._pusch_transmitter._num_bits_per_symbol,
            self._pusch_transmitter._target_coderate,
            self._pusch_transmitter.resource_grid,
        )

        if self._use_autoencoder:
            a, tau = self._channel_model
            num_samples = tf.shape(a)[0]

            # Randomly select batch_size number of indices
            idx = tf.random.shuffle(tf.range(num_samples))[:batch_size]

            # gather the corresponding CIRs
            a_batch  = tf.gather(a, idx, axis=0)
            tau_batch = tf.gather(tau, idx, axis=0)

            # cir to frequency-domain channel
            h = cir_to_ofdm_channel(self._frequencies, a_batch, tau_batch, normalize=True)

            y = self._channel(x, h, no)
        else:
            y, h = self._channel(x, no)

        if self._use_autoencoder and self._training:
            if self._training_mode == "conventional":
                if self._perfect_csi:
                    llr = self._pusch_receiver(y, no, h)
                else:
                    llr = self._pusch_receiver(y, no)
                loss = self._bce(c, llr)
                return loss
            elif self._training_mode == "rl":
                # stop gradient
                y = tf.stop_gradient(y)

                if self._perfect_csi:
                    llr = self._pusch_receiver(y, no, h)
                else:
                    llr = self._pusch_receiver(y, no)
                bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(c, llr), axis=-1) # average over the bits mapped to a same baseband symbol
                rx_loss = tf.reduce_mean(bce) # Rx loss is the usual average BCE
                # on the Tx side, the BCE is seen as a feedback from the Rx through which backpropagation is not possible
                bce = tf.stop_gradient(bce)
                x_map_eps = tf.stop_gradient(x_map_eps)
                p = x_map_eps-x_map
                tx_loss = tf.square(tf.math.real(p)) + tf.square(tf.math.imag(p))
                tx_loss = -bce[..., tf.newaxis] * tx_loss / self._rl_perturbation_variance
                tx_loss = tf.reduce_mean(tx_loss)
                return tx_loss, rx_loss
        else:
            if self._perfect_csi:
                b_hat = self._pusch_receiver(y, no, h)
            else:
                b_hat = self._pusch_receiver(y, no)
            return b, b_hat
