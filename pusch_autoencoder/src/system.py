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
    def __init__(self, channel_model, perfect_csi, use_autoencoder=False, training=False, 
                 const_reg_weight=0.1, const_d_min=0.25):
        super().__init__()

        self._training = training

        # Constellation regularization parameters
        self._const_reg_weight = const_reg_weight
        self._const_d_min = const_d_min

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
        self._pusch_transmitter = (PUSCHTrainableTransmitter(pusch_configs, output_domain=self._domain, training=self._training)
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
        receiver_kwargs = {"mimo_detector": detector,"input_domain": self._domain, "pusch_transmitter": self._pusch_transmitter}

        # perfect/imperfect CSI
        if self._perfect_csi:
            receiver_kwargs["channel_estimator"] = "perfect"

        # enable/disable training and pass _pusch_transmitter
        if self._use_autoencoder:
            receiver_kwargs["training"] = training
        
        self._pusch_receiver = receiver(**receiver_kwargs)

        # configure differentiable channel for autoencoder, iterable channel for baseline
        if self._use_autoencoder:
            self._frequencies = subcarrier_frequencies(self._pusch_transmitter.resource_grid.fft_size, self._pusch_transmitter.resource_grid.subcarrier_spacing)
            self._channel = ApplyOFDMChannel(add_awgn=True)

            if self._training:
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
    
    @property
    def constellation(self):
        """Returns normalized constellation points (complex) from transmitter."""
        return self._pusch_transmitter.get_normalized_constellation()
    
    def get_constellation_min_distance(self):
        """Returns the minimum pairwise distance in the constellation (for monitoring)."""
        points = tf.stack([tf.math.real(self.constellation), 
                          tf.math.imag(self.constellation)], axis=-1)
        diff = points[:, tf.newaxis, :] - points[tf.newaxis, :, :]
        distances = tf.norm(diff, axis=-1)
        # Mask diagonal
        mask = 1.0 - tf.eye(tf.shape(points)[0])
        distances = distances + (1.0 - mask) * 1e6
        return tf.reduce_min(distances)
        
    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):

        if self._use_autoencoder:
            x_map, x, b, c = self._pusch_transmitter(batch_size)
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
            if self._perfect_csi:
                llr = self._pusch_receiver(y, no, h)
            else:
                llr = self._pusch_receiver(y, no)
            
            # Detection loss (BCE)
            bce_loss = self._bce(c, llr)
            
            # Constellation regularization loss to prevent collapse
            const_reg_loss = constellation_regularization_loss(
                tf.math.real(self.constellation),
                tf.math.imag(self.constellation),
                d_min_weight=1.0,
                grid_weight=0.0,
                uniformity_weight=0.0,
                d_min=self._const_d_min,
            )
            
            # Combined loss
            loss = bce_loss + self._const_reg_weight * const_reg_loss
            return loss
        else:
            if self._perfect_csi:
                b_hat = self._pusch_receiver(y, no, h)
            else:
                b_hat = self._pusch_receiver(y, no)
            return b, b_hat
        
def min_distance_loss(points_r, points_i, d_min=0.4, margin=0.1):
    """
    Penalizes constellation points that are closer than d_min.
    
    This is the most effective regularizer to prevent collapse.
    For 16-QAM with unit average power, the nominal minimum distance
    is approximately 2/sqrt(10) ≈ 0.632. Setting d_min=0.4 gives some
    flexibility while preventing full collapse to QPSK.
    
    Args:
        points_r: [num_points] real parts of constellation
        points_i: [num_points] imaginary parts of constellation
        d_min: minimum allowed distance between any two points
        margin: soft margin for the hinge loss
        
    Returns:
        Scalar loss value
    """
    points = tf.stack([points_r, points_i], axis=-1)  # [N, 2]
    
    # Compute pairwise distances
    # diff[i,j] = points[i] - points[j]
    diff = points[:, tf.newaxis, :] - points[tf.newaxis, :, :]  # [N, N, 2]
    distances = tf.norm(diff, axis=-1)  # [N, N]
    
    # Mask out diagonal (self-distances)
    mask = 1.0 - tf.eye(tf.shape(points)[0])
    distances = distances * mask + tf.eye(tf.shape(points)[0]) * 1e6  # Large value on diagonal
    
    # Hinge loss: penalize distances below d_min
    violations = tf.nn.relu(d_min + margin - distances)
    
    # Average over all pairs (excluding diagonal)
    num_pairs = tf.cast(tf.shape(points)[0] * (tf.shape(points)[0] - 1), tf.float32)
    loss = tf.reduce_sum(violations) / num_pairs
    
    return loss


def grid_structure_loss(points_r, points_i):
    """
    Encourages constellation to maintain a grid-like structure.
    
    For 16-QAM, we have 4 levels per dimension: {-3, -1, +1, +3} (normalized).
    This loss encourages points to cluster near these grid positions.
    
    Args:
        points_r: [num_points] real parts
        points_i: [num_points] imaginary parts  
        
    Returns:
        Scalar loss value
    """
    # Target grid levels (normalized for unit power 16-QAM)
    # Standard 16-QAM: {-3, -1, 1, 3} / sqrt(10)
    scale = tf.sqrt(10.0)
    grid_levels = tf.constant([-3.0, -1.0, 1.0, 3.0], dtype=tf.float32) / scale
    
    def snap_to_grid_loss(coords):
        # For each coordinate, find distance to nearest grid level
        coords_expanded = coords[:, tf.newaxis]  # [N, 1]
        grid_expanded = grid_levels[tf.newaxis, :]  # [1, 4]
        distances = tf.abs(coords_expanded - grid_expanded)  # [N, 4]
        min_distances = tf.reduce_min(distances, axis=-1)  # [N]
        return tf.reduce_mean(min_distances)
    
    loss_r = snap_to_grid_loss(points_r)
    loss_i = snap_to_grid_loss(points_i)
    
    return loss_r + loss_i


def uniformity_loss(points_r, points_i):
    """
    Encourages uniform spacing by maximizing the minimum distance.
    
    This is a softer version that pushes points apart without
    requiring a specific structure.
    
    Args:
        points_r: [num_points] real parts
        points_i: [num_points] imaginary parts
        
    Returns:
        Scalar loss (negative, to be minimized)
    """
    points = tf.stack([points_r, points_i], axis=-1)  # [N, 2]
    
    diff = points[:, tf.newaxis, :] - points[tf.newaxis, :, :]
    distances = tf.norm(diff, axis=-1)
    
    # Mask diagonal
    mask = 1.0 - tf.eye(tf.shape(points)[0])
    distances = distances + (1.0 - mask) * 1e6
    
    # We want to maximize the minimum distance
    # Equivalent to minimizing negative min distance
    # Use softmin for differentiability
    temperature = 0.1
    softmin = -temperature * tf.reduce_logsumexp(-distances / temperature)
    
    return -softmin  # Negative because we want to maximize min distance


def constellation_regularization_loss(points_r, points_i, 
                                       d_min_weight=1.0,
                                       grid_weight=0.0,
                                       uniformity_weight=0.0,
                                       d_min=0.4):
    """
    Combined constellation regularization loss.
    
    Recommended usage:
    - For preventing collapse: use d_min_weight=1.0, others=0
    - For encouraging QAM structure: add grid_weight=0.5
    - For general spreading: use uniformity_weight=0.5
    
    Args:
        points_r: [num_points] real parts of constellation
        points_i: [num_points] imaginary parts of constellation
        d_min_weight: weight for minimum distance loss
        grid_weight: weight for grid structure loss
        uniformity_weight: weight for uniformity loss
        d_min: minimum distance threshold
        
    Returns:
        Combined scalar loss
    """
    total_loss = 0.0
    
    if d_min_weight > 0:
        total_loss += d_min_weight * min_distance_loss(points_r, points_i, d_min=d_min)
    
    if grid_weight > 0:
        total_loss += grid_weight * grid_structure_loss(points_r, points_i)
    
    if uniformity_weight > 0:
        total_loss += uniformity_weight * uniformity_loss(points_r, points_i)
    
    return total_loss