from sionna.phy.nr import PUSCHTransmitter
from sionna.phy.mapping import Mapper, Constellation
import tensorflow as tf


class PUSCHTrainableTransmitter(PUSCHTransmitter):
    """
    PUSCH Transmitter with trainable constellation.
    
    Uses explicit tf.Variable for constellation points (real/imag) with
    manual normalization to ensure unit power is maintained during training.
    """
    
    def __init__(self, *args, training=False, **kwargs):
        self._training = training
        
        # parent constructor
        super().__init__(*args, **kwargs)
        
        self._setup_custom_constellation()
    
    @property
    def trainable_variables(self):
        """Return the trainable constellation variables."""
        return [self._points_r, self._points_i]
    
    def get_normalized_constellation(self):
        """
        Returns the centered and normalized constellation points.
        
        Matches Sionna's Constellation.call() behavior:
        1. Center: subtract mean
        2. Normalize: divide by sqrt(mean energy) for unit power
        """
        points = tf.complex(self._points_r, self._points_i)
        
        # Center (subtract mean)
        points = points - tf.reduce_mean(points)
        
        # Normalize to unit power
        energy = tf.reduce_mean(tf.square(tf.abs(points)))
        points = points / tf.cast(tf.sqrt(energy), points.dtype)
        
        return points

    def _setup_custom_constellation(self):
        """Setup trainable constellation with explicit tf.Variables."""
        # Original QAM constellation used as initialization
        qam_points = Constellation("qam", num_bits_per_symbol=self._num_bits_per_symbol).points

        # Trainable real/imag parts as tf.Variables
        init_r = tf.math.real(qam_points)
        init_i = tf.math.imag(qam_points)

        self._points_r = tf.Variable(
            tf.cast(init_r, self.rdtype),
            trainable=self._training,
            name="constellation_real"
        )
        self._points_i = tf.Variable(
            tf.cast(init_i, self.rdtype),
            trainable=self._training,
            name="constellation_imag"
        )

        # Custom constellation - we'll update points in call() with normalization
        self._constellation = Constellation(
            "custom",
            num_bits_per_symbol=self._num_bits_per_symbol,
            points=tf.complex(self._points_r, self._points_i),
            normalize=False,  # We handle normalization manually
            center=False      # We handle centering manually if needed
        )

        # Replace the mapper to use our trainable constellation
        self._mapper = Mapper(constellation=self._constellation)

    
    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : int or [batch_size, num_layers, tb_size], tf.float32
            Either batch_size (if return_bits=True) or bits tensor (if return_bits=False)
        """
        # Update constellation with normalized points (unit power)
        self._constellation.points = self.get_normalized_constellation()
        
        if self._return_bits:
            # inputs defines batch_size
            batch_size = inputs
            b = self._binary_source([batch_size, self._num_tx, self._tb_size])
        else:
            b = inputs

        # Encode transport block
        c = self._tb_encoder(b)

        # Map to constellations
        x_map = self._mapper(c)

        # Map to layers
        x_layer = self._layer_mapper(x_map)

        # Apply resource grid mapping
        x_grid = self._resource_grid_mapper(x_layer)

        # (Optionally) apply PUSCH precoding
        if self._precoding == "codebook":
            x_pre = self._precoder(x_grid)
        else:
            x_pre = x_grid

        # (Optionally) apply OFDM modulation
        if self._output_domain == "time":
            x = self._ofdm_modulator(x_pre)
        else:
            x = x_pre

        if self._return_bits:
            return x_map, x, b, c
        else:
            return x