import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import leaky_relu
from sionna.phy.mimo import lmmse_equalizer
from sionna.phy.mapping import Demapper, Constellation
from sionna.phy.utils import log10

from .config import Config


class Conv2DResBlock(Layer):
    """Residual block with two convolutions."""

    def __init__(self, filters: int, kernel_size=(3, 3), name: str = None):
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = kernel_size
        self._layer_norm1 = LayerNormalization(axis=-1)
        self._conv1 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding="same")
        self._layer_norm2 = LayerNormalization(axis=-1)
        self._conv2 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding="same")

    def call(self, x):
        y = self._layer_norm1(x)
        y = leaky_relu(y, alpha=0.1)
        y = self._conv1(y)
        y = self._layer_norm2(y)
        y = leaky_relu(y, alpha=0.1)
        y = self._conv2(y)
        return x + y


class PUSCHNeuralDetector(Layer):
    """
    Neural MIMO detector with Sionna LMMSE + Demapper and residual learning.
    
    Uses Sionna's lmmse_equalizer and Demapper directly, with a neural network
    that learns corrections to the LMMSE LLRs.
    """
    
    def __init__(
        self,
        cfg: Config,
        num_conv2d_filters: int = 128,
        num_res_blocks: int = 6,
        kernel_size=(3, 3),
    ):
        super().__init__()
        self._cfg = cfg
        self.num_conv2d_filters = int(num_conv2d_filters)
        self.num_res_blocks = int(num_res_blocks)
        self.kernel_size = kernel_size

        self._num_bits_per_symbol = int(self._cfg.num_bits_per_symbol)
        self._num_ue = int(self._cfg.num_ue)
        self._num_streams_per_ue = int(self._cfg.num_layers)
        self._num_streams_total = self._num_ue * self._num_streams_per_ue
        self._num_bs = int(self._cfg.num_bs)
        self._num_bs_ant = int(self._cfg.num_bs_ant)
        self._num_rx_ant = self._num_bs * self._num_bs_ant
        self._pusch_pilot_indices = list(self._cfg.pusch_pilot_indices)
        self._pusch_num_symbols_per_slot = int(self._cfg.pusch_num_symbols_per_slot)

        all_symbols = list(range(self._pusch_num_symbols_per_slot))
        pilots = set(self._pusch_pilot_indices)
        self._data_symbol_indices = [s for s in all_symbols if s not in pilots]

        # Create default 16-QAM constellation as "custom" type so points can be updated
        # Initialize with standard QAM points
        qam_points = Constellation("qam", self._num_bits_per_symbol).points
        self._constellation = Constellation(
            "custom",
            num_bits_per_symbol=self._num_bits_per_symbol,
            points=qam_points,
            normalize=False,  # Points from transmitter are already normalized
        )
        
        # Sionna's Demapper - uses constellation.points at call time
        self._demapper = Demapper("maxlog", constellation=self._constellation)

        # Input features for refinement network
        self._c_in = (
            2 * self._num_streams_total +    # LMMSE estimates (real/imag)
            self._num_streams_total +        # no_eff from LMMSE
            2 * self._num_streams_total +    # Matched filter (real/imag)
            self._num_streams_total +        # Gram diagonal
            self._num_streams_total * (self._num_streams_total - 1) +  # Gram off-diag
            self._num_streams_total +        # err_var
            1                                # noise
        )

        # Refinement network
        self._conv2d_in = Conv2D(
            filters=self.num_conv2d_filters,
            kernel_size=(3, 3),
            padding="same",
            name="conv_in",
        )

        self._res_blocks = [
            Conv2DResBlock(
                filters=self.num_conv2d_filters,
                kernel_size=self.kernel_size,
                name=f"conv_resblock_{i}",
            )
            for i in range(self.num_res_blocks)
        ]

        self._conv2d_out = Conv2D(
            filters=self._num_streams_total * self._num_bits_per_symbol,
            kernel_size=(3, 3),
            padding="same",
            activation=None,
            name="conv_out",
        )
        
        # Learnable scaling for correction
        self._correction_scale = tf.Variable(
            0.3, trainable=True, name="correction_scale", dtype=tf.float32
        )

    @property
    def trainable_variables(self):
        vars_ = []
        vars_ += self._conv2d_in.trainable_variables
        for block in self._res_blocks:
            vars_ += block.trainable_variables
        vars_ += self._conv2d_out.trainable_variables
        vars_ += [self._correction_scale]
        return vars_

    def _reshape_logits_to_llr(self, logits, num_data_symbols):
        """Reshape [B, H, W, S*bits] -> [B, num_ue, streams_per_ue, num_data_symbols*bits]"""
        B = tf.shape(logits)[0]
        logits = tf.reshape(
            logits, [B, num_data_symbols, self._num_streams_total, self._num_bits_per_symbol]
        )
        logits = tf.reshape(
            logits, [B, num_data_symbols, self._num_ue, self._num_streams_per_ue, self._num_bits_per_symbol]
        )
        logits = tf.transpose(logits, [0, 2, 3, 1, 4])
        llr = tf.reshape(
            logits, [B, self._num_ue, self._num_streams_per_ue, num_data_symbols * self._num_bits_per_symbol]
        )
        llr.set_shape([None, self._num_ue, self._num_streams_per_ue, None])
        return llr

    def call(self, y, h_hat, err_var, no, constellation=None, training=None):
        """
        Neural MIMO detector.
        
        Args:
            y: [B, num_bs, num_bs_ant, num_ofdm_symbols, num_subcarriers] complex
            h_hat: [B, num_bs, num_bs_ant, num_ue, num_streams, num_ofdm_symbols, num_subcarriers] complex
            err_var: [B, 1, 1, num_ue, num_streams, num_ofdm_symbols, num_subcarriers] float
            no: [B] or scalar, noise variance
            constellation: [num_points] complex, optional trainable constellation points
        """
        y = tf.cast(y, tf.complex64)
        h_hat = tf.cast(h_hat, tf.complex64)
        err_var = tf.cast(err_var, tf.float32)
        no = tf.cast(no, tf.float32)

        # Update constellation if trainable points provided
        if constellation is not None:
            self._constellation.points = tf.cast(constellation, tf.complex64)

        # Mask pilots
        data_idx = tf.constant(self._data_symbol_indices, dtype=tf.int32)
        y = tf.gather(y, data_idx, axis=3)
        h_hat = tf.gather(h_hat, data_idx, axis=5)
        err_var = tf.gather(err_var, data_idx, axis=5)

        B = tf.shape(y)[0]
        H = tf.shape(y)[3]
        W = tf.shape(y)[4]
        num_data_symbols = H * W

        # Reshape inputs for LMMSE: need [B, H, W, ...] format
        # y: [B, num_bs, num_bs_ant, H, W] -> [B, H, W, num_rx_ant]
        y_flat = tf.reshape(y, [B, -1, H, W])
        y_flat = tf.transpose(y_flat, [0, 2, 3, 1])
        
        # h_hat: [B, num_bs, num_bs_ant, num_ue, streams, H, W] -> [B, H, W, num_rx_ant, num_streams]
        h_flat = tf.reshape(h_hat, [B, -1, self._num_streams_total, H, W])
        h_flat = tf.transpose(h_flat, [0, 3, 4, 1, 2])
        
        # err_var: [B, 1, 1, num_ue, streams, H, W] -> [B, H, W, num_streams]
        err_var_t = tf.squeeze(err_var, axis=[1, 2])
        err_var_flat = tf.transpose(err_var_t, [0, 3, 4, 1, 2])
        err_var_flat = tf.reshape(err_var_flat, [B, H, W, self._num_streams_total])

        # Build noise covariance matrix S for LMMSE
        no_expanded = no[:, tf.newaxis, tf.newaxis]
        avg_err_var = tf.reduce_mean(err_var_flat, axis=-1)
        total_noise_var = no_expanded + avg_err_var
        eye = tf.eye(self._num_rx_ant, dtype=tf.complex64)
        eye = eye[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
        s = tf.cast(total_noise_var[..., tf.newaxis, tf.newaxis], tf.complex64) * eye

        # === LMMSE using Sionna ===
        x_lmmse, no_eff = lmmse_equalizer(y_flat, h_flat, s, whiten_interference=True)
        
        # === Demapping using Sionna's Demapper ===
        # Flatten all dims: [B, H, W, S] -> [B*H*W*S]
        x_lmmse_flat = tf.reshape(x_lmmse, [-1])
        no_eff_flat = tf.reshape(no_eff, [-1])
        
        # Call demapper once on all symbols
        # Output: [B*H*W*S, num_bits_per_symbol]
        llr_lmmse_flat = self._demapper(x_lmmse_flat, no_eff_flat)
        
        # Reshape back: [B*H*W*S, bits] -> [B, H, W, S, bits] -> [B, H, W, S*bits]
        llr_lmmse = tf.reshape(llr_lmmse_flat, [B, H, W, self._num_streams_total, self._num_bits_per_symbol])
        llr_lmmse = tf.reshape(llr_lmmse, [B, H, W, self._num_streams_total * self._num_bits_per_symbol])

        # === Build features for refinement network ===
        # Compute matched filter and Gram for additional features
        h_conj_t = tf.transpose(tf.math.conj(h_flat), [0, 1, 2, 4, 3])
        y_col = y_flat[..., tf.newaxis]
        z_mf = tf.squeeze(tf.matmul(h_conj_t, y_col), axis=-1)
        gram = tf.matmul(h_conj_t, h_flat)
        
        # Feature assembly
        x_lmmse_feats = tf.concat([tf.math.real(x_lmmse), tf.math.imag(x_lmmse)], axis=-1)
        no_eff_feats = tf.math.log(no_eff + 1e-10)
        z_mf_feats = tf.concat([tf.math.real(z_mf), tf.math.imag(z_mf)], axis=-1)
        gram_diag = tf.math.real(tf.linalg.diag_part(gram))
        
        # Gram off-diagonal
        mask = 1.0 - tf.eye(self._num_streams_total, dtype=tf.float32)
        mask = mask[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
        gram_masked = gram * tf.cast(mask, gram.dtype)
        gram_offdiag = tf.abs(gram_masked)
        gram_offdiag_flat = tf.reshape(gram_offdiag, [B, H, W, -1])
        indices = [i * self._num_streams_total + j 
                   for i in range(self._num_streams_total) 
                   for j in range(self._num_streams_total) if i != j]
        gram_offdiag_feats = tf.gather(gram_offdiag_flat, indices, axis=-1)
        
        no_log = log10(no + 1e-10)
        no_expanded_feat = tf.broadcast_to(no_log[:, tf.newaxis, tf.newaxis, tf.newaxis], [B, H, W, 1])
        
        z = tf.concat([
            x_lmmse_feats, no_eff_feats, z_mf_feats,
            gram_diag, gram_offdiag_feats, err_var_flat, no_expanded_feat,
        ], axis=-1)
        z = tf.cast(z, tf.float32)

        # === Refinement network ===
        z_feat = self._conv2d_in(z)
        for block in self._res_blocks:
            z_feat = block(z_feat)
        llr_correction = self._conv2d_out(z_feat)
        
        # === Final LLR = LMMSE LLR + learned correction ===
        llr_final = llr_lmmse + self._correction_scale * llr_correction

        return self._reshape_logits_to_llr(llr_final, num_data_symbols)