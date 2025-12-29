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
        self._conv1 = Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, padding="same"
        )
        self._layer_norm2 = LayerNormalization(axis=-1)
        self._conv2 = Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, padding="same"
        )

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
    LS channel estimator refinement using a NN
    followed by
    traditional LMMSE detection and demapping using refined channel estimator
    followed by
    LLR refinement using a NN

    Architecture:
        1. Shared backbone:
           processes input features (h_ls, y, z_mf, gram, err_var, no)
        2. CE head:
           lightweight projection from shared features -> delta_h, delta_loge
        3. Apply scaled corrections:
           h_refined = h + scale_h * delta_h
           err_var_refined = exp(log(err_var) + scale_e * delta_loge)
        4. LMMSE:
           equalization using refined (h, err_var)
        5. Detection refinement:
           inject LMMSE output into shared features -> llr_correction
        6. Final LLR:
           llr_lmmse + scale_llr * llr_correction

    Trainable correction scales:
        - _h_correction_scale: unbounded, controls channel estimate refinement
        - _err_var_correction_scale_raw: softplus-transformed to ensure positive
        - _llr_correction_scale: unbounded, controls LLR refinement
    """

    def __init__(
        self,
        cfg: Config,
        num_conv2d_filters: int = 128,
        num_shared_res_blocks: int = 4,
        num_det_res_blocks: int = 6,
        kernel_size=(3, 3),
    ):
        super().__init__()
        self._cfg = cfg
        self.num_conv2d_filters = int(num_conv2d_filters)
        self.num_shared_res_blocks = int(num_shared_res_blocks)
        self.num_det_res_blocks = int(num_det_res_blocks)
        self.kernel_size = kernel_size

        # Dimensions from config
        self._num_bits_per_symbol = int(self._cfg.num_bits_per_symbol)
        self._num_ue = int(self._cfg.num_ue)
        self._num_streams_per_ue = int(self._cfg.num_layers)
        self._num_streams_total = self._num_ue * self._num_streams_per_ue
        self._num_bs = int(self._cfg.num_bs)
        self._num_bs_ant = int(self._cfg.num_bs_ant)
        self._num_rx_ant = self._num_bs * self._num_bs_ant
        self._pusch_pilot_indices = list(self._cfg.pusch_pilot_indices)
        self._pusch_num_symbols_per_slot = int(self._cfg.pusch_num_symbols_per_slot)

        # Data symbol indices (excluding pilots)
        all_symbols = list(range(self._pusch_num_symbols_per_slot))
        pilots = set(self._pusch_pilot_indices)
        self._data_symbol_indices = [s for s in all_symbols if s not in pilots]

        # Constellation + demapper
        qam_points = Constellation("qam", self._num_bits_per_symbol).points
        self._constellation = Constellation(
            "custom",
            num_bits_per_symbol=self._num_bits_per_symbol,
            points=qam_points,
            normalize=False,
        )
        self._demapper = Demapper("maxlog", constellation=self._constellation)

        # ===== Trainable correction scales =====
        # h and llr scales: unbounded, initialized to 0.0
        self._h_correction_scale = tf.Variable(
            0.0, trainable=True, name="h_correction_scale", dtype=tf.float32
        )
        self._llr_correction_scale = tf.Variable(
            0.0, trainable=True, name="llr_correction_scale", dtype=tf.float32
        )
        # err_var scale: softplus-transformed to ensure positive
        # Initialize x value so that softplus(x) = 1.0
        # softplus(x) = log(1 + exp(x)), i.e., softplus(x) = 1.0 gives x = 0.0
        self._err_var_correction_scale_raw = tf.Variable(
            0.0, trainable=True, name="err_var_correction_scale_raw", dtype=tf.float32
        )

        # ===== Input feature dimensions =====
        S = self._num_streams_total
        Nr = self._num_rx_ant

        self._c_in_shared = (
            2 * Nr * S  # h_ls real/imag
            + 2 * Nr  # y real/imag
            + 2 * S  # z_mf real/imag
            + S  # gram_diag
            + S * (S - 1)  # gram_offdiag
            + S  # err_var
            + 1  # noise (log)
        )

        # ===== Shared Backbone =====
        self._shared_conv_in = Conv2D(
            filters=self.num_conv2d_filters,
            kernel_size=(3, 3),
            padding="same",
            name="shared_conv_in",
        )
        self._shared_res_blocks = [
            Conv2DResBlock(
                filters=self.num_conv2d_filters,
                kernel_size=self.kernel_size,
                name=f"shared_resblock_{i}",
            )
            for i in range(self.num_shared_res_blocks)
        ]

        # CE Head NN
        self._ce_head_conv1 = tf.keras.Sequential(
            [
                Conv2D(self.num_conv2d_filters, (3, 3), padding="same"),
                LayerNormalization(axis=-1),
                tf.keras.layers.LeakyReLU(0.1),
                Conv2D(self.num_conv2d_filters, (3, 3), padding="same"),
                LayerNormalization(axis=-1),
                tf.keras.layers.LeakyReLU(0.1),
            ],
            name="ce_head_conv1",
        )
        # delta_h output: real/imag for all (rx_ant, stream) entries
        self._ce_head_out_h = Conv2D(
            filters=2 * Nr * S,
            kernel_size=(1, 1),
            padding="same",
            activation=None,
            name="ce_head_out_h",
        )
        # delta_log_err output: per-stream log-domain update
        self._ce_head_out_loge = Conv2D(
            filters=S,
            kernel_size=(1, 1),
            padding="same",
            activation=None,
            name="ce_head_out_loge",
        )

        # ===== Detection Continuation =====
        self._c_lmmse_feats = (
            2 * S  # x_lmmse real/imag
            + S  # no_eff (log)
            + S * self._num_bits_per_symbol  # llr_lmmse
        )

        # Injection conv: combines shared features + LMMSE features
        self._det_inject_conv = Conv2D(
            filters=self.num_conv2d_filters,
            kernel_size=(3, 3),
            padding="same",
            name="det_inject_conv",
        )

        # Detection continuation ResBlocks
        self._det_res_blocks = [
            Conv2DResBlock(
                filters=self.num_conv2d_filters,
                kernel_size=self.kernel_size,
                name=f"det_resblock_{i}",
            )
            for i in range(self.num_det_res_blocks)
        ]

        # Output conv for LLR correction
        self._det_conv_out = Conv2D(
            filters=S * self._num_bits_per_symbol,
            kernel_size=(3, 3),
            padding="same",
            activation=None,
            name="det_conv_out",
        )

        # Exposed for auxiliary losses during training
        self.last_h_hat_refined = None
        self.last_err_var_refined = None
        self.last_err_var_refined_flat = None

    @property
    def err_var_correction_scale(self):
        """Return the effective (positive) err_var correction scale."""
        return tf.nn.softplus(self._err_var_correction_scale_raw)

    @property
    def trainable_variables(self):
        """Collect all trainable variables."""
        vars_ = []
        # Correction scales
        vars_ += [self._h_correction_scale]
        vars_ += [self._err_var_correction_scale_raw]
        vars_ += [self._llr_correction_scale]
        # Shared backbone
        vars_ += self._shared_conv_in.trainable_variables
        for block in self._shared_res_blocks:
            vars_ += block.trainable_variables
        # CE head
        vars_ += self._ce_head_conv1.trainable_variables
        vars_ += self._ce_head_out_h.trainable_variables
        vars_ += self._ce_head_out_loge.trainable_variables
        # Detection continuation
        vars_ += self._det_inject_conv.trainable_variables
        for block in self._det_res_blocks:
            vars_ += block.trainable_variables
        vars_ += self._det_conv_out.trainable_variables
        return vars_

    def _reshape_logits_to_llr(self, logits, num_data_symbols):
        """
        Reshape [B, H, W, S*bits] -> [B, num_ue, streams_per_ue, num_data_symbols*bits]
        """
        B = tf.shape(logits)[0]
        logits = tf.reshape(
            logits,
            [B, num_data_symbols, self._num_streams_total, self._num_bits_per_symbol],
        )
        logits = tf.reshape(
            logits,
            [
                B,
                num_data_symbols,
                self._num_ue,
                self._num_streams_per_ue,
                self._num_bits_per_symbol,
            ],
        )
        logits = tf.transpose(logits, [0, 2, 3, 1, 4])
        llr = tf.reshape(
            logits,
            [
                B,
                self._num_ue,
                self._num_streams_per_ue,
                num_data_symbols * self._num_bits_per_symbol,
            ],
        )
        llr.set_shape([None, self._num_ue, self._num_streams_per_ue, None])
        return llr

    def call(self, y, h_hat, err_var, no, constellation=None, training=None):
        """
        Forward pass with merged backbone and trainable correction scales.

        Args:
            y: complex
            [B, num_bs, num_bs_ant, num_ofdm_syms, num_subcarriers]
            h_hat: complex
            [B, num_bs, num_bs_ant, num_ue, num_streams, num_ofdm_syms, num_subcarriers]
            err_var: float
            [B, 1, 1, num_ue, num_streams, num_ofdm_syms, num_subcarriers]
            no: [B] or scalar, noise variance
            constellation: complex, optional trainable constellation points
            [num_points]
            training: bool, training mode flag

        Returns:
            llr: [B, num_ue, num_streams_per_ue, num_data_symbols * num_bits_per_symbol]
        """
        # Cast inputs
        y = tf.cast(y, tf.complex64)
        h_hat = tf.cast(h_hat, tf.complex64)
        err_var = tf.cast(err_var, tf.float32)
        no = tf.cast(no, tf.float32)

        # Update constellation if trainable points provided
        if constellation is not None:
            self._constellation.points = tf.cast(constellation, tf.complex64)

        # Indices for data-only slicing (to be done after channel estimation refinement)
        data_idx = tf.constant(self._data_symbol_indices, dtype=tf.int32)

        # Use FULL symbol grid for CE refinement (pilots + data)
        B = tf.shape(y)[0]
        H = tf.shape(y)[3]  # num_ofdm_syms (incl pilots)
        W = tf.shape(y)[4]  # num_subcarriers

        S = self._num_streams_total
        Nr = self._num_rx_ant

        # ===== Reshape inputs to [B, H, W, ...] format =====
        # y: [B, num_bs, num_bs_ant, H, W] -> [B, H, W, Nr]
        y_flat = tf.reshape(y, [B, -1, H, W])
        y_flat = tf.transpose(y_flat, [0, 2, 3, 1])

        # h_hat: [B, num_bs, num_bs_ant, num_ue, streams, H, W] -> [B, H, W, Nr, S]
        h_flat = tf.reshape(h_hat, [B, -1, S, H, W])
        h_flat = tf.transpose(h_flat, [0, 3, 4, 1, 2])

        # err_var: [B, 1, 1, num_ue, streams, H, W] -> [B, H, W, S]
        err_var_t = tf.squeeze(err_var, axis=[1, 2])
        err_var_flat = tf.transpose(err_var_t, [0, 3, 4, 1, 2])
        err_var_flat = tf.reshape(err_var_flat, [B, H, W, S])

        # ===== Compute features ONCE =====
        # Matched filter: z_mf = H^H @ y
        h_conj_t = tf.transpose(tf.math.conj(h_flat), [0, 1, 2, 4, 3])  # [B,H,W,S,Nr]
        y_col = y_flat[..., tf.newaxis]  # [B,H,W,Nr,1]
        z_mf = tf.squeeze(tf.matmul(h_conj_t, y_col), axis=-1)  # [B,H,W,S]

        # Gram matrix: gram = H^H @ H
        gram = tf.matmul(h_conj_t, h_flat)  # [B,H,W,S,S]
        gram_diag = tf.math.real(tf.linalg.diag_part(gram))  # [B,H,W,S]

        # Gram off-diagonal (interference structure)
        mask = 1.0 - tf.eye(S, dtype=tf.float32)
        mask = mask[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
        gram_masked = gram * tf.cast(mask, gram.dtype)
        gram_offdiag = tf.abs(gram_masked)
        gram_offdiag_flat = tf.reshape(gram_offdiag, [B, H, W, -1])
        # Extract only off-diagonal elements (S*(S-1) values)
        indices = [i * S + j for i in range(S) for j in range(S) if i != j]
        gram_offdiag_feats = tf.gather(gram_offdiag_flat, indices, axis=-1)

        # h_flat features: [B,H,W,Nr,S] -> [B,H,W,Nr*S] real/imag
        h_flat_features = tf.reshape(h_flat, [B, H, W, Nr * S])
        h_feats = tf.concat(
            [tf.math.real(h_flat_features), tf.math.imag(h_flat_features)], axis=-1
        )

        # y features
        y_feats = tf.concat([tf.math.real(y_flat), tf.math.imag(y_flat)], axis=-1)

        # z_mf features
        z_mf_feats = tf.concat([tf.math.real(z_mf), tf.math.imag(z_mf)], axis=-1)

        # Noise feature (log scale, broadcast to spatial dims)
        no_log = log10(no + 1e-10)
        no_feat = tf.broadcast_to(
            no_log[:, tf.newaxis, tf.newaxis, tf.newaxis], [B, H, W, 1]
        )

        # ===== Assemble shared input features =====
        shared_input = tf.concat(
            [
                h_feats,  # 2 * Nr * S
                y_feats,  # 2 * Nr
                z_mf_feats,  # 2 * S
                gram_diag,  # S
                gram_offdiag_feats,  # S * (S-1)
                err_var_flat,  # S
                no_feat,  # 1
            ],
            axis=-1,
        )
        shared_input = tf.cast(shared_input, tf.float32)

        # ===== Shared Backbone =====
        shared_features = self._shared_conv_in(shared_input)
        for block in self._shared_res_blocks:
            shared_features = block(shared_features)
        # shared_features: [B, H, W, num_filters]

        # ===== CE Head =====
        ce_hidden = self._ce_head_conv1(shared_features)
        ce_hidden = leaky_relu(ce_hidden, alpha=0.1)

        delta_h_raw = self._ce_head_out_h(ce_hidden)  # [B,H,W, 2*Nr*S]
        delta_loge = self._ce_head_out_loge(ce_hidden)  # [B,H,W, S]

        # Parse delta_h: [B,H,W, 2*Nr*S] -> complex [B,H,W,Nr,S]
        delta_h_raw = tf.cast(delta_h_raw, tf.float32)
        delta_h_r = delta_h_raw[..., : Nr * S]
        delta_h_i = delta_h_raw[..., Nr * S :]
        delta_h_c = tf.complex(delta_h_r, delta_h_i)
        delta_h_c = tf.reshape(delta_h_c, [B, H, W, Nr, S])

        # ===== Apply SCALED channel refinement =====
        h_scale = tf.cast(self._h_correction_scale, tf.complex64)
        h_flat_refined = h_flat + h_scale * tf.cast(delta_h_c, h_flat.dtype)

        # ===== Apply SCALED err_var refinement in log-domain =====
        err_var_scale = self.err_var_correction_scale  # softplus-transformed
        log_err = tf.math.log(err_var_flat + 1e-10)
        log_err_refined = log_err + err_var_scale * tf.cast(delta_loge, log_err.dtype)
        err_var_flat_refined = tf.exp(log_err_refined)

        # ===== Store refined estimates for auxiliary losses =====
        # h_hat_refined: [B,H,W,Nr,S] -> [B, num_bs, num_bs_ant, num_ue, streams, H, W]
        h_ref_t = tf.transpose(h_flat_refined, [0, 3, 4, 1, 2])  # [B,Nr,S,H,W]
        h_ref_t = tf.reshape(
            h_ref_t,
            [
                B,
                self._num_bs,
                self._num_bs_ant,
                self._num_ue,
                self._num_streams_per_ue,
                H,
                W,
            ],
        )
        self.last_h_hat_refined = h_ref_t

        # err_var_refined: [B,H,W,S] -> [B,1,1,num_ue,streams,H,W]
        ev_ref = tf.reshape(
            err_var_flat_refined, [B, H, W, self._num_ue, self._num_streams_per_ue]
        )
        ev_ref = tf.transpose(ev_ref, [0, 3, 4, 1, 2])
        ev_ref = ev_ref[:, tf.newaxis, tf.newaxis, ...]
        self.last_err_var_refined = ev_ref
        self.last_err_var_refined_flat = tf.cast(err_var_flat_refined, tf.float32)

        # ===== LMMSE Equalization =====
        # Slice to DATA symbols only for detection
        y_flat_data = tf.gather(y_flat, data_idx, axis=1)  # [B, H_data, W, Nr]
        shared_features_data = tf.gather(
            shared_features, data_idx, axis=1
        )  # [B, H_data, W, F]
        h_flat_refined_data = tf.gather(
            h_flat_refined, data_idx, axis=1
        )  # [B, H_data, W, Nr, S]
        err_var_flat_refined_data = tf.gather(
            err_var_flat_refined, data_idx, axis=1
        )  # [B, H_data, W, S]

        H_data = tf.shape(y_flat_data)[1]
        num_data_symbols = H_data * W

        # Build noise covariance matrix
        no_expanded = no[:, tf.newaxis, tf.newaxis]
        sum_err_var = tf.reduce_sum(err_var_flat_refined_data, axis=-1)
        total_noise_var = no_expanded + sum_err_var
        eye = tf.eye(Nr, dtype=tf.complex64)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
        s_cov_data = (
            tf.cast(total_noise_var[..., tf.newaxis, tf.newaxis], tf.complex64) * eye
        )

        # LMMSE equalization
        x_lmmse, no_eff = lmmse_equalizer(
            y_flat_data, h_flat_refined_data, s_cov_data, whiten_interference=True
        )
        # x_lmmse: [B, H, W, S], no_eff: [B, H, W, S]

        # Demapping to get baseline LLRs
        x_lmmse_flat_dm = tf.reshape(x_lmmse, [-1])
        no_eff_flat_dm = tf.reshape(no_eff, [-1])
        llr_lmmse_flat = self._demapper(x_lmmse_flat_dm, no_eff_flat_dm)
        llr_lmmse = tf.reshape(
            llr_lmmse_flat, [B, H_data, W, S, self._num_bits_per_symbol]
        )
        llr_lmmse = tf.reshape(llr_lmmse, [B, H_data, W, S * self._num_bits_per_symbol])

        # ===== Build LMMSE features for detection continuation =====
        x_lmmse_feats = tf.concat(
            [tf.math.real(x_lmmse), tf.math.imag(x_lmmse)], axis=-1
        )  # 2*S
        no_eff_feats = tf.math.log(no_eff + 1e-10)  # S

        lmmse_features = tf.concat(
            [
                x_lmmse_feats,  # 2 * S
                no_eff_feats,  # S
                llr_lmmse,  # S * bits
            ],
            axis=-1,
        )
        lmmse_features = tf.cast(lmmse_features, tf.float32)

        # ===== Detection Continuation =====
        # Concatenate shared features with LMMSE features
        combined_features = tf.concat([shared_features_data, lmmse_features], axis=-1)

        # Injection conv to fuse and reduce dimensions
        det_features = self._det_inject_conv(combined_features)
        det_features = leaky_relu(det_features, alpha=0.1)

        # Detection ResBlocks
        for block in self._det_res_blocks:
            det_features = block(det_features)

        # Output LLR correction
        llr_correction = self._det_conv_out(det_features)

        # ===== Final LLR with SCALED correction =====
        llr_final = llr_lmmse + self._llr_correction_scale * llr_correction

        return self._reshape_logits_to_llr(llr_final, num_data_symbols)
