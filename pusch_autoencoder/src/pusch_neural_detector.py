import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, LayerNormalization
from tensorflow.nn import leaky_relu
from sionna.phy.utils import log10
from .config import Config


class Conv2DResBlock(Layer):
    """
    Improved residual block with two convolutions (standard ResNet style).
    """

    def __init__(
        self,
        filters: int,
        kernel_size=(3, 3),
        name: str = None,
    ):
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = kernel_size

        self._layer_norm1 = LayerNormalization(axis=-1)
        self._conv1 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
        )
        self._layer_norm2 = LayerNormalization(axis=-1)
        self._conv2 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
        )

    def call(self, x):
        # Pre-activation ResNet style
        y = self._layer_norm1(x)
        y = leaky_relu(y, alpha=0.1)
        y = self._conv1(y)
        y = self._layer_norm2(y)
        y = leaky_relu(y, alpha=0.1)
        y = self._conv2(y)
        return x + y


class PUSCHNeuralDetector(Layer):
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

        # System parameters from config
        self._num_bits_per_symbol = int(self._cfg.num_bits_per_symbol)
        self._num_ue = int(self._cfg.num_ue)
        self._num_streams_per_ue = int(self._cfg.num_layers)
        self._num_streams_total = self._num_ue * self._num_streams_per_ue
        self._num_bs = int(self._cfg.num_bs)
        self._num_bs_ant = int(self._cfg.num_bs_ant)
        self._pusch_pilot_indices = list(self._cfg.pusch_pilot_indices)
        self._pusch_num_subcarriers = int(self._cfg.pusch_num_subcarriers)
        self._pusch_num_symbols_per_slot = int(self._cfg.pusch_num_symbols_per_slot)

        # Data symbol indices (OFDM symbols without pilots)
        all_symbols = list(range(self._pusch_num_symbols_per_slot))
        pilots = set(self._pusch_pilot_indices)
        self._data_symbol_indices = [s for s in all_symbols if s not in pilots]

        # Compute input feature dimension:
        # - Matched filter output z_mf: 2 * num_streams_total (real/imag)
        # - Gram matrix diagonal (signal power): num_streams_total
        # - Gram matrix off-diagonal (interference): num_streams_total * (num_streams_total - 1)
        # - Error variance per stream: num_streams_total
        # - Noise variance: 1
        self._c_in = (
            2 * self._num_streams_total +           # z_mf (matched filter)
            self._num_streams_total +               # Gram diagonal
            self._num_streams_total * (self._num_streams_total - 1) +  # Gram off-diag (real only, it's Hermitian)
            self._num_streams_total +               # err_var
            1                                       # noise
        )

        # Baseline: 2-layer MLP with Conv2D (captures local correlations)
        self._baseline_conv1 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            padding="same",
            activation=None,
            name="baseline_conv1",
        )
        self._baseline_ln = LayerNormalization(axis=-1)
        self._baseline_conv2 = Conv2D(
            filters=self._num_streams_total * self._num_bits_per_symbol,
            kernel_size=(1, 1),
            padding="same",
            activation=None,
            name="baseline_conv2",
        )

        # Deep residual network for refinement
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

    @property
    def trainable_variables(self):
        vars_ = []
        vars_ += self._baseline_conv1.trainable_variables
        vars_ += self._baseline_ln.trainable_variables
        vars_ += self._baseline_conv2.trainable_variables
        vars_ += self._conv2d_in.trainable_variables
        for block in self._res_blocks:
            vars_ += block.trainable_variables
        vars_ += self._conv2d_out.trainable_variables
        return vars_

    def _compute_matched_filter_features(self, y, h_hat, err_var, no):
        """
        Compute matched filter output and Gram matrix features.
        
        These are the fundamental features that classical MIMO detectors use.
        By providing them directly, the network doesn't have to learn
        matrix multiplication from scratch.
        
        Args:
            y: [B, num_bs, num_bs_ant, H, W] complex received signal
            h_hat: [B, num_bs, num_bs_ant, num_ue, num_streams_per_ue, H, W] complex channel
            err_var: [B, 1, 1, num_ue, num_streams_per_ue, H, W] real error variance
            no: [B] or scalar, noise variance
            
        Returns:
            z_mf: [B, H, W, 2*num_streams] matched filter output (real/imag)
            gram_diag: [B, H, W, num_streams] Gram matrix diagonal
            gram_offdiag: [B, H, W, num_streams*(num_streams-1)] Gram off-diagonal
        """
        B = tf.shape(y)[0]
        H = tf.shape(y)[3]
        W = tf.shape(y)[4]
        
        # Reshape for matrix operations
        # y: [B, num_bs * num_bs_ant, H, W] -> [B, H, W, num_bs * num_bs_ant]
        y_flat = tf.reshape(y, [B, -1, H, W])  # [B, num_rx_ant, H, W]
        y_flat = tf.transpose(y_flat, [0, 2, 3, 1])  # [B, H, W, num_rx_ant]
        
        # h_hat: [B, num_bs, num_bs_ant, num_ue, num_streams_per_ue, H, W]
        # -> [B, num_rx_ant, num_streams, H, W] -> [B, H, W, num_rx_ant, num_streams]
        h_flat = tf.reshape(h_hat, [B, -1, self._num_streams_total, H, W])
        h_flat = tf.transpose(h_flat, [0, 3, 4, 1, 2])  # [B, H, W, num_rx_ant, num_streams]
        
        # Matched filter: z_mf = H^H @ y
        # H^H: [B, H, W, num_streams, num_rx_ant]
        # y: [B, H, W, num_rx_ant, 1]
        h_conj_t = tf.transpose(tf.math.conj(h_flat), [0, 1, 2, 4, 3])  # [B, H, W, num_streams, num_rx_ant]
        y_col = y_flat[..., tf.newaxis]  # [B, H, W, num_rx_ant, 1]
        z_mf = tf.squeeze(tf.matmul(h_conj_t, y_col), axis=-1)  # [B, H, W, num_streams]
        
        # Split into real/imag
        z_mf_real = tf.math.real(z_mf)
        z_mf_imag = tf.math.imag(z_mf)
        z_mf_feats = tf.concat([z_mf_real, z_mf_imag], axis=-1)  # [B, H, W, 2*num_streams]
        
        # Gram matrix: G = H^H @ H
        # G: [B, H, W, num_streams, num_streams]
        gram = tf.matmul(h_conj_t, h_flat)  # [B, H, W, num_streams, num_streams]
        
        # Diagonal elements (real, since Hermitian)
        gram_diag = tf.math.real(tf.linalg.diag_part(gram))  # [B, H, W, num_streams]
        
        # Off-diagonal elements (flattened, taking real part since |G_ij| matters for interference)
        # We'll take the magnitude of off-diagonal elements
        mask = 1.0 - tf.eye(self._num_streams_total, dtype=tf.float32)
        mask = mask[tf.newaxis, tf.newaxis, tf.newaxis, :, :]  # [1, 1, 1, S, S]
        gram_masked = gram * tf.cast(mask, gram.dtype)
        gram_offdiag = tf.abs(gram_masked)  # Magnitude of interference terms
        # Flatten off-diagonal: [B, H, W, num_streams * num_streams] but we need to remove diagonal
        gram_offdiag_flat = tf.reshape(gram_offdiag, [B, H, W, -1])
        # Remove zeros (diagonal positions) - simpler to just keep all for now
        # Actually, let's just use the upper triangle to avoid redundancy
        indices = []
        for i in range(self._num_streams_total):
            for j in range(self._num_streams_total):
                if i != j:
                    indices.append(i * self._num_streams_total + j)
        gram_offdiag_feats = tf.gather(gram_offdiag_flat, indices, axis=-1)
        
        return z_mf_feats, gram_diag, gram_offdiag_feats

    def _reshape_logits_to_llr(self, logits, num_data_symbols):
        """
        logits: [B, H, W, F] with F = num_streams_total * num_bits_per_symbol
        """
        B = tf.shape(logits)[0]
        
        logits = tf.reshape(
            logits,
            [B, num_data_symbols, self._num_streams_total, self._num_bits_per_symbol]
        )
        logits = tf.reshape(
            logits,
            [B, num_data_symbols, self._num_ue, self._num_streams_per_ue, self._num_bits_per_symbol]
        )
        logits = tf.transpose(logits, [0, 2, 3, 1, 4])
        llr = tf.reshape(
            logits,
            [B, self._num_ue, self._num_streams_per_ue, num_data_symbols * self._num_bits_per_symbol]
        )
        llr.set_shape([None, self._num_ue, self._num_streams_per_ue, None])
        return llr

    def call(self, y, h_hat, err_var, no, training=None):
        """
        PUSCH Neural MIMO-OFDM Detector with matched filter features.
        """
        y = tf.cast(y, tf.complex64)
        h_hat = tf.cast(h_hat, tf.complex64)
        err_var = tf.cast(err_var, tf.float32)
        no = tf.cast(no, tf.float32)

        # Mask pilots
        data_idx = tf.constant(self._data_symbol_indices, dtype=tf.int32)
        y = tf.gather(y, data_idx, axis=3)
        h_hat = tf.gather(h_hat, data_idx, axis=5)
        err_var = tf.gather(err_var, data_idx, axis=5)

        B = tf.shape(y)[0]
        H = tf.shape(y)[3]
        W = tf.shape(y)[4]
        num_data_symbols = H * W

        # === Compute matched filter features ===
        z_mf_feats, gram_diag, gram_offdiag = self._compute_matched_filter_features(
            y, h_hat, err_var, no
        )

        # Error variance features
        err_var_t = tf.squeeze(err_var, axis=[1, 2])  # [B, num_ue, streams, H, W]
        err_feats = tf.transpose(err_var_t, [0, 3, 4, 1, 2])  # [B, H, W, num_ue, streams]
        err_feats = tf.reshape(err_feats, [B, H, W, self._num_streams_total])

        # Noise variance feature
        no_log = log10(no + 1e-10)
        no_expanded = tf.broadcast_to(
            no_log[..., tf.newaxis, tf.newaxis, tf.newaxis],
            [B, H, W, 1]
        )

        # === Build feature tensor ===
        z = tf.concat([
            z_mf_feats,      # Matched filter output (critical!)
            gram_diag,       # Signal power per stream
            gram_offdiag,    # Interference structure
            err_feats,       # Channel estimation error
            no_expanded,     # Noise level
        ], axis=-1)
        z = tf.cast(z, tf.float32)

        # === Baseline detector (2-layer MLP with 1x1 convs) ===
        baseline = self._baseline_conv1(z)
        baseline = self._baseline_ln(baseline)
        baseline = leaky_relu(baseline, alpha=0.1)
        logits_baseline = self._baseline_conv2(baseline)
        llr_baseline = self._reshape_logits_to_llr(logits_baseline, num_data_symbols)

        # === Deep residual refinement ===
        z_feat = self._conv2d_in(z)
        for block in self._res_blocks:
            z_feat = block(z_feat)
        logits_residual = self._conv2d_out(z_feat)
        llr_residual = self._reshape_logits_to_llr(logits_residual, num_data_symbols)

        # === Combined output ===
        llr = llr_baseline + llr_residual
        return llr
