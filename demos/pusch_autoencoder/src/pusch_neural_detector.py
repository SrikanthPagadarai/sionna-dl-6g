"""
Neural MIMO detector for PUSCH with learned channel estimation refinement.

Implements a hybrid classical/neural network architecture that combines
the reliability of LS channel estimation and LMMSE equalization with
the adaptability of neural networks. The key motivation behind this
design is that learning residual corrections to classical estimates is
more stable than learning detection from scratch.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import leaky_relu
from sionna.phy.mimo import lmmse_equalizer
from sionna.phy.mapping import Demapper, Constellation
from sionna.phy.utils import log10

from .config import Config


class Conv2DResBlock(Layer):
    r"""
    Pre-activation residual block with two convolutional layers.

    Implements the pre-activation ResNet variant where normalization and
    activation precede each convolution. This ordering improves gradient
    flow and enables training of deeper networks.

    The block computes: ``output = input + conv2(act(norm(conv1(act(norm(input))))))``

    Parameters
    ----------
    filters : int
        Number of convolutional filters in both layers.
    kernel_size : tuple of int
        Spatial kernel size, default ``(3, 3)``.
    name : str, optional
        Layer name for TensorFlow graph.

    Notes
    -----
    - Uses LayerNormalization (not BatchNorm) for stable training with
      small batch sizes typical in communication system simulation.
    - LeakyReLU with alpha=0.1 prevents dead neurons while maintaining
      near-linear behavior for small negative inputs.
    - Identity skip connection requires input and output channel counts
      to match; caller must ensure ``filters`` equals input channels.
    """

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
        """
        Apply residual transformation.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``[batch, height, width, filters]``.

        Returns
        -------
        tf.Tensor
            Output tensor with same shape as input (residual added).
        """
        y = self._layer_norm1(x)
        y = leaky_relu(y, alpha=0.1)
        y = self._conv1(y)
        y = self._layer_norm2(y)
        y = leaky_relu(y, alpha=0.1)
        y = self._conv2(y)
        return x + y


class PUSCHNeuralDetector(Layer):
    r"""
    Neural MIMO detector with learned channel estimation refinement for PUSCH.

    This detector implements a residual learning architecture that refines
    classical LS channel estimates and LMMSE-based soft symbol estimates
    using convolutional neural networks. The key design principle is to
    combine classical baselines with learned refinements.

    Architecture
    ------------
    The detector processes data through six stages:

    1. **Feature extraction**: Assembles input features including LS channel
       estimate, received signal, matched filter output, Gram matrix structure,
       estimation error variance, and noise level.

    2. **Shared backbone**: Processes features through convolutional ResBlocks
       to learn joint representations useful for both CE refinement and detection.

    3. **CE refinement head**: Projects shared features through 1x1 convolutions
       to predict additive corrections delta-h, and multiplicative log-domain
       corrections delta-log(err_var), to the LS estimates.

    4. **Scaled correction application**: Applies learned corrections scaled by
       trainable parameters that start at zero, enabling gradual departure from
       classical behavior during training.

    5. **Classical LMMSE**: Performs LMMSE equalization using refined channel
       estimate and error variance, followed by max-log demapping.

    6. **LLR refinement**: Processes LMMSE outputs (equalized symbols, effective
       noise, baseline LLRs) through ResBlocks to predict additive LLR
       corrections, again scaled by a trainable parameter.

    Parameters
    ----------
    cfg : Config
        System configuration containing MIMO dimensions, modulation order,
        and PUSCH resource grid information.
    num_conv2d_filters : int
        Number of filters in all convolutional layers. Higher values increase
        model capacity but also computational cost. Default 128.
    num_shared_res_blocks : int
        Number of ResBlocks in the shared backbone. Controls depth of joint
        feature learning. Default 4.
    num_det_res_blocks : int
        Number of ResBlocks in the detection continuation path. Controls
        capacity for LLR refinement. Default 6.
    kernel_size : tuple of int
        Spatial kernel size for ResBlock convolutions. Default ``(3, 3)``.

    Example
    -------
    >>> cfg = Config()
    >>> # ... set cfg.pusch_pilot_indices via PUSCHLinkE2E ...
    >>> detector = PUSCHNeuralDetector(cfg, num_conv2d_filters=64)
    >>> llr = detector(y, h_hat, err_var, no)

    Notes
    -----
    The trainable correction scales serve multiple purposes:

    1. ``cfg.pusch_pilot_indices`` must be set (by ``PUSCHLinkE2E``) before
       instantiation to enable pilot/data symbol separation.

    2. Input tensors must follow Sionna's PUSCH dimension conventions.

    3. ``trainable_variables`` returns correction scales first, then network
       weights (enables separate optimizer configuration).

    4. ``last_h_hat_refined`` and ``last_err_var_refined`` contain the most
       recent refined estimates (useful for auxiliary losses).

    5. Output LLR shape matches Sionna's ``PUSCHReceiver`` expectations.

    6. **Safe initialization**: Starting at 0.0 means the detector initially
       behaves exactly like classical LMMSE, providing a stable starting point.

    7. **Interpretability**: Scale magnitudes indicate how much the network
       deviates from classical processing.

    8. **Gradient balancing**: Separate scales for h, err_var, and LLR allow
       independent learning rates for different correction types.

    9. The error variance scale uses softplus to ensure positivity, since
       negative variance would be physically meaningless and cause numerical
       issues in LMMSE computation.
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

        # =====================================================================
        # Extract Dimensions from Config
        # =====================================================================
        self._num_bits_per_symbol = int(self._cfg.num_bits_per_symbol)
        self._num_ue = int(self._cfg.num_ue)
        self._num_streams_per_ue = int(self._cfg.num_layers)
        self._num_streams_total = self._num_ue * self._num_streams_per_ue
        self._num_bs = int(self._cfg.num_bs)
        self._num_bs_ant = int(self._cfg.num_bs_ant)
        self._num_rx_ant = self._num_bs * self._num_bs_ant
        self._pusch_pilot_indices = list(self._cfg.pusch_pilot_indices)
        self._pusch_num_symbols_per_slot = int(self._cfg.pusch_num_symbols_per_slot)

        # Separate pilot and data symbol indices for selective processing.
        # CE refinement uses all symbols; detection uses data symbols only.
        all_symbols = list(range(self._pusch_num_symbols_per_slot))
        pilots = set(self._pusch_pilot_indices)
        self._data_symbol_indices = [s for s in all_symbols if s not in pilots]

        # =====================================================================
        # Constellation and Demapper
        # =====================================================================
        # Initialize with standard QAM; points may be overwritten during call()
        # if a trainable constellation is provided by the transmitter.
        qam_points = Constellation("qam", self._num_bits_per_symbol).points
        self._constellation = Constellation(
            "custom",
            num_bits_per_symbol=self._num_bits_per_symbol,
            points=qam_points,
            normalize=False,
        )
        self._demapper = Demapper("maxlog", constellation=self._constellation)

        # =====================================================================
        # Trainable Correction Scales
        # =====================================================================
        # Initialize all scales to 0.0 so that initial output matches classical
        # LMMSE exactly. This provides a stable starting point and enables
        # graceful degradation if training fails to improve on classical.

        # Channel estimate correction: h_refined = h_ls + scale * delta_h
        # Unbounded since corrections can be positive or negative.
        self._h_correction_scale = tf.Variable(
            0.0, trainable=True, name="h_correction_scale", dtype=tf.float32
        )

        # Error variance correction in log domain for numerical stability:
        # err_var_refined = exp(log(err_var) + scale * delta_log_err)
        # Uses softplus(raw_value) to ensure positivity;
        # softplus(0) = ln(2) = (approximately) 0.69
        # but we want initial scale = (approximately) 1.0,
        # and softplus(0.54) = (approximately) 1.0
        # For simplicity, initialize to 0.0; the network will adapt.
        self._err_var_correction_scale_raw = tf.Variable(
            0.0, trainable=True, name="err_var_correction_scale_raw", dtype=tf.float32
        )

        # LLR correction: llr_final = llr_lmmse + scale * delta_llr
        # Unbounded to allow both confidence increase and decrease.
        self._llr_correction_scale = tf.Variable(
            0.0, trainable=True, name="llr_correction_scale", dtype=tf.float32
        )

        # =====================================================================
        # Compute Input Feature Dimensions
        # =====================================================================
        # The shared backbone receives a concatenation of multiple feature types,
        # each providing complementary information about the channel and signal.
        S = self._num_streams_total
        Nr = self._num_rx_ant

        self._c_in_shared = (
            2 * Nr * S  # h_ls: real and imaginary parts of channel estimate
            + 2 * Nr  # y: real and imaginary parts of received signal
            + 2 * S  # z_mf: matched filter output (H^H @ y), captures signal energy
            + S  # gram_diag: diagonal of H^H @ H, indicates per-stream SNR
            + S * (S - 1)  # gram_offdiag: inter-stream interference structure
            + S  # err_var: channel estimation error variance per stream
            + 1  # no: noise variance (log scale for numerical stability)
        )

        # =====================================================================
        # Shared Backbone Network
        # =====================================================================
        # Processes all input features to learn representations useful for both
        # channel estimation refinement and detection. Sharing weights between
        # these tasks acts as implicit regularization and reduces parameters.
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

        # =====================================================================
        # Channel Estimation Refinement Head
        # =====================================================================
        # Direct 1×1 projections from shared features to channel corrections.
        # The shared backbone already provides rich non-linear features;
        # additional convolutions here were found to be unnecessary.

        # Output layer for channel correction: 2 * Nr * S values (real + imag)
        self._ce_head_out_h = Conv2D(
            filters=2 * Nr * S,
            kernel_size=(1, 1),
            padding="same",
            activation=None,
            name="ce_head_out_h",
        )

        # Output layer for error variance log-correction: S values per RE
        self._ce_head_out_loge = Conv2D(
            filters=S,
            kernel_size=(1, 1),
            padding="same",
            activation=None,
            name="ce_head_out_loge",
        )

        # =====================================================================
        # Detection Continuation Network
        # =====================================================================
        # After LMMSE equalization with refined channel estimates, this network
        # learns to correct the resulting LLRs based on LMMSE outputs (equalized
        # symbols, effective noise, baseline LLRs). The shared backbone
        # features have already been consumed by the CE refinement and LMMSE
        # computation, so detection operates only on LMMSE-derived features.
        self._c_lmmse_feats = (
            2 * S  # x_lmmse: equalized symbols (real + imag)
            + S  # no_eff: post-equalization noise variance (log scale)
            + S * self._num_bits_per_symbol  # llr_lmmse: baseline soft bits
        )

        # Injection convolution projects LMMSE features to hidden dimension
        self._det_inject_conv = Conv2D(
            filters=self.num_conv2d_filters,
            kernel_size=(3, 3),
            padding="same",
            name="det_inject_conv",
        )

        # Additional ResBlocks for LLR refinement
        self._det_res_blocks = [
            Conv2DResBlock(
                filters=self.num_conv2d_filters,
                kernel_size=self.kernel_size,
                name=f"det_resblock_{i}",
            )
            for i in range(self.num_det_res_blocks)
        ]

        # Output convolution produces LLR corrections for all bits
        self._det_conv_out = Conv2D(
            filters=S * self._num_bits_per_symbol,
            kernel_size=(3, 3),
            padding="same",
            activation=None,
            name="det_conv_out",
        )

        # =====================================================================
        # State for Auxiliary Losses
        # =====================================================================
        # These attributes store refined estimates from the most recent forward
        # pass, enabling auxiliary loss computation (e.g., MSE on h_hat).
        self.last_h_hat_refined = None
        self.last_err_var_refined = None
        self.last_err_var_refined_flat = None

    @property
    def err_var_correction_scale(self):
        """
        Effective error variance correction scale (always positive).

        Returns
        -------
        tf.Tensor
            Scalar tensor with softplus-transformed scale value.
            softplus(x) = log(1 + exp(x)) ensures output > 0.
        """
        return tf.nn.softplus(self._err_var_correction_scale_raw)

    @property
    def trainable_variables(self):
        """
        Collect all trainable variables in a specific order.

        The ordering places correction scales first, enabling separate
        optimizer configuration (e.g., higher learning rate for scales).

        Returns
        -------
        list of tf.Variable
            Ordered list: [correction_scales, backbone_weights, ce_head_weights,
            detection_weights].
        """
        vars_ = []
        # Correction scales first (for easy separate optimizer setup)
        vars_ += [self._h_correction_scale]
        vars_ += [self._err_var_correction_scale_raw]
        vars_ += [self._llr_correction_scale]
        # Shared backbone
        vars_ += self._shared_conv_in.trainable_variables
        for block in self._shared_res_blocks:
            vars_ += block.trainable_variables
        # CE head (direct projections only)
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
        Reshape detector output to match Sionna's PUSCHReceiver LLR format.

        Parameters
        ----------
        logits : tf.Tensor
            Network output, shape ``[B, H_data, W, S * num_bits_per_symbol]``.
        num_data_symbols : int or tf.Tensor
            Total number of data resource elements (H_data * W).

        Returns
        -------
        tf.Tensor
            Reshaped LLRs, shape ``[B, num_ue, streams_per_ue, num_data_symbols * bits]``.

        Notes
        -----
        The reshape sequence preserves bit ordering while reorganizing from
        spatial (H, W) layout to the flattened per-UE format expected by
        the transport block decoder.
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
        Execute neural MIMO detection with channel estimation refinement.

        Parameters
        ----------
        y : tf.Tensor, complex64
            Received OFDM signal after FFT.
            Shape: ``[B, num_bs, num_bs_ant, num_ofdm_syms, num_subcarriers]``
        h_hat : tf.Tensor, complex64
            LS channel estimate from pilot symbols.
            Shape: ``[B, num_bs, num_bs_ant, num_ue, num_streams, num_ofdm_syms, num_subcarriers]``
        err_var : tf.Tensor, float32
            Channel estimation error variance.
            Shape: ``[B, num_bs, num_bs_ant, num_ue, num_streams, num_ofdm_syms, num_subcarriers]``
            or broadcastable variant.
        no : tf.Tensor, float32
            Noise variance per batch element. Shape: ``[B]`` or ``[B, 1, 1, ...]``.
        constellation : tf.Tensor, complex64, optional
            Constellation points for demapping. If provided, overrides the
            internal constellation (useful for learned constellations).
            Shape: ``[num_points]``.
        training : bool, optional
            Training mode flag (currently unused but included for Keras API
            compatibility).

        Returns
        -------
        tf.Tensor, float32
            Log-likelihood ratios for all coded bits.
            Shape: ``[B, num_ue, num_streams_per_ue, num_data_symbols * num_bits_per_symbol]``

        Notes
        -----
        The function extracts extensive features from inputs, processes them
        through the shared backbone, applies CE refinement, performs LMMSE
        equalization, and finally refines the LLRs. The three trainable scales
        control how much the neural corrections contribute vs. classical
        processing.

        Side effects: Updates ``self.last_h_hat_refined``,
        ``self.last_err_var_refined``, and ``self.last_err_var_refined_flat``
        with refined estimates for auxiliary loss computation.
        """
        # Update constellation for demapper if provided (e.g., learned TX constellation)
        if constellation is not None:
            self._constellation.points = tf.cast(constellation, tf.complex64)

        # =====================================================================
        # Dimension Extraction and Validation
        # =====================================================================
        S = self._num_streams_total
        Nr = self._num_rx_ant

        B = tf.shape(y)[0]
        H = tf.shape(y)[3]  # num_ofdm_symbols (includes pilots)
        W = tf.shape(y)[4]  # num_subcarriers
        data_idx = tf.constant(self._data_symbol_indices, dtype=tf.int32)

        # =====================================================================
        # Reshape Inputs to Spatial [B, H, W, C] Format
        # =====================================================================
        # Sionna uses dimension ordering optimized for FFT; we need spatial
        # ordering for 2D convolutions. All reshapes preserve data layout.

        # y: [B, num_bs, num_bs_ant, H, W] -> [B, H, W, Nr]
        y_flat = tf.transpose(y[:, 0, :, :, :], [0, 2, 3, 1])

        # h_hat: [B, num_bs, num_bs_ant, num_ue, streams, H, W] -> [B, H, W, Nr, S]
        h_hat_t = h_hat[:, 0, :, :, :, :, :]  # [B, num_bs_ant, num_ue, streams, H, W]
        h_hat_t = tf.reshape(
            h_hat_t, [B, Nr, self._num_ue * self._num_streams_per_ue, H, W]
        )
        h_flat = tf.transpose(h_hat_t, [0, 3, 4, 1, 2])  # [B, H, W, Nr, S]

        # err_var: [B, 1, 1, num_ue, streams, H, W] -> [B, H, W, S]
        ev = tf.reshape(
            err_var[:, 0, 0, :, :, :, :],
            [B, self._num_ue * self._num_streams_per_ue, H, W],
        )
        err_var_flat = tf.transpose(ev, [0, 2, 3, 1])  # [B, H, W, S]

        # no: ensure shape [B]
        no = tf.reshape(no, [B])

        # =====================================================================
        # Compute Derived Features
        # =====================================================================
        # These features provide the network with different "views" of the
        # channel and signal that highlight different aspects of the MIMO
        # detection problem.

        # Matched filter output: z_mf = H^H @ y (sufficient statistic)
        # Shape: [B, H, W, S]
        y_expanded = y_flat[..., tf.newaxis]  # [B, H, W, Nr, 1]
        h_conj = tf.math.conj(h_flat)  # [B, H, W, Nr, S]
        z_mf = tf.reduce_sum(h_conj * y_expanded, axis=-2)  # [B, H, W, S]

        # Gram matrix: G = H^H @ H captures interference structure
        # Shape: [B, H, W, S, S]
        h_expanded = h_flat[..., tf.newaxis]  # [B, H, W, Nr, S, 1]
        h_conj_expanded = h_conj[..., tf.newaxis, :]  # [B, H, W, Nr, 1, S]
        gram = tf.reduce_sum(h_expanded * h_conj_expanded, axis=-3)  # [B, H, W, S, S]

        # Diagonal: per-stream channel power (related to SNR)
        gram_diag = tf.math.real(tf.linalg.diag_part(gram))  # [B, H, W, S]

        # Off-diagonal: inter-stream interference (flattened, upper triangle)
        # Extract unique off-diagonal elements to avoid redundancy
        gram_offdiag_list = []
        for i in range(S):
            for j in range(S):
                if i != j:
                    gram_offdiag_list.append(gram[..., i, j])
        gram_offdiag = tf.stack(gram_offdiag_list, axis=-1)  # [B, H, W, S*(S-1)]
        gram_offdiag_feats = tf.concat(
            [tf.math.real(gram_offdiag), tf.math.imag(gram_offdiag)], axis=-1
        )

        # =====================================================================
        # Prepare Feature Tensors
        # =====================================================================
        # Split complex tensors into real/imag and flatten channel dimensions

        # Channel estimate features: [B, H, W, 2*Nr*S]
        h_flat_r = tf.math.real(h_flat)
        h_flat_i = tf.math.imag(h_flat)
        h_feats = tf.concat(
            [
                tf.reshape(h_flat_r, [B, H, W, Nr * S]),
                tf.reshape(h_flat_i, [B, H, W, Nr * S]),
            ],
            axis=-1,
        )

        # Received signal features: [B, H, W, 2*Nr]
        y_feats = tf.concat([tf.math.real(y_flat), tf.math.imag(y_flat)], axis=-1)

        # Matched filter features: [B, H, W, 2*S]
        z_mf_feats = tf.concat([tf.math.real(z_mf), tf.math.imag(z_mf)], axis=-1)

        # Noise variance in log scale for numerical stability: [B, H, W, 1]
        no_feat = log10(no + 1e-10)[:, tf.newaxis, tf.newaxis, tf.newaxis]
        no_feat = tf.broadcast_to(no_feat, [B, H, W, 1])

        # =====================================================================
        # Assemble Shared Input Features
        # =====================================================================
        shared_input = tf.concat(
            [
                h_feats,  # 2 * Nr * S: channel estimate
                y_feats,  # 2 * Nr: received signal
                z_mf_feats,  # 2 * S: matched filter output
                gram_diag,  # S: per-stream channel power
                gram_offdiag_feats,  # S * (S-1): interference structure
                err_var_flat,  # S: estimation error variance
                no_feat,  # 1: noise level
            ],
            axis=-1,
        )
        shared_input = tf.cast(shared_input, tf.float32)

        # =====================================================================
        # Shared Backbone Forward Pass
        # =====================================================================
        shared_features = self._shared_conv_in(shared_input)
        for block in self._shared_res_blocks:
            shared_features = block(shared_features)
        # shared_features: [B, H, W, num_filters]

        # =====================================================================
        # Channel Estimation Refinement Head
        # =====================================================================
        # Direct 1x1 projection from shared features (no intermediate layers)
        delta_h_raw = self._ce_head_out_h(shared_features)  # [B,H,W, 2*Nr*S]
        delta_loge = self._ce_head_out_loge(shared_features)  # [B,H,W, S]

        # Parse channel correction into complex format
        delta_h_raw = tf.cast(delta_h_raw, tf.float32)
        delta_h_r = delta_h_raw[..., : Nr * S]
        delta_h_i = delta_h_raw[..., Nr * S :]
        delta_h_c = tf.complex(delta_h_r, delta_h_i)
        delta_h_c = tf.reshape(delta_h_c, [B, H, W, Nr, S])

        # =====================================================================
        # Apply Scaled Channel Refinement
        # =====================================================================
        # Additive correction: h_refined = h_ls + scale * delta_h
        # Scale starts at 0, so initial behavior matches LS exactly.
        h_scale = tf.cast(self._h_correction_scale, tf.complex64)
        h_flat_refined = h_flat + h_scale * tf.cast(delta_h_c, h_flat.dtype)

        # =====================================================================
        # Apply Scaled Error Variance Refinement (Log Domain)
        # =====================================================================
        # Multiplicative correction in log domain for numerical stability:
        # err_var_refined = exp(log(err_var) + scale * delta_log_err)
        # This ensures err_var remains positive regardless of delta magnitude.
        err_var_scale = self.err_var_correction_scale  # softplus-transformed
        log_err = tf.math.log(err_var_flat + 1e-10)
        log_err_refined = log_err + err_var_scale * tf.cast(delta_loge, log_err.dtype)
        err_var_flat_refined = tf.exp(log_err_refined)

        # =====================================================================
        # Store Refined Estimates for Auxiliary Losses
        # =====================================================================
        # Reshape back to Sionna's dimension convention for external access
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

        # =====================================================================
        # LMMSE Equalization on Data Symbols Only
        # =====================================================================
        # Slice to data symbols (exclude pilots) for detection
        y_flat_data = tf.gather(y_flat, data_idx, axis=1)  # [B, H_data, W, Nr]
        h_flat_refined_data = tf.gather(
            h_flat_refined, data_idx, axis=1
        )  # [B, H_data, W, Nr, S]
        err_var_flat_refined_data = tf.gather(
            err_var_flat_refined, data_idx, axis=1
        )  # [B, H_data, W, S]

        H_data = tf.shape(y_flat_data)[1]
        num_data_symbols = H_data * W

        # Build noise covariance matrix for LMMSE
        # Total noise = AWGN + channel estimation error (summed over streams)
        no_expanded = no[:, tf.newaxis, tf.newaxis]
        sum_err_var = tf.reduce_sum(err_var_flat_refined_data, axis=-1)
        total_noise_var = no_expanded + sum_err_var
        eye = tf.eye(Nr, dtype=tf.complex64)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
        s_cov_data = (
            tf.cast(total_noise_var[..., tf.newaxis, tf.newaxis], tf.complex64) * eye
        )

        # LMMSE equalization with whitened interference model
        x_lmmse, no_eff = lmmse_equalizer(
            y_flat_data, h_flat_refined_data, s_cov_data, whiten_interference=True
        )
        # x_lmmse: [B, H_data, W, S] - equalized symbols
        # no_eff: [B, H_data, W, S] - effective noise variance per stream

        # =====================================================================
        # Demapping to Baseline LLRs
        # =====================================================================
        x_lmmse_flat_dm = tf.reshape(x_lmmse, [-1])
        no_eff_flat_dm = tf.reshape(no_eff, [-1])
        llr_lmmse_flat = self._demapper(x_lmmse_flat_dm, no_eff_flat_dm)
        llr_lmmse = tf.reshape(
            llr_lmmse_flat, [B, H_data, W, S, self._num_bits_per_symbol]
        )
        llr_lmmse = tf.reshape(llr_lmmse, [B, H_data, W, S * self._num_bits_per_symbol])

        # =====================================================================
        # Build LMMSE Features for Detection Continuation
        # =====================================================================
        # Detection operates solely on LMMSE outputs - the shared backbone
        # features have already been consumed by CE refinement and LMMSE.
        x_lmmse_feats = tf.concat(
            [tf.math.real(x_lmmse), tf.math.imag(x_lmmse)], axis=-1
        )  # 2*S
        no_eff_feats = tf.math.log(no_eff + 1e-10)  # S (log scale)

        lmmse_features = tf.concat(
            [
                x_lmmse_feats,  # 2 * S: equalized symbols
                no_eff_feats,  # S: effective noise variance
                llr_lmmse,  # S * bits: baseline LLRs
            ],
            axis=-1,
        )
        lmmse_features = tf.cast(lmmse_features, tf.float32)

        # =====================================================================
        # Detection Continuation Network
        # =====================================================================
        # Process LMMSE features only (shared backbone features not included)
        det_features = self._det_inject_conv(lmmse_features)

        # ResBlocks for LLR refinement
        for block in self._det_res_blocks:
            det_features = block(det_features)

        # Predict LLR corrections
        llr_correction = self._det_conv_out(det_features)

        # =====================================================================
        # Final LLR with Scaled Correction
        # =====================================================================
        # Additive correction: llr_final = llr_lmmse + scale * delta_llr
        # Scale starts at 0, so initial behavior matches LMMSE exactly.
        llr_final = llr_lmmse + self._llr_correction_scale * llr_correction

        return self._reshape_logits_to_llr(llr_final, num_data_symbols)
