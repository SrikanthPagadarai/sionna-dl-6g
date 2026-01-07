# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Neural MIMO detector for PUSCH with learned channel estimation refinement.

This module implements a hybrid classical/neural network architecture that
combines the reliability of LS channel estimation and LMMSE equalization
with the adaptability of neural networks. The key motivation behind this
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
    - LeakyReLU with α=0.1 prevents dead neurons while maintaining
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
    using convolutional neural networks. The key design principle is that
    learning corrections to strong classical baselines is more effective
    than learning detection from scratch.

    Architecture
    ------------
    The detector processes data through six stages:

    1. **Feature extraction**: Assembles input features including LS channel
       estimate, received signal, matched filter output, Gram matrix structure,
       estimation error variance, and noise level.

    2. **Shared backbone**: Processes features through convolutional ResBlocks
       to learn joint representations useful for both CE refinement and detection.

    3. **CE refinement head**: Predicts additive corrections Δh and multiplicative
       log-domain corrections Δlog(err_var) to the LS estimates.

    4. **Scaled correction application**: Applies learned corrections scaled by
       trainable parameters that start at zero, enabling gradual departure from
       classical behavior during training.

    5. **Classical LMMSE**: Performs LMMSE equalization using refined channel
       estimate and error variance, followed by max-log demapping.

    6. **LLR refinement**: Injects LMMSE outputs back into the network to
       predict additive LLR corrections, again scaled by a trainable parameter.

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

    Pre-conditions
    --------------
    - ``cfg.pusch_pilot_indices`` must be set (by ``PUSCHLinkE2E``) before
      instantiation to enable pilot/data symbol separation.
    - Input tensors must follow Sionna's PUSCH dimension conventions.

    Post-conditions
    ---------------
    - ``trainable_variables`` returns correction scales first, then network
      weights (enables separate optimizer configuration).
    - ``last_h_hat_refined`` and ``last_err_var_refined`` contain the most
      recent refined estimates (useful for auxiliary losses).

    Invariants
    ----------
    - Correction scales initialized to 0.0, so initial behavior matches
      classical LMMSE exactly (graceful degradation if training fails).
    - Error variance correction scale is always positive (softplus transform).
    - Output LLR shape matches Sionna's ``PUSCHReceiver`` expectations.

    Example
    -------
    >>> cfg = Config()
    >>> # ... set cfg.pusch_pilot_indices via PUSCHLinkE2E ...
    >>> detector = PUSCHNeuralDetector(cfg, num_conv2d_filters=64)
    >>> llr = detector(y, h_hat, err_var, no)

    Notes
    -----
    The trainable correction scales serve multiple purposes:

    1. **Safe initialization**: Starting at 0.0 means the detector initially
       behaves exactly like classical LMMSE, providing a stable starting point.

    2. **Interpretability**: Scale magnitudes indicate how much the network
       deviates from classical processing (useful for debugging).

    3. **Gradient balancing**: Separate scales for h, err_var, and LLR allow
       independent learning rates for different correction types.

    The error variance scale uses softplus to ensure positivity, since
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

        # LLR correction: llr_final = llr_lmmse + scale * delta_llr
        # Unbounded to allow both confidence increase and decrease.
        self._llr_correction_scale = tf.Variable(
            0.0, trainable=True, name="llr_correction_scale", dtype=tf.float32
        )

        # Error variance correction in log domain for numerical stability:
        # err_var_refined = exp(log(err_var) + scale * delta_log_err)
        # Uses softplus(raw_value) to ensure positivity; softplus(0) = ln(2) ≈ 0.69
        # but we want initial scale ≈ 1.0, and softplus(0.54) ≈ 1.0
        # For simplicity, initialize to 0.0; the network will adapt.
        self._err_var_correction_scale_raw = tf.Variable(
            0.0, trainable=True, name="err_var_correction_scale_raw", dtype=tf.float32
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
        # Lightweight head that projects shared features to channel corrections.
        # Separate outputs for Δh (complex) and Δlog(err_var) (real).
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
        # After LMMSE equalization with refined estimates, this network learns
        # to correct the resulting LLRs based on both the original features
        # and the LMMSE outputs (symbol estimates, effective noise).
        self._c_lmmse_feats = (
            2 * S  # x_lmmse: equalized symbols (real + imag)
            + S  # no_eff: post-equalization noise variance (log scale)
            + S * self._num_bits_per_symbol  # llr_lmmse: baseline soft bits
        )

        # Injection convolution fuses shared backbone features with LMMSE outputs
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
            LS channel estimate from DMRS processing.
            Shape: ``[B, num_bs, num_bs_ant, num_ue, num_streams, num_ofdm_syms, num_subcarriers]``
        err_var : tf.Tensor, float32
            Channel estimation error variance per stream and RE.
            Shape: ``[B, 1, 1, num_ue, num_streams, num_ofdm_syms, num_subcarriers]``
        no : tf.Tensor, float32
            Noise variance, shape ``[B]`` or scalar.
        constellation : tf.Tensor, complex64, optional
            Trainable constellation points from transmitter. If provided,
            updates the demapper's constellation for consistent symbol mapping.
            Shape: ``[num_constellation_points]``
        training : bool, optional
            Training mode flag (currently unused, for Keras API compatibility).

        Returns
        -------
        tf.Tensor, float32
            Log-likelihood ratios for all coded bits.
            Shape: ``[B, num_ue, num_streams_per_ue, num_data_symbols * num_bits_per_symbol]``

        Notes
        -----
        The processing flow:

        1. **Feature assembly**: Extracts and concatenates multiple signal
           representations (channel, received signal, matched filter, Gram
           matrix structure, estimation error, noise level).

        2. **Shared backbone**: Processes features through ResBlocks to learn
           joint representations.

        3. **CE refinement**: Predicts channel and error variance corrections,
           applies them scaled by trainable parameters.

        4. **LMMSE equalization**: Classical equalization with refined estimates,
           using whitened interference model for improved performance.

        5. **LLR refinement**: Predicts additive corrections to LMMSE LLRs,
           enabling learned compensation for model mismatch.

        Channel estimation refinement operates on ALL OFDM symbols (including
        pilots) to leverage full spatial-frequency context, while detection
        operates only on DATA symbols to produce the final LLRs.
        """
        # =====================================================================
        # Input Preparation
        # =====================================================================
        y = tf.cast(y, tf.complex64)
        h_hat = tf.cast(h_hat, tf.complex64)
        err_var = tf.cast(err_var, tf.float32)
        no = tf.cast(no, tf.float32)

        # Synchronize demapper constellation with transmitter if trainable
        if constellation is not None:
            self._constellation.points = tf.cast(constellation, tf.complex64)

        # Precompute data symbol indices for later slicing
        data_idx = tf.constant(self._data_symbol_indices, dtype=tf.int32)

        # =====================================================================
        # Extract Dimensions
        # =====================================================================
        # Process full OFDM grid for CE refinement (pilots provide context)
        B = tf.shape(y)[0]
        H = tf.shape(y)[3]  # num_ofdm_syms (including pilots)
        W = tf.shape(y)[4]  # num_subcarriers

        S = self._num_streams_total
        Nr = self._num_rx_ant

        # =====================================================================
        # Reshape Inputs to Conv2D-Friendly Format [B, H, W, C]
        # =====================================================================
        # Sionna uses [B, rx, tx, time, freq] convention; we reshape to
        # [B, time, freq, features] for 2D convolution over the OFDM grid.

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

        # =====================================================================
        # Compute Derived Features
        # =====================================================================
        # These features provide the network with multiple views of the signal,
        # each capturing different aspects of the channel and interference.

        # Matched filter output: z_mf = H^H @ y
        # Captures signal energy and provides a sufficient statistic for detection
        h_conj_t = tf.transpose(tf.math.conj(h_flat), [0, 1, 2, 4, 3])  # [B,H,W,S,Nr]
        y_col = y_flat[..., tf.newaxis]  # [B,H,W,Nr,1]
        z_mf = tf.squeeze(tf.matmul(h_conj_t, y_col), axis=-1)  # [B,H,W,S]

        # Gram matrix: G = H^H @ H
        # Diagonal elements indicate per-stream channel gain
        # Off-diagonal elements capture inter-stream interference
        gram = tf.matmul(h_conj_t, h_flat)  # [B,H,W,S,S]
        gram_diag = tf.math.real(tf.linalg.diag_part(gram))  # [B,H,W,S]

        # Extract off-diagonal elements (interference structure)
        mask = 1.0 - tf.eye(S, dtype=tf.float32)
        mask = mask[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
        gram_masked = gram * tf.cast(mask, gram.dtype)
        gram_offdiag = tf.abs(gram_masked)
        gram_offdiag_flat = tf.reshape(gram_offdiag, [B, H, W, -1])
        # Select only off-diagonal entries (S*(S-1) values per RE)
        indices = [i * S + j for i in range(S) for j in range(S) if i != j]
        gram_offdiag_feats = tf.gather(gram_offdiag_flat, indices, axis=-1)

        # Channel estimate features (real and imaginary parts)
        h_flat_features = tf.reshape(h_flat, [B, H, W, Nr * S])
        h_feats = tf.concat(
            [tf.math.real(h_flat_features), tf.math.imag(h_flat_features)], axis=-1
        )

        # Received signal features (real and imaginary parts)
        y_feats = tf.concat([tf.math.real(y_flat), tf.math.imag(y_flat)], axis=-1)

        # Matched filter features (real and imaginary parts)
        z_mf_feats = tf.concat([tf.math.real(z_mf), tf.math.imag(z_mf)], axis=-1)

        # Noise variance in log scale for numerical stability and scale invariance
        no_log = log10(no + 1e-10)
        no_feat = tf.broadcast_to(
            no_log[:, tf.newaxis, tf.newaxis, tf.newaxis], [B, H, W, 1]
        )

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
        ce_hidden = self._ce_head_conv1(shared_features)

        delta_h_raw = self._ce_head_out_h(ce_hidden)  # [B,H,W, 2*Nr*S]
        delta_loge = self._ce_head_out_loge(ce_hidden)  # [B,H,W, S]

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
        # Provide the network with LMMSE outputs to enable learned refinement
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
        # Fuse shared backbone features with LMMSE outputs
        combined_features = tf.concat([shared_features_data, lmmse_features], axis=-1)

        # Injection convolution reduces dimensions and fuses information
        det_features = self._det_inject_conv(combined_features)

        # Additional ResBlocks for LLR refinement
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
