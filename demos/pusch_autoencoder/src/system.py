# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
End-to-end PUSCH link simulation for autoencoder training and evaluation.

Provides the core class that connects trainable transmitter and receiver components
through a realistic ray-traced channel. Supports both baseline (classical LMMSE)
and autoencoder (neural detector) configurations, enabling direct comparison of
learned vs. classical approaches.

The system implements the full uplink signal processing chain:
TX bits -> Encoding -> Modulation -> OFDM -> Channel -> OFDM Rx -> Detection -> Decoding
"""

import numpy as np
import tensorflow as tf

from sionna.phy.channel import (
    OFDMChannel,
    subcarrier_frequencies,
    cir_to_ofdm_channel,
    ApplyOFDMChannel,
)
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.utils import ebnodb2no
from sionna.phy.ofdm import LinearDetector
from sionna.phy.mimo import StreamManagement

from .config import Config
from .pusch_trainable_transmitter import PUSCHTrainableTransmitter
from .pusch_neural_detector import PUSCHNeuralDetector
from .pusch_trainable_receiver import PUSCHTrainableReceiver


class PUSCHLinkE2E(tf.keras.Model):
    r"""
    End-to-end differentiable PUSCH link model for MU-MIMO autoencoder training.

    This class simulates a complete 5G NR PUSCH uplink that can operate in two modes:

    1. **Baseline mode** (``use_autoencoder=False``): Uses standard QAM constellation
       with LS channel estimation + LMMSE equalization for BER/BLER benchmarking.

    2. **Autoencoder mode** (``use_autoencoder=True``): Uses trainable constellation
       points and a trainable mixed-mode receiver, enabling end-to-end gradient-based
       optimization of both transmitter and receiver.

    The model supports both perfect and imperfect CSI scenarios, where imperfect
    CSI uses LS channel estimation.

    Parameters
    ----------
    channel_model : tuple or CIRDataset
        For baseline mode: ``CIRDataset`` object for on-demand CIR generation.
        For autoencoder mode: tuple ``(a, tau)`` of pre-loaded CIR tensors.
    perfect_csi : bool
        If ``True``, provides ground-truth channel to the receiver.
        If ``False``, receiver performs LS channel estimation.
    use_autoencoder : bool
        If ``True``, uses trainable transmitter and neural detector.
        If ``False``, uses standard PUSCH TX/RX with LMMSE detection.
    training : bool
        If ``True``, ``call()`` returns the training loss (BCE + regularization).
        If ``False``, ``call()`` returns ``(bits, bits_hat)`` for BER evaluation.
    const_reg_weight : float
        Weight for constellation regularization loss. Prevents constellation
        collapse during training by penalizing points that are too close.
    const_d_min : float
        Minimum distance threshold for constellation regularization. Points
        closer than this distance incur a penalty.
    config : Config, optional
        System configuration. Defaults to ``Config()`` if not provided.
        Use this to customize system parameters like ``num_bs_ant``.

    Notes
    -----
    - For autoencoder mode, ``channel_model`` must be a tuple ``(a, tau)`` where:
      - ``a``: Complex CIR coefficients with shape
               ``[num_samples, num_bs, num_bs_ant, num_ue, num_ue_ant, num_paths, num_time_steps]``
      - ``tau``: Path delays with shape
                 ``[num_samples, num_bs, num_ue, num_paths]``
    - For baseline mode, ``channel_model`` must be a valid ``CIRDataset``.
    - ``self._cfg`` contains PUSCH resource grid information after construction.
    - ``self.trainable_variables`` returns all trainable weights (TX + RX).
    - In training mode, ``call()`` returns a scalar loss tensor.
    - In inference mode, ``call()`` returns ``(b, b_hat)`` bit tensors.
    - The PUSCH configuration (PRBs, MCS, layers) remains fixed after init.
    - Constellation normalization maintains unit average power.
    - Channel model type (tuple vs CIRDataset) determines internal processing path.

    Example
    -------
    >>> # Autoencoder training setup
    >>> cir_manager = CIRManager()
    >>> a, tau = cir_manager.load_from_tfrecord(group_for_mumimo=True)
    >>> model = PUSCHLinkE2E((a, tau), perfect_csi=False,
    ...                       use_autoencoder=True, training=True)
    >>> loss = model(batch_size=32, ebno_db=10.0)

    >>> # Baseline evaluation setup
    >>> channel_model = cir_manager.build_channel_model()
    >>> model = PUSCHLinkE2E(channel_model, perfect_csi=False,
    ...                       use_autoencoder=False)
    >>> b, b_hat = model(batch_size=32, ebno_db=10.0)

    Notes
    -----
    The choice between ``OFDMChannel`` (baseline) and ``ApplyOFDMChannel``
    (autoencoder) is intentional: ``ApplyOFDMChannel`` accepts pre-computed
    frequency-domain channel matrices, enabling efficient batched training
    with pre-loaded CIR data, while ``OFDMChannel`` wraps a ``CIRDataset``
    for on-demand generation during Monte Carlo simulation. The channel in
    the case of the latter is a Python generator through which gradients
    cannot be propagated.
    """

    def __init__(
        self,
        channel_model,
        perfect_csi,
        use_autoencoder=False,
        training=False,
        const_reg_weight=0.1,
        const_d_min=0.35,
        config=None,
    ):
        super().__init__()

        self._training = training

        # Constellation regularization prevents collapse to lower-order modulation
        # (e.g., 16-QAM collapsing to QPSK) during joint TX/RX optimization.
        self._const_reg_weight = const_reg_weight
        self._const_d_min = const_d_min

        self._channel_model = channel_model
        self._perfect_csi = perfect_csi
        self._use_autoencoder = use_autoencoder

        # Centralized config to enforce a single system configuration
        # Use provided config or create default
        self._cfg = config if config is not None else Config()

        # Cache frequently-used config values for cleaner code
        self._num_prb = self._cfg.num_prb
        self._mcs_index = self._cfg.mcs_index
        self._num_layers = self._cfg.num_layers
        self._mcs_table = self._cfg.mcs_table
        self._domain = self._cfg.domain

        self._num_ue_ant = self._cfg.num_ue_ant
        self._num_ue = self._cfg.num_ue
        # Subcarrier spacing must match the value used during CIR generation
        # to ensure correct Doppler and delay spread scaling.
        self._subcarrier_spacing = self._cfg.subcarrier_spacing

        # =====================================================================
        # PUSCH Configuration for First UE
        # =====================================================================
        # The first UE's config serves as the template; others are cloned with
        # different DMRS port assignments to enable MU-MIMO multiplexing.
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = self._subcarrier_spacing / 1000.0
        pusch_config.carrier.n_size_grid = self._num_prb
        pusch_config.num_antenna_ports = self._num_ue_ant
        pusch_config.num_layers = self._num_layers
        # Codebook precoding with TPMI=1 selects a fixed precoding matrix,
        # simplifying the autoencoder by removing precoder optimization.
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 1
        # DMRS configuration: Type 1, single-symbol, with additional position
        # for improved channel tracking in time-varying scenarios.
        pusch_config.dmrs.dmrs_port_set = list(range(self._num_layers))
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.additional_position = 1
        # 2 CDM groups without data reserves sufficient DMRS density for
        # reliable channel estimation across 4 co-scheduled UEs.
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.tb.mcs_index = self._mcs_index
        pusch_config.tb.mcs_table = self._mcs_table

        # Propagate PUSCH grid info to Config so neural detector can access it
        self._cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
        self._cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
        self._cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

        # =====================================================================
        # Create PUSCH Configs for All UEs
        # =====================================================================
        # Each UE gets a unique DMRS port set to enable orthogonal pilot
        # transmission and per-UE channel estimation at the BS.
        pusch_configs = [pusch_config]
        for i in range(1, self._num_ue):
            pc = pusch_config.clone()
            pc.dmrs.dmrs_port_set = list(
                range(i * self._num_layers, (i + 1) * self._num_layers)
            )
            pusch_configs.append(pc)

        # =====================================================================
        # Transmitter Setup
        # =====================================================================
        # Autoencoder uses trainable constellation; baseline uses fixed QAM.
        self._pusch_transmitter = (
            PUSCHTrainableTransmitter(
                pusch_configs, output_domain=self._domain, training=self._training
            )
            if self._use_autoencoder
            else PUSCHTransmitter(pusch_configs, output_domain=self._domain)
        )
        self._cfg.resource_grid = self._pusch_transmitter.resource_grid

        # =====================================================================
        # Detector Setup
        # =====================================================================
        # Stream management defines the RX-TX association matrix for MU-MIMO.
        # All UEs are associated with the single BS (all-ones matrix).
        rx_tx_association = np.ones([1, self._num_ue], bool)
        stream_management = StreamManagement(rx_tx_association, self._num_layers)

        if self._use_autoencoder:
            self._detector = PUSCHNeuralDetector(self._cfg)
        else:
            # LMMSE with max-log demapping provides the classical baseline
            self._detector = LinearDetector(
                equalizer="lmmse",
                output="bit",
                demapping_method="maxlog",
                resource_grid=self._pusch_transmitter.resource_grid,
                stream_management=stream_management,
                constellation_type="qam",
                num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol,
            )

        # =====================================================================
        # Receiver Setup
        # =====================================================================
        receiver = PUSCHTrainableReceiver if self._use_autoencoder else PUSCHReceiver
        receiver_kwargs = {
            "mimo_detector": self._detector,
            "input_domain": self._domain,
            "pusch_transmitter": self._pusch_transmitter,
        }

        # Perfect CSI bypasses channel estimation entirely
        if self._perfect_csi:
            receiver_kwargs["channel_estimator"] = "perfect"

        # Training flag controls whether receiver returns LLRs or decoded bits
        if self._use_autoencoder:
            receiver_kwargs["training"] = training

        self._pusch_receiver = receiver(**receiver_kwargs)

        # =====================================================================
        # Channel Setup
        # =====================================================================
        # Autoencoder mode: ApplyOFDMChannel accepts pre-computed H matrices,
        # enabling efficient training with pre-loaded CIR tensors.
        # Baseline mode: OFDMChannel wraps CIRDataset for Monte Carlo sampling.
        if self._use_autoencoder:
            self._frequencies = subcarrier_frequencies(
                self._pusch_transmitter.resource_grid.fft_size,
                self._pusch_transmitter.resource_grid.subcarrier_spacing,
            )
            self._channel = ApplyOFDMChannel(add_awgn=True)

            if self._training:
                # BCE loss on coded bits (LLRs vs. true bits) for training
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
        """
        Collect all trainable variables from transmitter and receiver.

        Returns
        -------
        list
            Combined list of trainable ``tf.Variable`` objects from:

            - Transmitter: constellation real/imag coordinates
            - Receiver: neural detector weights and correction scales
        """
        vars_ = []
        if hasattr(self, "_pusch_transmitter"):
            vars_ += list(self._pusch_transmitter.trainable_variables)
        if hasattr(self, "_pusch_receiver"):
            vars_ += list(self._pusch_receiver.trainable_variables)
        return vars_

    @property
    def constellation(self):
        """
        Get the current normalized constellation points.

        Returns
        -------
        tf.Tensor
            Complex tensor of shape ``[num_points]`` with unit average power.
            For 16-QAM, this is 16 complex values.
        """
        return self._pusch_transmitter.get_normalized_constellation()

    def get_constellation_min_distance(self):
        """
        Compute the minimum Euclidean distance between constellation points.

        This metric indicates constellation quality: larger minimum distance
        generally improves noise resilience. For unit-power 16-QAM, the
        theoretical minimum distance is ``2/sqrt(10) ≈ 0.632``.

        Returns
        -------
        tf.Tensor
            Scalar tensor containing the minimum pairwise distance.

        Notes
        -----
        The diagonal of the pairwise distance matrix (self-distances) is
        masked with a large value to exclude it from the minimum computation.
        """
        points = tf.stack(
            [tf.math.real(self.constellation), tf.math.imag(self.constellation)],
            axis=-1,
        )
        diff = points[:, tf.newaxis, :] - points[tf.newaxis, :, :]
        distances = tf.norm(diff, axis=-1)
        # Mask diagonal with large value to exclude self-distances
        mask = 1.0 - tf.eye(tf.shape(points)[0])
        distances = distances + (1.0 - mask) * 1e6
        return tf.reduce_min(distances)

    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        """
        Execute forward pass through the end-to-end PUSCH link.

        Parameters
        ----------
        batch_size : int
            Number of transport blocks to simulate in parallel.
        ebno_db : tf.Tensor
            Energy per bit to noise power spectral density ratio in dB.
            Can be scalar (same SNR for all samples) or vector ``[batch_size]``.

        Returns
        -------
        tf.Tensor or tuple
            - **Training mode**: Scalar loss tensor (BCE + constellation regularization)
            - **Inference mode**: Tuple ``(b, b_hat)`` where:
              - ``b``: Original bits, shape ``[batch_size, num_ue, tb_size]``
              - ``b_hat``: Detected bits, same shape as ``b``

        Notes
        -----
        JIT compilation is disabled (``jit_compile=False``) because the neural
        detector uses dynamic shapes and control flow that are incompatible
        with XLA compilation.
        """
        # =====================================================================
        # Transmitter Processing
        # =====================================================================
        if self._use_autoencoder:
            # Returns: mapped symbols, OFDM signal, original bits, coded bits
            x_map, x, b, c = self._pusch_transmitter(batch_size)
        else:
            # Baseline returns only OFDM signal and original bits
            x, b = self._pusch_transmitter(batch_size)

        # Convert Eb/N0 to noise variance using coderate and bits/symbol
        no = ebnodb2no(
            ebno_db,
            self._pusch_transmitter._num_bits_per_symbol,
            self._pusch_transmitter._target_coderate,
            self._pusch_transmitter.resource_grid,
        )

        # =====================================================================
        # Channel Application
        # =====================================================================
        if self._use_autoencoder:
            # Unpack pre-loaded CIR tensors and sample a random batch
            a, tau = self._channel_model
            num_samples = tf.shape(a)[0]

            # Random sampling enables diverse channel realizations per batch
            # without requiring a tf.data.Dataset pipeline.
            idx = tf.random.shuffle(tf.range(num_samples))[:batch_size]

            a_batch = tf.gather(a, idx, axis=0)
            tau_batch = tf.gather(tau, idx, axis=0)

            # Convert time-domain CIR to frequency-domain channel matrix.
            # Normalization ensures consistent SNR interpretation.
            h = cir_to_ofdm_channel(
                self._frequencies, a_batch, tau_batch, normalize=True
            )

            y = self._channel(x, h, no)
        else:
            # OFDMChannel handles CIR sampling internally via CIRDataset
            y, h = self._channel(x, no)

        # =====================================================================
        # Receiver Processing
        # =====================================================================
        if self._use_autoencoder and self._training:
            # Training: compute loss from LLRs
            if self._perfect_csi:
                llr = self._pusch_receiver(y, no, h)
            else:
                llr = self._pusch_receiver(y, no)

            # Binary cross-entropy between coded bits and soft LLR estimates
            bce_loss = self._bce(c, llr)

            # Constellation regularization prevents collapse during joint
            # TX/RX optimization by penalizing points that are too close.
            const_reg_loss = constellation_regularization_loss(
                tf.math.real(self.constellation),
                tf.math.imag(self.constellation),
                d_min_weight=1.0,
                grid_weight=0.0,
                uniformity_weight=0.0,
                d_min=self._const_d_min,
            )

            # Weighted combination: primary objective + regularization
            loss = bce_loss + self._const_reg_weight * const_reg_loss
            return loss
        else:
            # Inference: decode bits and return for BER/BLER computation
            if self._perfect_csi:
                b_hat = self._pusch_receiver(y, no, h)
            else:
                b_hat = self._pusch_receiver(y, no)
            return b, b_hat


# =============================================================================
# Constellation Regularization Loss Functions
# =============================================================================
# These losses prevent the trainable constellation from degenerating during
# joint TX/RX optimization. Without regularization, the neural receiver
# can compensate for any constellation shape, eliminating gradient
# pressure on the transmitter and causing collapse to lower-order modulations.


def min_distance_loss(points_r, points_i, d_min=0.4, margin=0.1):
    r"""
    Penalize constellation points that are closer than a minimum distance.

    This is the primary regularizer to prevent constellation collapse during
    autoencoder training. It implements a soft hinge loss that activates when
    any pair of points is closer than ``d_min + margin``.

    Parameters
    ----------
    points_r : tf.Tensor
        Real parts of constellation points, shape ``[num_points]``.
    points_i : tf.Tensor
        Imaginary parts of constellation points, shape ``[num_points]``.
    d_min : float
        Minimum allowed distance between any two points. For unit-power
        16-QAM, the nominal minimum distance is approximately,
        ``2/sqrt(10)`` or ``0.632``. Setting ``d_min=0.4`` provides
        margin while preventing QPSK collapse.
    margin : float
        Soft margin for the hinge loss. The penalty activates when distance
        is less than ``d_min + margin``, providing smooth gradients.

    Returns
    -------
    tf.Tensor
        Scalar loss value, averaged over all point pairs.

    Notes
    -----
    The hinge formulation ``relu(d_min + margin - distance)`` ensures:

    1. Zero gradient when constraints are satisfied (no unnecessary pressure)
    2. Linear penalty growth as violations increase (stable optimization)
    3. Smooth transition at the boundary (no gradient discontinuities)
    """
    points = tf.stack([points_r, points_i], axis=-1)  # [N, 2]

    # Pairwise distance matrix: diff[i,j] = points[i] - points[j]
    diff = points[:, tf.newaxis, :] - points[tf.newaxis, :, :]  # [N, N, 2]
    distances = tf.norm(diff, axis=-1)  # [N, N]

    # Exclude diagonal (self-distances are always zero)
    mask = 1.0 - tf.eye(tf.shape(points)[0])
    distances = distances * mask + tf.eye(tf.shape(points)[0]) * 1e6

    # Hinge loss: penalize distances below threshold
    violations = tf.nn.relu(d_min + margin - distances)

    # Average over all unique pairs (N*(N-1) off-diagonal entries)
    num_pairs = tf.cast(tf.shape(points)[0] * (tf.shape(points)[0] - 1), tf.float32)
    loss = tf.reduce_sum(violations) / num_pairs

    return loss


def grid_structure_loss(points_r, points_i):
    r"""
    Encourage constellation points to align with a regular QAM grid.

    This loss penalizes deviation from the standard 16-QAM grid positions,
    which can be useful when the goal is to learn a refined (but still
    grid-like) constellation rather than an arbitrary geometry.

    Parameters
    ----------
    points_r : tf.Tensor
        Real parts of constellation points, shape ``[num_points]``.
    points_i : tf.Tensor
        Imaginary parts of constellation points, shape ``[num_points]``.

    Returns
    -------
    tf.Tensor
        Scalar loss value (sum of I and Q deviations from grid).

    Notes
    -----
    The target grid levels are ``{-3, -1, +1, +3} / sqrt(10)`` for unit-power
    16-QAM. Each point is penalized by its distance to the nearest grid level
    in both I and Q dimensions independently.

    This loss is typically used with low weight (e.g., 0.1-0.5) alongside
    ``min_distance_loss`` to encourage structure while allowing optimization.
    """
    # Standard 16-QAM grid levels normalized to unit power
    scale = tf.sqrt(10.0)
    grid_levels = tf.constant([-3.0, -1.0, 1.0, 3.0], dtype=tf.float32) / scale

    def snap_to_grid_loss(coords):
        """Compute mean distance to nearest grid level."""
        coords_expanded = coords[:, tf.newaxis]  # [N, 1]
        grid_expanded = grid_levels[tf.newaxis, :]  # [1, 4]
        distances = tf.abs(coords_expanded - grid_expanded)  # [N, 4]
        min_distances = tf.reduce_min(distances, axis=-1)  # [N]
        return tf.reduce_mean(min_distances)

    loss_r = snap_to_grid_loss(points_r)
    loss_i = snap_to_grid_loss(points_i)

    return loss_r + loss_i


def uniformity_loss(points_r, points_i):
    r"""
    Encourage uniform point spacing by maximizing the minimum distance.

    Unlike ``min_distance_loss`` which only penalizes violations of a threshold,
    this loss continuously pushes points apart to maximize the minimum pairwise
    distance. This can help spread points more evenly in the constellation space.

    Parameters
    ----------
    points_r : tf.Tensor
        Real parts of constellation points, shape ``[num_points]``.
    points_i : tf.Tensor
        Imaginary parts of constellation points, shape ``[num_points]``.

    Returns
    -------
    tf.Tensor
        Scalar loss value (negative softmin of distances, to be minimized).

    Notes
    -----
    The loss uses a softmin (negative log-sum-exp) approximation of the minimum
    distance for smooth gradients. The temperature parameter (0.1) controls
    the smoothness: lower temperature -> closer to true min but sharper gradients.

    Minimizing this loss is equivalent to maximizing the minimum distance,
    which pushes points apart without requiring a specific target structure.
    """
    points = tf.stack([points_r, points_i], axis=-1)  # [N, 2]

    diff = points[:, tf.newaxis, :] - points[tf.newaxis, :, :]
    distances = tf.norm(diff, axis=-1)

    # Exclude diagonal entries
    mask = 1.0 - tf.eye(tf.shape(points)[0])
    distances = distances + (1.0 - mask) * 1e6

    # Softmin approximation for differentiable minimum
    # softmin(x) = -T * log(sum(exp(-x/T))) approaches min(x) as T -> 0
    temperature = 0.1
    softmin = -temperature * tf.reduce_logsumexp(-distances / temperature)

    # Return negative since we want to maximize min distance (minimize loss)
    return -softmin


def constellation_regularization_loss(
    points_r,
    points_i,
    d_min_weight=1.0,
    grid_weight=0.0,
    uniformity_weight=0.0,
    d_min=0.4,
):
    r"""
    Combined constellation regularization loss with configurable components.

    This function provides a flexible interface to combine multiple regularization
    objectives for constellation optimization. The weights control the relative
    importance of each term.

    Parameters
    ----------
    points_r : tf.Tensor
        Real parts of constellation points, shape ``[num_points]``.
    points_i : tf.Tensor
        Imaginary parts of constellation points, shape ``[num_points]``.
    d_min_weight : float
        Weight for minimum distance loss. Set to 1.0 for collapse prevention.
    grid_weight : float
        Weight for grid structure loss. Set to 0.0 to allow free-form optimization,
        or 0.1-0.5 to encourage QAM-like structure.
    uniformity_weight : float
        Weight for uniformity loss. Set to 0.0 for standard training, or
        0.1-0.5 to encourage even point spacing.
    d_min : float
        Minimum distance threshold passed to ``min_distance_loss``.

    Returns
    -------
    tf.Tensor
        Combined scalar loss value.

    Example
    -------
    >>> # Prevent collapse only (default for autoencoder training)
    >>> loss = constellation_regularization_loss(pr, pi, d_min_weight=1.0)

    >>> # Encourage QAM structure while preventing collapse
    >>> loss = constellation_regularization_loss(pr, pi, d_min_weight=1.0,
    ...                                          grid_weight=0.3)

    Notes
    -----
    Recommended configurations:

    - **Collapse prevention only**: ``d_min_weight=1.0``, others=0
    - **Structured optimization**: ``d_min_weight=1.0``, ``grid_weight=0.3``
    - **Maximum spreading**: ``d_min_weight=1.0``, ``uniformity_weight=0.3``
    """
    total_loss = 0.0

    if d_min_weight > 0:
        total_loss += d_min_weight * min_distance_loss(points_r, points_i, d_min=d_min)

    if grid_weight > 0:
        total_loss += grid_weight * grid_structure_loss(points_r, points_i)

    if uniformity_weight > 0:
        total_loss += uniformity_weight * uniformity_loss(points_r, points_i)

    return total_loss
