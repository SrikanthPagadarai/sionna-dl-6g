# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Baseline receiver for MIMO-OFDM neural receiver demo.

Implements a conventional receiver chain that serves as the
performance baseline for comparison against the neural receiver:

    Channel Estimation -> LMMSE Equalization -> Soft Demapping -> LDPC Decoding

The receiver supports two CSI modes:

1. **Perfect CSI** (``cfg.perfect_csi=True``): Uses ground-truth channel
   coefficients from the CSI object. Establishes an upper bound on achievable
   performance with ideal channel knowledge.

2. **Estimated CSI** (``cfg.perfect_csi=False``): Uses LS channel estimation
   from pilot symbols with nearest-neighbor interpolation. Represents realistic
   receiver performance with practical channel estimation.

This baseline receiver is used during inference to compare BER/BLER against
the neural receiver, demonstrating the gains from learned signal processing.
"""

import tensorflow as tf
from typing import Dict, Any
from sionna.phy.ofdm import LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mapping import Demapper
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from .config import Config
from .csi import CSI


class Rx:
    """
    Conventional MIMO-OFDM receiver with LS estimation and LMMSE equalization.

    Implements the standard receive processing chain used as a baseline for
    neural receiver comparison. The chain consists of:

    1. **Channel Estimation**: LS estimation at pilot positions with
       nearest-neighbor interpolation to data positions.

    2. **LMMSE Equalization**: Linear minimum mean square error spatial
       filtering to separate MIMO streams.

    3. **Soft Demapping**: APP (a posteriori probability) demapper producing
       soft LLR values for each coded bit.

    4. **LDPC Decoding**: 5G NR LDPC decoder producing hard bit decisions.

    Parameters
    ----------
    cfg : Config
        Configuration object containing modulation, coding, and CSI settings.

    csi : CSI
        Channel state information object providing ground-truth channel
        coefficients for perfect-CSI mode.

    Attributes
    ----------
    _cfg : Config
        Reference to configuration object.

    _csi : CSI
        Reference to CSI object for perfect-CSI path.

    Note
    ----
    The receiver shares the ``CSI`` instance with the transmit chain to ensure
    the same channel realization is used for both transmission and perfect-CSI
    reception. This is critical for fair performance evaluation.

    Example
    -------
    >>> cfg = Config(perfect_csi=False)
    >>> csi = CSI(cfg)
    >>> rx = Rx(cfg, csi)
    >>> out = rx(y, h_freq, no)
    >>> decoded_bits = out["b_hat"]
    """

    def __init__(self, cfg: Config, csi: CSI):
        """
        Initialize baseline receiver components.

        Parameters
        ----------
        cfg : Config
            Configuration specifying modulation, coding, and CSI mode.

        csi : CSI
            CSI object for accessing ground-truth channel (perfect-CSI mode)
            and the ``remove_nulled_scs`` utility.

        Post-conditions
        ---------------
        - ``_ce`` is configured for LS estimation with NN interpolation.
        - ``_eq`` is configured for LMMSE equalization matching the resource grid.
        - ``_demapper`` uses APP demapping for the configured constellation.
        - ``_decoder`` is paired with an encoder of matching (k, n) dimensions.
        """
        self._cfg = cfg
        self._csi = csi

        # LS channel estimator with nearest-neighbor interpolation
        self._ce = LSChannelEstimator(self._cfg.rg, interpolation_type="nn")

        # LMMSE equalizer for MIMO stream separation
        self._eq = LMMSEEqualizer(self._cfg.rg, self._cfg.sm)

        # APP demapper: computes exact LLRs given constellation and noise
        self._demapper = Demapper(
            "app", self._cfg.modulation, self._cfg.num_bits_per_symbol
        )

        # LDPC decoder paired with encoder to ensure matching code structure
        self._decoder = LDPC5GDecoder(
            LDPC5GEncoder(self._cfg.k, self._cfg.n), hard_out=True
        )

    @tf.function
    def __call__(
        self,
        y: tf.Tensor,
        h_freq: tf.Tensor,
        no: tf.Tensor,
    ) -> Dict[str, Any]:
        """
        Process received signal through baseline receiver chain.

        Parameters
        ----------
        y : tf.Tensor, complex64, [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
            Received OFDM signal after channel and noise.

        h_freq : tf.Tensor, complex64, [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
            Ground-truth frequency-domain channel response. Used only when
            ``cfg.perfect_csi=True``; otherwise channel is estimated from pilots.

        no : tf.Tensor, float32, [batch] or scalar
            Noise power spectral density for equalization and demapping.

        Returns
        -------
        Dict[str, tf.Tensor]
            Dictionary containing intermediate and final outputs:

            - ``"h_hat"``: Estimated (or perfect) channel coefficients,
              shape [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_data_symbols].

            - ``"err_var"``: Channel estimation error variance,
              scalar 0.0 for perfect CSI, estimated otherwise.

            - ``"x_hat"``: Equalized symbols after LMMSE filtering,
              shape [batch, num_tx, num_streams, num_data_symbols].

            - ``"no_eff"``: Effective noise variance after equalization,
              used by demapper for accurate LLR computation.

            - ``"llr"``: Log-likelihood ratios for coded bits,
              shape [batch, num_tx, num_streams, n].

            - ``"b_hat"``: Decoded information bits after LDPC decoding,
              shape [batch, num_tx, num_streams, k].

        Pre-conditions
        --------------
        - ``y`` must contain valid received samples including pilot positions.
        - ``no`` must be positive (zero causes division issues in LMMSE).
        - For perfect CSI mode, ``h_freq`` must match the channel used in transmission.

        Post-conditions
        ---------------
        - ``h_hat`` contains channel estimates on data subcarriers only
          (guard bands and pilots removed).
        - ``b_hat`` contains hard binary decisions in {0, 1}.
        - All intermediate tensors are available for analysis/debugging.

        Note
        ----
        The perfect-CSI path uses ``csi.remove_nulled_scs`` to extract channel
        coefficients on active subcarriers, matching the format expected by
        the LMMSE equalizer. The error variance is set to zero since there
        is no estimation error with perfect knowledge.
        """
        # =====================================================================
        # Channel Estimation
        # Perfect CSI: use ground truth; Estimated: LS + interpolation
        # =====================================================================
        if self._cfg.perfect_csi:
            # Extract channel on active subcarriers (remove guards and DC null)
            # Error variance is zero when CSI is perfect
            h_hat = self._csi.remove_nulled_scs(h_freq)
            err_var = tf.cast(0.0, tf.float32)
        else:
            # LS estimation at pilots, NN interpolation to data positions
            # Returns both estimates and estimation error variance
            h_hat, err_var = self._ce(y, no)

        # =====================================================================
        # Equalization, Demapping, and Decoding
        # =====================================================================
        # LMMSE equalization: separates MIMO streams, outputs soft symbols
        # no_eff is the effective noise after equalization (for demapper)
        x_hat, no_eff = self._eq(y, h_hat, err_var, no)

        # APP demapping: converts soft symbols to bit LLRs
        # Uses constellation geometry and effective noise variance
        llr = self._demapper(x_hat, no_eff)

        # LDPC decoding: iterative belief propagation to hard decisions
        b_hat = self._decoder(llr)

        return {
            "h_hat": h_hat,
            "err_var": err_var,
            "x_hat": x_hat,
            "no_eff": no_eff,
            "llr": llr,
            "b_hat": b_hat,
        }
