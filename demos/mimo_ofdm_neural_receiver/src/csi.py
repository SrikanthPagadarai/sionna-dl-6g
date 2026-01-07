# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Generate Channel State Information (CSI).

Provides the ``CSI`` class which encapsulates all channel-related
setup and generation for a MIMO-OFDM simulation. It handles:

1. Antenna array configuration (UT and BS arrays with dual polarization)
2. 3GPP CDL channel model instantiation
3. Channel impulse response (CIR) generation
4. Conversion from CIR to frequency-domain channel response

The CSI object is shared across Tx, Channel, and Rx components to ensure
they all operate on the same channel realization within a simulation iteration.
This is critical for fair comparison between perfect-CSI and estimated-CSI
receiver paths.
"""

import os

# Suppress TensorFlow logging before import to avoid cluttering output
# Level 0 = all messages; used here for debugging, change to 2+ for production
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import sionna.phy  # noqa: E402
import tensorflow as tf  # noqa: E402
from sionna.phy.channel.tr38901 import CDL, AntennaArray  # noqa: E402
from sionna.phy.ofdm import RemoveNulledSubcarriers  # noqa: E402
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel  # noqa: E402
from .config import Config  # noqa: E402


class CSI:
    """
    Channel State Information generator for MIMO-OFDM simulations.

    This class manages the complete channel generation pipeline from antenna
    array configuration through to frequency-domain channel coefficients.
    A single ``CSI`` instance should be shared across all pipeline components
    (Tx, Channel, Rx) to ensure consistent channel realizations.

    The channel generation follows the 3GPP TR 38.901 CDL model:

    1. **Antenna Arrays**: Configured with dual cross-polarized elements
       following the 38.901 antenna pattern specification.

    2. **CDL Channel**: Generates delay-domain channel impulse response
       with configurable model (A-E), delay spread, and Doppler.

    3. **Frequency Response**: CIR is converted to frequency-domain via
       ``cir_to_ofdm_channel`` for OFDM processing.

    Parameters
    ----------
    cfg : Config
        Configuration object containing PHY parameters (carrier frequency,
        antenna counts, CDL model selection, etc.).

    Attributes
    ----------
    cfg : Config
        Reference to the configuration object.

    remove_nulled_scs : RemoveNulledSubcarriers
        Utility layer for extracting channel coefficients on active
        (non-nulled) subcarriers. Used by Rx for perfect-CSI path.

    Note
    ----
    The ``build()`` method must be called once per simulation batch to
    generate a new channel realization. The returned ``h_freq`` tensor
    should be passed to both the Channel and Rx components.

    Example
    -------
    >>> cfg = Config(cdl_model="C", carrier_frequency=3.5e9)
    >>> csi = CSI(cfg)
    >>> h_freq = csi.build(batch_size=32)
    >>> # Use h_freq with Channel and Rx
    """

    def __init__(self, cfg: Config):
        """
        Initialize CSI generator with antenna arrays and CDL channel model.

        Parameters
        ----------
        cfg : Config
            Configuration object specifying PHY-layer parameters.

        Post-conditions
        ---------------
        - Sionna's global RNG seed is set to ``cfg.seed`` for reproducibility.
        - UT and BS antenna arrays are configured with dual cross-polarization.
        - CDL channel model is instantiated with configured parameters.
        - Subcarrier frequencies are pre-computed for CIR-to-CFR conversion.
        - ``remove_nulled_scs`` is ready for perfect-CSI extraction.
        """
        self.cfg = cfg

        # Set global seed for deterministic channel realizations across runs
        # This affects CDL's internal random path gains and delays
        sionna.phy.config.seed = int(self.cfg._seed)

        # =====================================================================
        # Antenna Array Configuration
        # Using dual cross-polarized arrays per 3GPP 38.901 specification.
        # num_cols = num_ant/2 because dual polarization doubles effective elements.
        # Cross-polarization provides diversity gain in fading channels.
        # =====================================================================
        self._ut_array = AntennaArray(
            num_rows=1,
            num_cols=int(self.cfg.num_ut_ant / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.cfg.carrier_frequency,
        )
        self._bs_array = AntennaArray(
            num_rows=1,
            num_cols=int(self.cfg.num_bs_ant / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.cfg.carrier_frequency,
        )

        # =====================================================================
        # CDL Channel Model
        # Models A-C: NLOS with increasing delay spread
        # Models D-E: LOS with Rician fading
        # min_speed sets the minimum UE velocity for Doppler calculation
        # =====================================================================
        self._cdl = CDL(
            model=self.cfg.cdl_model,
            delay_spread=self.cfg.delay_spread,
            carrier_frequency=self.cfg.carrier_frequency,
            ut_array=self._ut_array,
            bs_array=self._bs_array,
            direction=self.cfg.direction,
            min_speed=self.cfg.speed,
        )

        # =====================================================================
        # Subcarrier Frequencies
        # Pre-computed frequency grid for converting CIR to frequency response.
        # These frequencies are relative to the carrier (baseband equivalent).
        # =====================================================================
        self._frequencies = subcarrier_frequencies(
            self.cfg.rg.fft_size, self.cfg.rg.subcarrier_spacing
        )

        # =====================================================================
        # Perfect CSI Extraction Utility
        # Removes guard bands and DC null from full channel response to get
        # coefficients only on data/pilot subcarriers for Rx processing.
        # =====================================================================
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.cfg.rg)

    def build(self, batch_size: int | tf.Tensor):
        """
        Generate frequency-domain channel response for a batch of samples.

        This method generates new channel impulse responses from the CDL model
        and converts them to frequency-domain coefficients. Each call produces
        an independent channel realization (different random path gains).

        Parameters
        ----------
        batch_size : int or tf.Tensor
            Number of independent channel realizations to generate.
            Can be a Python int or a scalar TensorFlow tensor.

        Returns
        -------
        h_freq : tf.Tensor, complex64
            Frequency-domain channel response with shape
            [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size].

            Each element ``h_freq[b,r,ra,t,ta,s,f]`` is the complex channel
            gain from TX antenna ``ta`` of transmitter ``t`` to RX antenna
            ``ra`` of receiver ``r``, on OFDM symbol ``s`` and subcarrier ``f``,
            for batch sample ``b``.

        Note
        ----
        - The CDL model internally uses Sionna's global RNG seeded in ``__init__``.
          For reproducible results across runs, ensure ``cfg.seed`` is fixed and
          no other code modifies ``sionna.phy.config.seed`` between calls.
        - Channel varies across OFDM symbols if ``cfg.speed > 0`` (Doppler).
        - For ``cfg.speed == 0``, channel is static within each batch sample
          but varies across batch samples.
        """
        # Handle both Python int and TensorFlow tensor batch_size inputs
        # This flexibility allows use in both eager and graph execution modes
        if isinstance(batch_size, tf.Tensor):
            bs = tf.cast(batch_size, tf.int32)
        else:
            bs = int(batch_size)

        # Generate channel impulse response (CIR) from CDL model
        # a: path coefficients [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # tau: path delays [batch, num_rx, num_tx, num_paths]
        a, tau = self._cdl(
            batch_size=bs,
            num_time_steps=self.cfg.rg.num_ofdm_symbols,
            sampling_frequency=1.0 / self.cfg.rg.ofdm_symbol_duration,
        )

        # Convert CIR to frequency-domain channel response
        # normalize=True ensures unit average energy for fair SNR comparison
        h_freq = cir_to_ofdm_channel(self._frequencies, a, tau, normalize=True)
        return h_freq
