import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import sionna.phy  # noqa: E402
import tensorflow as tf  # noqa: E402
from sionna.phy.channel.tr38901 import CDL, AntennaArray  # noqa: E402
from sionna.phy.ofdm import RemoveNulledSubcarriers  # noqa: E402
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel  # noqa: E402
from .config import Config  # noqa: E402


class CSI:
    """
    Shared context for a *single simulation iteration*.

    On construction:
      - Builds CDL, subcarrier frequencies
      - Generates CIR for the provided batch_size and rg.num_ofdm_symbols
      - Converts to frequency-domain channel h_freq (cached here)

    Exposes:
      .cfg, .rg, .h_freq, .remove_nulled_scs
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Determinism if desired
        sionna.phy.config.seed = int(self.cfg._seed)

        # Antenna arrays
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

        # CDL channel
        self._cdl = CDL(
            model=self.cfg.cdl_model,
            delay_spread=self.cfg.delay_spread,
            carrier_frequency=self.cfg.carrier_frequency,
            ut_array=self._ut_array,
            bs_array=self._bs_array,
            direction=self.cfg.direction,
            min_speed=self.cfg.speed,
        )

        # Subcarrier frequencies for mapping CIR -> H(f)
        self._frequencies = subcarrier_frequencies(
            self.cfg.rg.fft_size, self.cfg.rg.subcarrier_spacing
        )

        # For perfect UL-CSI
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.cfg.rg)

    # Build h_freq for this simulation iteration
    def build(self, batch_size: int | tf.Tensor):
        if isinstance(batch_size, tf.Tensor):
            bs = tf.cast(batch_size, tf.int32)
        else:
            bs = int(batch_size)

        a, tau = self._cdl(
            batch_size=bs,
            num_time_steps=self.cfg.rg.num_ofdm_symbols,
            sampling_frequency=1.0 / self.cfg.rg.ofdm_symbol_duration,
        )
        h_freq = cir_to_ofdm_channel(self._frequencies, a, tau, normalize=True)
        return h_freq
