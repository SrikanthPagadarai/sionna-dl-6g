import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402
from sionna.phy.mapping import BinarySource  # noqa: E402
from sionna.phy import ofdm, mapping  # noqa: E402
from sionna.phy.fec.ldpc import LDPC5GEncoder  # noqa: E402
from sionna.phy.ofdm import OFDMModulator  # noqa: E402

from .config import Config  # noqa: E402


def _to_int(x):
    """Convert TensorFlow scalar to Python int."""
    val = tf.get_static_value(x)
    if val is not None:
        return int(val)
    return int(x.numpy() if hasattr(x, "numpy") else x)


class Tx(tf.keras.Model):
    """Minimal OFDM Transmitter: encoding, mapping, RG mapping."""

    def __init__(self, config: Config):
        super().__init__()

        self.rg = ofdm.ResourceGrid(
            num_ofdm_symbols=config.num_ofdm_symbols,
            fft_size=config.fft_size,
            subcarrier_spacing=config.subcarrier_spacing,
            num_tx=config.num_ut,
            num_streams_per_tx=config.num_streams_per_tx,
            num_guard_carriers=config.num_guard_carriers,
            dc_null=config.dc_null,
            cyclic_prefix_length=config.cyclic_prefix_length,
            pilot_pattern=config.pilot_pattern,
            pilot_ofdm_symbol_indices=config.pilot_ofdm_symbol_indices,
        )

        # LDPC code dimensions
        n_data_syms = _to_int(self.rg.num_data_symbols)
        self.n = n_data_syms * config.num_bits_per_symbol  # coded bits
        self.k = int(round(self.n * config.coderate))  # info bits

        if not (0 < self.k < self.n):
            raise ValueError(f"Invalid LDPC dims: k={self.k}, n={self.n}")

        # TX chain blocks
        self._bit_src = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = mapping.Mapper("qam", config.num_bits_per_symbol)
        self.rg_mapper = ofdm.ResourceGridMapper(self.rg)
        self.ofdm_modulator = OFDMModulator(config.cyclic_prefix_length)

    def call(self, batch_size: tf.Tensor) -> dict:
        """Run transmitter end-to-end, generating random bits internally."""
        B = _to_int(batch_size)

        bits = self._bit_src(
            [B, int(self.rg.num_tx), int(self.rg.num_streams_per_tx), self.k]
        )
        bits = tf.cast(bits, self.encoder.rdtype)

        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)
        x_time = self.ofdm_modulator(x_rg)

        return {"bits": bits, "codewords": codewords, "x_rg": x_rg, "x_time": x_time}
