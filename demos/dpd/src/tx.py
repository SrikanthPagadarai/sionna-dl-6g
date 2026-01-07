# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
OFDM transmitter for DPD system evaluation.

Implements a 5G NR-style OFDM transmitter chain that generates
baseband waveform suitable for DPD design and evaluation.

The transmit chain follows the standard order:
    random bits -> LDPC encoding -> QAM mapping -> resource grid -> OFDM modulation
"""

import os

# Suppress TF info/warning messages before importing TensorFlow.
# Must be set before TF import to take effect.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402
from sionna.phy.mapping import BinarySource  # noqa: E402
from sionna.phy import ofdm, mapping  # noqa: E402
from sionna.phy.fec.ldpc import LDPC5GEncoder  # noqa: E402
from sionna.phy.ofdm import OFDMModulator  # noqa: E402

from .config import Config  # noqa: E402


def _to_int(x):
    """
    Convert TensorFlow scalar to Python int.

    Sionna returns TensorFlow tensors for grid dimensions, but Python ints
    are needed for shape specifications and range calculations. This helper
    handles both graph-mode (static value) and eager-mode (numpy) cases.

    Parameters
    ----------
    x : tf.Tensor or int
        Scalar value to convert.

    Returns
    -------
    int
        Python integer value.
    """
    val = tf.get_static_value(x)
    if val is not None:
        return int(val)
    return int(x.numpy() if hasattr(x, "numpy") else x)


class Tx(tf.keras.Model):
    """
    5G NR-style OFDM transmitter for DPD design and evaluation.

    Implements a complete transmit chain from random bit generation through
    OFDM modulation. The output is a baseband time-domain signal ready for
    upsampling and PA transmission.

    Parameters
    ----------
    config : Config
        Configuration object containing all RF and OFDM parameters.

    Attributes
    ----------
    rg : sionna.ofdm.ResourceGrid
        OFDM resource grid defining time-frequency structure.
    n : int
        LDPC codeword length (coded bits per stream per slot).
    k : int
        LDPC information bits per stream per slot.
    encoder : LDPC5GEncoder
        5G NR LDPC encoder.
    mapper : Mapper
        QAM constellation mapper.
    rg_mapper : ResourceGridMapper
        Maps QAM symbols to resource grid locations.
    ofdm_modulator : OFDMModulator
        Converts frequency-domain grid to time-domain signal.

    Notes
    -----
    **Signal Flow:**

    1. ``BinarySource`` generates uniform random bits
    2. ``LDPC5GEncoder`` adds redundancy for error correction
    3. ``Mapper`` converts bit groups to complex QAM symbols
    4. ``ResourceGridMapper`` assigns symbols to subcarriers
    5. ``OFDMModulator`` performs IFFT and adds cyclic prefix

    **LDPC Code Design:**

    The code dimensions are derived from the resource grid capacity:

    - ``n`` = (number of data symbols) * (bits per symbol)
    - ``k`` = ``n`` * code_rate

    The 5G NR LDPC encoder automatically selects an appropriate base
    graph and lifting size for the given (k, n) pair.

    **miscellaneous:**

    - Config must define valid OFDM parameters
    - Code rate must yield 0 < k < n
    - Output ``x_time`` has shape ``[batch, 1, 1, num_time_samples]``
    - Output is complex64 baseband signal normalized for PA input

    Example
    -------
    >>> config = Config()
    >>> tx = Tx(config)
    >>> outputs = tx(batch_size=16)
    >>> outputs["x_time"].shape
    TensorShape([16, 1, 1, ...])
    """

    def __init__(self, config: Config):
        super().__init__()

        # Build resource grid from config parameters.
        # The grid defines the time-frequency structure of the OFDM signal.
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

        # Derive LDPC code dimensions from resource grid capacity.
        # n (codeword length) is constrained by available data symbols.
        # k (info bits) is set by desired code rate.
        n_data_syms = _to_int(self.rg.num_data_symbols)
        self.n = n_data_syms * config.num_bits_per_symbol
        self.k = int(round(self.n * config.coderate))

        if not (0 < self.k < self.n):
            raise ValueError(f"Invalid LDPC dims: k={self.k}, n={self.n}")

        # Instantiate TX chain components
        self._bit_src = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = mapping.Mapper("qam", config.num_bits_per_symbol)
        self.rg_mapper = ofdm.ResourceGridMapper(self.rg)
        self.ofdm_modulator = OFDMModulator(config.cyclic_prefix_length)

    def call(self, batch_size: tf.Tensor) -> dict:
        """
        Generate a batch of OFDM transmit waveforms.

        Generates random information bits and processes them through the
        complete transmit chain. All intermediate results are returned
        to support DPD training which needs access to original bits
        for BER evaluation.

        Parameters
        ----------
        batch_size : tf.Tensor or int
            Number of independent OFDM signal batches to generate.

        Returns
        -------
        dict
            Dictionary containing all transmit chain outputs:

            - ``bits`` : tf.Tensor, shape ``[B, num_tx, num_streams, k]``
                Original information bits (for BER calculation).
            - ``codewords`` : tf.Tensor, shape ``[B, num_tx, num_streams, n]``
                LDPC encoded bits.
            - ``x_rg`` : tf.Tensor, shape ``[B, num_tx, num_streams, num_ofdm_symbols, fft_size]``
                Frequency-domain resource grid (complex symbols).
            - ``x_time`` : tf.Tensor, shape ``[B, num_tx, num_streams, num_time_samples]``
                Time-domain baseband signal after OFDM modulation.

        Notes
        -----
        The returned ``bits`` tensor is essential for end-to-end BER
        measurement. After the signal passes through DPD and PA,
        decoded bits are compared against these original bits.
        """
        B = _to_int(batch_size)

        # Generate random information bits.
        bits = self._bit_src(
            [B, int(self.rg.num_tx), int(self.rg.num_streams_per_tx), self.k]
        )
        # Cast to encoder's expected dtype (handles mixed precision).
        bits = tf.cast(bits, self.encoder.rdtype)

        # TX processing chain.
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)
        x_time = self.ofdm_modulator(x_rg)

        return {"bits": bits, "codewords": codewords, "x_rg": x_rg, "x_time": x_time}
