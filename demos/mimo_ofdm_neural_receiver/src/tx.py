# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Transmitter for MIMO-OFDM neural receiver demo.

Implements the complete transmit signal processing chain:

    Binary Source -> [LDPC Encoder] -> QAM Mapper -> Resource Grid Mapper

Supports two modes:

1. **Full encoding mode** (``channel_coding_off=False``): Information bits are
   LDPC-encoded before modulation. Used during inference to measure true BER/BLER.

2. **Training mode** (``channel_coding_off=True``): Random coded bits are
   generated directly (bypassing encoding). This allows the neural receiver to
   train on LLR prediction without backpropagating through the non-differentiable
   LDPC encoder.

The output includes both information bits (``b``) and coded bits (``c``) to
support both BER computation and BCE loss calculation during training.
"""

import os

# Suppress TensorFlow C++ logging before import
# Level 0 shows all messages; increase to 2+ for quieter operation
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import tensorflow as tf  # noqa: E402
from typing import Dict, Any  # noqa: E402
from sionna.phy.mapping import BinarySource, Mapper  # noqa: E402
from sionna.phy.fec.ldpc import LDPC5GEncoder  # noqa: E402
from sionna.phy.ofdm import ResourceGridMapper  # noqa: E402
from .config import Config  # noqa: E402


class Tx:
    """
    MIMO-OFDM Transmitter with optional LDPC encoding.

    Implements the transmit processing chain that generates OFDM resource grids
    from random information bits. The chain consists of:

    1. **Binary Source**: Generates random bits (information or coded).
    2. **LDPC Encoder** (optional): Applies 5G NR LDPC encoding.
    3. **QAM Mapper**: Maps bit sequences to constellation symbols.
    4. **Resource Grid Mapper**: Places symbols and pilots on OFDM grid.

    Parameters
    ----------
    cfg : Config
        Configuration object containing modulation, coding, and resource
        grid parameters.

    channel_coding_off : bool, (default False)
        If True, bypasses LDPC encoding and generates random coded bits
        directly. Used during training to avoid backpropagating through
        the non-differentiable encoder.

    Attributes
    ----------
    _cfg : Config
        Reference to configuration object.

    _channel_coding_off : bool
        Whether encoding is bypassed.

    _num_streams_per_tx : int
        Number of spatial streams (equals number of UT antennas).

    Note
    ----
    In training mode, the neural receiver learns to predict LLRs for random
    bit patterns. The BCE loss compares predicted LLRs against the known
    transmitted coded bits ``c``, enabling gradient-based optimization.

    Example
    -------
    >>> cfg = Config(num_bits_per_symbol=BitsPerSym.QPSK)
    >>> tx = Tx(cfg, channel_coding_off=False)
    >>> out = tx(batch_size=32, h_freq=h_freq)
    >>> print(out["b"].shape)  # Information bits
    >>> print(out["x_rg"].shape)  # Transmitted resource grid
    """

    def __init__(self, cfg: Config, channel_coding_off: bool = False):
        """
        Initialize transmitter components.

        Parameters
        ----------
        cfg : Config
            Configuration specifying modulation order, code rate, and
            resource grid structure.

        channel_coding_off : bool, (default False)
            If True, skip LDPC encoding (training mode).
            If False, apply full encoding (inference mode).

        Post-conditions
        ---------------
        - ``_binary_source`` is ready to generate random bits.
        - ``_encoder`` is configured with code dimensions (k, n) from config.
        - ``_mapper`` uses QAM constellation with configured bits per symbol.
        - ``_rg_mapper`` is bound to the resource grid from config.
        """
        self._cfg = cfg
        self._channel_coding_off = channel_coding_off

        # Binary source generates uniform random bits {0, 1}
        self._binary_source = BinarySource()

        # 5G NR LDPC encoder: k information bits -> n coded bits
        self._encoder = LDPC5GEncoder(self._cfg.k, self._cfg.n)

        # QAM mapper: groups of num_bits_per_symbol bits -> complex symbols
        self._mapper = Mapper(self._cfg.modulation, self._cfg.num_bits_per_symbol)

        # Resource grid mapper: places data symbols and pilots on OFDM grid
        self._rg_mapper = ResourceGridMapper(self._cfg.rg)

        # Cache stream count for output tensor shaping
        self._num_streams_per_tx = self._cfg.num_streams_per_tx

    @tf.function
    def __call__(self, batch_size: tf.Tensor) -> Dict[str, Any]:
        """
        Generate transmitted OFDM resource grid for a batch.

        Parameters
        ----------
        batch_size : tf.Tensor, int32, scalar
            Number of independent transmissions to generate.

        Returns
        -------
        Dict[str, tf.Tensor]
            Dictionary containing:

            - ``"b"``: Information bits before encoding, shape
              [batch, 1, num_streams, k]. None if ``channel_coding_off=True``.

            - ``"c"``: Coded bits after encoding (or raw bits if encoding off),
              shape [batch, 1, num_streams, n].

            - ``"x"``: QAM symbols after mapping, shape
              [batch, 1, num_streams, n/num_bits_per_symbol].

            - ``"x_rg"``: Complete OFDM resource grid with pilots, shape
              [batch, num_tx, num_streams, num_ofdm_symbols, fft_size].

        """
        # =====================================================================
        # Bit Generation and Encoding
        # Two paths: training (random coded bits) vs inference (encode info bits)
        # =====================================================================
        b = None
        if self._channel_coding_off:
            # Training mode: generate random coded bits directly
            # This avoids backprop through the non-differentiable LDPC encoder
            c = self._binary_source(
                [batch_size, 1, self._num_streams_per_tx, self._cfg.n]
            )
        else:
            # Inference mode: generate info bits, then encode
            # b is needed for BER computation against decoded bits
            b = self._binary_source(
                [batch_size, 1, self._num_streams_per_tx, self._cfg.k]
            )
            c = self._encoder(b)

        # =====================================================================
        # Modulation and Resource Grid Mapping
        # =====================================================================
        # Map coded bits to QAM constellation points
        x = self._mapper(c)

        # Place symbols on resource grid (inserts pilots automatically)
        x_rg = self._rg_mapper(x)

        return {"b": b, "c": c, "x": x, "x_rg": x_rg}
