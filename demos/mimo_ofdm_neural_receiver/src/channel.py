# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
OFDM channel application module for MIMO-OFDM neural receiver simulation.

This module provides a thin wrapper around Sionna's ``ApplyOFDMChannel`` that
applies frequency-domain channel effects and additive white Gaussian noise
(AWGN) to transmitted OFDM resource grids. The wrapper standardizes the
output format as a dictionary for consistent interfacing with other pipeline
components (Tx, Rx, NeuralRx).
"""

import tensorflow as tf
from typing import Dict, Any
from sionna.phy.channel import ApplyOFDMChannel


class Channel:
    """
    Frequency-domain OFDM channel with AWGN.

    Applies a pre-computed frequency-domain channel response to the transmitted
    resource grid and adds Gaussian noise. This class wraps Sionna's
    ``ApplyOFDMChannel`` to provide a consistent dictionary-based output
    interface matching the Tx and Rx components.

    The channel operation is:

    .. math::

        y[k] = H[k] \\cdot x[k] + n[k]

    where :math:`x[k]` is the transmitted signal on subcarrier :math:`k`,
    :math:`H[k]` is the frequency-domain channel coefficient, and
    :math:`n[k] \\sim \\mathcal{CN}(0, N_0)` is complex Gaussian noise.

    Note
    ----
    The channel frequency response ``h_freq`` must be pre-computed externally
    (typically by the ``CSI`` class) and passed to ``__call__``. This design
    ensures that the same channel realization is shared between the channel
    application and the receiver's perfect-CSI path.

    Example
    -------
    >>> channel = Channel()
    >>> y_out = channel(x_rg, h_freq, no)
    >>> received_signal = y_out["y"]
    """

    def __init__(self):
        """
        Initialize the OFDM channel applicator.

        Post-conditions
        ---------------
        - ``_apply`` is configured to add AWGN to the channel output.
          The noise power is controlled per-call via the ``no`` argument.
        """
        # AWGN is always enabled; noise power (no) controls SNR per call
        self._apply = ApplyOFDMChannel(add_awgn=True)

    @tf.function
    def __call__(
        self, x_rg_tx: tf.Tensor, h_freq: tf.Tensor, no: tf.Tensor
    ) -> Dict[str, Any]:
        """
        Apply frequency-domain channel and AWGN to transmitted resource grid.

        Parameters
        ----------
        x_rg_tx : tf.Tensor, complex, [batch, num_tx, num_streams, num_ofdm_symbols, fft_size]
            Transmitted OFDM resource grid containing modulated symbols
            and pilots mapped to time-frequency positions.

        h_freq : tf.Tensor, complex, [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
            Frequency-domain channel response. Each element represents the
            complex channel gain between a TX-RX antenna pair on a specific
            subcarrier and OFDM symbol.

        no : tf.Tensor, float, [batch] or scalar
            Noise power spectral density :math:`N_0`. Controls the AWGN
            variance added to the received signal. Typically computed from
            Eb/N0 using ``sionna.phy.utils.ebnodb2no``.

        Returns
        -------
        Dict[str, tf.Tensor]
            Dictionary containing:

            - ``"y"``: Received signal after channel and noise,
              shape [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].

        Pre-conditions
        --------------
        - ``x_rg_tx`` must have pilots placed according to the ResourceGrid
          configuration used during Tx.
        - ``h_freq`` dimensions must be compatible with the MIMO configuration
          (num_tx_ant, num_rx_ant) and resource grid (num_ofdm_symbols, fft_size).
        - ``no`` must be positive (zero noise power causes numerical issues
          in downstream LLR computation).

        Post-conditions
        ---------------
        - Output ``"y"`` has the same time-frequency dimensions as input
          but with receive antenna dimension instead of transmit streams.
        - The received signal includes contributions from all TX antennas
          (MIMO spatial mixing) plus independent noise per RX antenna.

        Note
        ----
        This method is decorated with ``@tf.function`` for graph-mode
        execution. The first call triggers tracing; subsequent calls with
        matching tensor shapes reuse the traced graph.
        """
        y = self._apply(x_rg_tx, h_freq, no)
        return {"y": y}
