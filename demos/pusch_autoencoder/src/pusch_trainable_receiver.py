# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Trainable PUSCH receiver with neural detection and training mode support.

Extends Sionna's standard PUSCHReceiver to support autoencoder training by:

1. Providing access to soft LLR outputs (bypassing TB decoding in training mode)
2. Synchronizing the demapper constellation with the trainable transmitter
3. Exposing neural detector trainable variables for optimizer access

The receiver acts as a thin wrapper that routes signals through the neural
detector while maintaining compatibility with Sionna's PUSCH processing chain.

Training vs Inference Mode
--------------------------
- **Training mode**: Returns LLRs directly after layer demapping, enabling
  BCE loss computation against coded bits. TB decoding is skipped.

- **Inference mode**: Full PUSCH reception including TB decoding. Returns
  decoded information bits for BER/BLER evaluation.
"""

import tensorflow as tf
from sionna.phy.channel import time_to_ofdm_channel
from sionna.phy.nr import PUSCHReceiver


class PUSCHTrainableReceiver(PUSCHReceiver):
    r"""
    PUSCH Receiver variant for autoencoder training with neural detection.

    This class extends ``PUSCHReceiver`` to support end-to-end training by:

    1. Returning soft LLRs before TB decoding when in training mode
    2. Passing the trainable constellation to the neural detector for
       consistent symbol demapping
    3. Exposing the neural detector's trainable variables

    The receiver supports both perfect CSI (ground-truth channel provided)
    and imperfect CSI (LS estimation with neural refinement) scenarios.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to ``PUSCHReceiver``.
    training : bool
        If ``True``, ``call()`` returns LLRs without TB decoding.
        If ``False``, ``call()`` returns decoded bits. Default ``False``.
    pusch_transmitter : PUSCHTrainableTransmitter, optional
        Reference to the transmitter for constellation synchronization.
        Required for proper demapping when constellation is trainable.
    **kwargs : dict
        Keyword arguments passed to ``PUSCHReceiver`` (e.g., ``mimo_detector``,
        ``input_domain``, ``channel_estimator``).

    Example
    -------
    >>> # Training setup
    >>> detector = PUSCHNeuralDetector(cfg)
    >>> rx = PUSCHTrainableReceiver(
    ...     mimo_detector=detector,
    ...     input_domain="freq",
    ...     pusch_transmitter=tx,
    ...     training=True
    ... )
    >>> llr = rx(y, no)  # Returns soft LLRs for BCE loss

    >>> # Inference setup
    >>> rx_eval = PUSCHTrainableReceiver(
    ...     mimo_detector=detector,
    ...     input_domain="freq",
    ...     pusch_transmitter=tx,
    ...     training=False
    ... )
    >>> b_hat = rx_eval(y, no)  # Returns decoded bits for BER

    Notes
    -----
    The same constellation is used for mapping (TX) and demapping (RX)
    for autoencoder training. The ``_get_normalized_constellation()``
    method retrieves the current (normalized) constellation from the
    transmitter at each forward pass, ensuring the demapper always
    uses the same points that were used for mapping, even as they
    evolve during training.
    """

    def __init__(self, *args, training=False, pusch_transmitter=None, **kwargs):
        self._training = training
        self._pusch_transmitter = pusch_transmitter

        # Parent constructor sets up channel estimator, layer demapper,
        # TB decoder, and other standard PUSCH receiver components.
        super().__init__(*args, pusch_transmitter=pusch_transmitter, **kwargs)

    @property
    def trainable_variables(self):
        """
        Collect trainable variables from the neural MIMO detector.

        Returns
        -------
        list of tf.Variable
            Trainable variables from ``self._mimo_detector``, or empty list
            if the detector has no trainable variables (e.g., classical LMMSE).

        Notes
        -----
        This property enables the optimizer to access detector weights without
        knowing the internal structure. The receiver itself has no trainable
        parameters; all learning happens in the neural detector.
        """
        if hasattr(self._mimo_detector, "trainable_variables"):
            return self._mimo_detector.trainable_variables
        return []

    def _get_normalized_constellation(self):
        """
        Retrieve the current normalized constellation from the transmitter.

        This method ensures the receiver's demapper uses the same constellation
        points as the transmitter's mapper, which is essential for correct
        gradient computation during autoencoder training.

        Returns
        -------
        tf.Tensor or None
            Complex tensor of shape ``[num_points]`` with normalized
            constellation points, or ``None`` if no transmitter is linked.

        Notes
        -----
        The constellation is retrieved fresh on each call because the
        transmitter's constellation variables may have been updated by
        the optimizer since the last forward pass.
        """
        if self._pusch_transmitter is not None:
            return self._pusch_transmitter.get_normalized_constellation()
        return None

    def call(self, y, no, h=None):
        """
        Execute receiver processing chain with optional training mode.

        Parameters
        ----------
        y : tf.Tensor, complex64
            Received signal. Shape depends on ``input_domain``:

            - ``"freq"``: ``[batch, num_rx, num_rx_ant, num_ofdm_symbols, num_subcarriers]``
            - ``"time"``: ``[batch, num_rx, num_rx_ant, num_samples]``
        no : tf.Tensor, float32
            Noise variance, shape ``[batch]`` or scalar.
        h : tf.Tensor, complex64, optional
            Ground-truth channel matrix for perfect CSI mode. Shape:
            ``[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]``
            Required if ``channel_estimator="perfect"`` was set in constructor.

        Returns
        -------
        tf.Tensor
            - **Training mode** (``training=True``): LLRs for coded bits,
              shape ``[batch, num_ue, num_coded_bits]``
            - **Inference mode** (``training=False``): Decoded information bits,
              shape ``[batch, num_ue, tb_size]``
            - If ``return_tb_crc_status=True`` in inference mode, returns
              tuple ``(b_hat, tb_crc_status)``

        Notes
        -----
        The processing chain follows standard PUSCH reception:

        1. **OFDM demodulation** (if time domain): FFT and CP removal
        2. **Channel estimation**: Perfect CSI passthrough or LS estimation
        3. **MIMO detection**: Neural detector with constellation sync
        4. **Layer demapping**: Separate streams back to UE data
        5. **TB decoding** (inference only): LDPC decoding, CRC check

        In training mode, step 5 is skipped because:

        - TB decoding is non-differentiable (hard decisions)
        - Need LLRs for BCE loss against coded bits ``c``
        - The loss provides gradients to train the neural detector

        The squeeze operation on LLRs (when shape has singleton dimension 2)
        handles the case where ``num_layers=1``, ensuring consistent output
        shape regardless of MIMO layer configuration.
        """
        # =====================================================================
        # OFDM Demodulation (if time-domain input)
        # =====================================================================
        if self._input_domain == "time":
            y = self._ofdm_demodulator(y)

        # =====================================================================
        # Channel Estimation
        # =====================================================================
        if self._perfect_csi:
            # Perfect CSI: use ground-truth channel (for upper-bound evaluation)
            if self._input_domain == "time":
                # Convert time-domain CIR to frequency-domain channel
                h = time_to_ofdm_channel(h, self.resource_grid, self._l_min)

            # Apply precoding matrix if configured (transforms TX antenna dim)
            if self._w is not None:
                h = tf.transpose(h, perm=[0, 1, 3, 5, 6, 2, 4])
                h = tf.matmul(h, self._w)
                h = tf.transpose(h, perm=[0, 1, 5, 2, 6, 3, 4])
            h_hat = h
            # Zero error variance for perfect CSI (no estimation noise)
            err_var = tf.zeros_like(tf.math.real(h_hat[:, :1, :1, :, :, :, :]))
        else:
            # Imperfect CSI: LS estimation from DMRS (realistic scenario)
            h_hat, err_var = self._channel_estimator(y, no)

        # =====================================================================
        # MIMO Detection with Constellation Synchronization
        # =====================================================================
        # Pass trainable constellation to ensure demapper uses same points as mapper.
        # This is critical for correct gradient computation in autoencoder training.
        constellation = self._get_normalized_constellation()
        if constellation is not None:
            llr = self._mimo_detector(
                y, h_hat, err_var, no, constellation=constellation
            )
        else:
            llr = self._mimo_detector(y, h_hat, err_var, no)

        # =====================================================================
        # Layer Demapping
        # =====================================================================
        # Reorganize LLRs from layer structure back to per-UE format
        llr = self._layer_demapper(llr)

        # =====================================================================
        # Training vs Inference Output
        # =====================================================================
        if self._training:
            # Training: return LLRs for BCE loss computation
            # Squeeze singleton layer dimension if present (num_layers=1 case)
            if len(llr.shape) == 4 and llr.shape[2] == 1:
                llr = tf.squeeze(llr, axis=2)
            return llr
        else:
            # Inference: full TB decoding for BER/BLER evaluation
            b_hat, tb_crc_status = self._tb_decoder(llr)
            if self._return_tb_crc_status:
                return b_hat, tb_crc_status
            else:
                return b_hat
