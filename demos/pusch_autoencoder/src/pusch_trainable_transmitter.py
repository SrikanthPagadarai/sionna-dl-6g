# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Trainable PUSCH transmitter with learnable symbol constellation.

Extends Sionna's standard PUSCHTransmitter to support end-to-end
optimization of constellation points. The key is maintaining valid
constellation properties (unit average power, centered) while
allowing gradient flow through the constellation geometry.

Design Approach
---------------
Rather than training Sionna's internal Constellation object directly (which
applies normalization in ways that can complicate gradient flow), explicit
``tf.Variable`` tensors are maintaind for the real and imaginary parts of each
constellation point. Normalization is applied explicitly in ``call()`` before
mapping, ensuring consistent behavior while preserving gradients.

The transmitter can operate in two modes:

- **Training mode** (``training=True``): Constellation variables are trainable,
  and ``call()`` returns intermediate tensors needed for loss computation.

- **Inference mode** (``training=False``): Constellation is fixed, and the
  transmitter behaves like a standard ``PUSCHTransmitter``.
"""

from sionna.phy.nr import PUSCHTransmitter
from sionna.phy.mapping import Mapper, Constellation
import tensorflow as tf


class PUSCHTrainableTransmitter(PUSCHTransmitter):
    r"""
    PUSCH Transmitter with trainable constellation points.

    This subclass of ``PUSCHTransmitter`` supports learnable constellation
    geometry for autoencoder-based communication system design. The
    constellation points are stored as explicit ``tf.Variable`` tensors,
    enabling gradient-based optimization while maintaining unit average power.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to ``PUSCHTransmitter`` (typically a list
        of ``PUSCHConfig`` objects).
    training : bool
        If ``True``, constellation variables are trainable and ``call()``
        returns additional tensors for loss computation. Default ``False``.
    **kwargs : dict
        Keyword arguments passed to ``PUSCHTransmitter`` (e.g., ``output_domain``).

    Example
    -------
    >>> pusch_configs = [PUSCHConfig(), ...]  # Configure for each UE
    >>> tx = PUSCHTrainableTransmitter(pusch_configs, output_domain="freq",
    ...                                 training=True)
    >>> x_map, x, b, c = tx(batch_size=32)
    >>> # x_map: mapped symbols, x: OFDM signal, b: info bits, c: coded bits

    Notes
    -----
    The constellation normalization follows Sionna's ``Constellation.call()``
    behavior:

    1. **Centering**: Subtract mean to ensure zero-mean constellation
    2. **Power normalization**: Divide by sqrt(mean energy) for unit power

    This is applied every forward pass to ensure the power constraint is
    satisfied regardless of how the underlying variables have moved during
    optimization. The normalization is differentiable, so gradients flow
    back to the raw ``_points_r`` and ``_points_i`` variables.

    The choice to use separate real/imaginary variables (rather than a single
    complex variable) is intentional: TensorFlow optimizers handle real-valued
    gradients more naturally, and it avoids potential issues with complex
    gradient computation in some TF versions.
    """

    def __init__(self, *args, training=False, **kwargs):
        self._training = training

        # Parent constructor sets up standard PUSCH processing chain
        # including TB encoder, layer mapper, resource grid mapper, etc.
        super().__init__(*args, **kwargs)

        # Replace standard constellation with trainable version
        self._setup_custom_constellation()

    @property
    def trainable_variables(self):
        """
        Return the trainable constellation coordinate variables.

        Returns
        -------
        list of tf.Variable
            Two-element list: ``[_points_r, _points_i]`` representing the
            real and imaginary parts of constellation points.

        Notes
        -----
        These variables are trainable only if ``training=True`` was passed
        to the constructor. The ordering (real first, then imaginary) is
        consistent across calls.
        """
        return [self._points_r, self._points_i]

    # [get-normalized-constellation-start]
    def get_normalized_constellation(self):
        """
        Compute centered and power-normalized constellation points.

        This method applies the same normalization as Sionna's
        ``Constellation.call()`` to ensure the constellation has zero mean
        and unit average power, regardless of the current variable values.

        Returns
        -------
        tf.Tensor, complex64
            Normalized constellation points, shape ``[num_points]``.
            For 16-QAM, this is ``[16]`` complex values.

        Notes
        -----
        The normalization is differentiable and applied as:

        1. ``points = points - mean(points)``  (centering)
        2. ``points = points / sqrt(mean(|points|^2))``  (power normalization)

        This ensures:

        - Sum of constellation points equals zero (balanced I/Q)
        - Average symbol energy equals 1.0 (consistent SNR interpretation)

        The normalization prevents the constellation from drifting to
        trivially "good" solutions (e.g., all points at origin, or very
        large points that artificially boost SNR).
        """
        points = tf.complex(self._points_r, self._points_i)

        # Center: remove DC offset for balanced constellation
        points = points - tf.reduce_mean(points)

        # Normalize: scale to unit average power for consistent SNR
        energy = tf.reduce_mean(tf.square(tf.abs(points)))
        points = points / tf.cast(tf.sqrt(energy), points.dtype)

        return points

    # [get-normalized-constellation-end]

    def _setup_custom_constellation(self):
        """
        Initialize trainable constellation from standard QAM geometry.

        This method creates ``tf.Variable`` tensors for constellation
        coordinates, initializing them from Sionna's standard QAM points.
        The standard QAM provides a good starting point that the optimizer
        can refine for the specific channel conditions.

        The method also replaces the internal mapper to use the custom
        constellation with manual normalization control.

        Notes
        -----
        Both ``normalize`` and ``center`` are disabled in the ``Constellation``
        constructor because these operations are explicitly applied in
        ``get_normalized_constellation()``. This gives full control over
        when and how normalization occurs, which is important for:

        1. Ensuring gradients flow correctly through normalization
        2. Providing access to both raw and normalized points
        3. Matching the exact behavior of Sionna's standard constellation
        """
        # Initialize from standard QAM - a well-designed starting point
        # that Gray-coded bit mapping optimizes for AWGN channels
        qam_points = Constellation(
            "qam", num_bits_per_symbol=self._num_bits_per_symbol
        ).points

        # Store as separate real/imag variables for optimizer compatibility
        # (some TF optimizers handle real gradients better than complex)
        init_r = tf.math.real(qam_points)
        init_i = tf.math.imag(qam_points)

        self._points_r = tf.Variable(
            tf.cast(init_r, self.rdtype),
            trainable=self._training,
            name="constellation_real",
        )
        self._points_i = tf.Variable(
            tf.cast(init_i, self.rdtype),
            trainable=self._training,
            name="constellation_imag",
        )

        # Create constellation object with manual normalization
        # normalize=False, center=False because these are handled explicitly
        self._constellation = Constellation(
            "custom",
            num_bits_per_symbol=self._num_bits_per_symbol,
            points=tf.complex(self._points_r, self._points_i),
            normalize=False,
            center=False,
        )

        # Replace the mapper to use trainable constellation
        self._mapper = Mapper(constellation=self._constellation)

    def call(self, inputs):
        """
        Execute transmitter processing chain with trainable constellation.

        Parameters
        ----------
        inputs : int or tf.Tensor
            - If ``return_bits=True`` (default): ``int`` specifying batch size.
              Random bits will be generated internally.
            - If ``return_bits=False``: ``tf.Tensor`` of shape
              ``[batch_size, num_tx, tb_size]`` containing input bits.

        Returns
        -------
        tuple or tf.Tensor
            - If ``return_bits=True``: tuple ``(x_map, x, b, c)`` where:
              - ``x_map``: Mapped symbols before layer mapping, shape
                ``[batch_size, num_tx, num_symbols]``
              - ``x``: Transmitted signal (freq or time domain), shape
                ``[batch_size, num_tx, num_ant, num_ofdm_symbols, num_subcarriers]``
                or ``[batch_size, num_tx, num_ant, num_samples]``
              - ``b``: Information bits, shape ``[batch_size, num_tx, tb_size]``
              - ``c``: Coded bits, shape ``[batch_size, num_tx, num_coded_bits]``
            - If ``return_bits=False``: just ``x`` (transmitted signal)

        Notes
        -----
        The processing chain follows standard PUSCH transmission:

        1. **Bit generation** (if return_bits=True): Random binary source
        2. **TB encoding**: CRC attachment, LDPC encoding, rate matching
        3. **Symbol mapping**: Bits to complex symbols using trainable constellation
        4. **Layer mapping**: Symbols to MIMO layers
        5. **Resource grid mapping**: Layers to OFDM resource elements
        6. **Precoding** (if codebook): Apply TPMI-selected precoder
        7. **OFDM modulation** (if time domain): IFFT and CP insertion

        The constellation is centered and normalized at the start of each call to
        ensure unit power regardless of variable updates from previous training steps.
        """
        # Apply normalization before mapping to ensure unit power constraint.
        # This must happen every call because optimizer may have updated
        # the raw _points_r/_points_i variables since last call.
        self._constellation.points = self.get_normalized_constellation()

        if self._return_bits:
            # Generate random bits internally
            batch_size = inputs
            b = self._binary_source([batch_size, self._num_tx, self._tb_size])
        else:
            # Use provided bits
            b = inputs

        # TB encoding: CRC, segmentation, LDPC encoding, rate matching
        c = self._tb_encoder(b)

        # Map coded bits to constellation symbols
        x_map = self._mapper(c)

        # Distribute symbols across MIMO layers
        x_layer = self._layer_mapper(x_map)

        # Place symbols on OFDM resource grid with DMRS
        x_grid = self._resource_grid_mapper(x_layer)

        # Apply precoding if configured (codebook-based for this demo)
        if self._precoding == "codebook":
            x_pre = self._precoder(x_grid)
        else:
            x_pre = x_grid

        # Convert to time domain if requested (freq domain for this demo)
        if self._output_domain == "time":
            x = self._ofdm_modulator(x_pre)
        else:
            x = x_pre

        if self._return_bits:
            # Return intermediates needed for training loss computation
            # x_map: for constellation visualization
            # b: for BER computation
            # c: for BCE loss against LLRs
            return x_map, x, b, c
        else:
            return x
