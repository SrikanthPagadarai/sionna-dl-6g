# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
End-to-end MIMO-OFDM system.

Provides the ``System`` class which orchestrates the complete
simulation pipeline by composing Tx, Channel, and Rx components:

    Tx -> Channel -> Rx (baseline or neural)

The system supports two operational modes:

1. **Training mode** (``training=True``):
   - Channel coding disabled in Tx (random coded bits)
   - Neural receiver outputs raw LLRs (no LDPC decoding)
   - Returns BCE loss for gradient-based optimization

2. **Inference mode** (``training=False``):
   - Full LDPC encoding in Tx
   - Complete decoding in Rx (baseline or neural)
   - Returns (transmitted_bits, decoded_bits) for BER/BLER computation

The system is implemented as a Keras Model, enabling:
- Automatic variable tracking for optimization
- Integration with TensorFlow training loops
- Checkpointing and weight serialization
"""

import tensorflow as tf
from sionna.phy.utils import ebnodb2no
from tensorflow.keras import Model

from .config import Config, BitsPerSym, CDLModel
from .csi import CSI
from .tx import Tx
from .channel import Channel
from .rx import Rx
from .neural_rx import NeuralRx


class System(Model):
    """
    End-to-end MIMO-OFDM system with baseline and neural receiver options.

    This Keras Model composes all simulation components and provides a
    unified interface for both training and inference. The system generates
    transmitted signals, applies channel effects, and processes received
    signals through either a conventional or neural receiver.

    The processing pipeline is:

    1. **CSI Generation**: Create frequency-domain channel response
    2. **Transmission**: Generate bits, encode, modulate, map to OFDM grid
    3. **Channel**: Apply frequency-domain channel and add AWGN
    4. **Reception**: Process received signal (baseline LMMSE or neural CNN)
    5. **Output**: Return loss (training) or bit tensors (inference)

    Parameters
    ----------
    training : bool, (default False)
        If True, configure for training mode:
        - Disable channel coding in Tx/Rx
        - Return BCE loss instead of bit tensors

    perfect_csi : bool, (default False)
        If True, baseline receiver uses ground-truth CSI.
        Only affects baseline Rx; neural Rx never uses explicit CSI.

    cdl_model : {"A", "B", "C", "D", "E"}, (default "D")
        3GPP CDL channel model variant.

    delay_spread : float, (default 300e-9)
        RMS delay spread in seconds.

    carrier_frequency : float, (default 2.6e9)
        Carrier frequency in Hz.

    speed : float, (default 0.0)
        UE speed in m/s for Doppler modeling.

    num_bits_per_symbol : BitsPerSym, (default BitsPerSym.QPSK)
        Modulation order.

    use_neural_rx : bool, (default False)
        If True, use neural receiver; otherwise use baseline LMMSE receiver.

    num_conv2d_filters : int, (default 128)
        Neural receiver CNN width.

    num_resnet_layers : int, (default 2)
        Layers per residual block in neural receiver.

    num_res_blocks : int, (default 4)
        Number of residual blocks in neural receiver.

    name : str, (default "system")
        Keras model name for variable scoping.

    Attributes
    ----------
    bce : tf.keras.losses.BinaryCrossentropy
        Loss function for training (expects logits, not probabilities).

    Note
    ----
    The system accepts Eb/N0 in dB and internally converts to noise power
    using ``ebnodb2no``. This allows consistent SNR specification across
    different modulation orders and code rates.

    Both ``__call__`` and ``call_scalar`` are provided:
    - ``__call__``: Takes vector Eb/N0 (one per batch sample)
    - ``call_scalar``: Takes scalar Eb/N0 (broadcast to all samples)

    The scalar variant is required for compatibility with Sionna's
    ``PlotBER.simulate()`` which passes scalar SNR values.

    Example
    -------
    >>> # Training
    >>> system = System(training=True, use_neural_rx=True)
    >>> loss = system(batch_size, ebno_db_vector)
    >>> # Inference
    >>> system = System(training=False, use_neural_rx=True)
    >>> b, b_hat = system(batch_size, ebno_db_vector)
    """

    def __init__(
        self,
        *,
        training: bool = False,
        perfect_csi: bool = False,
        cdl_model: CDLModel = "D",
        delay_spread: float = 300e-9,
        carrier_frequency: float = 2.6e9,
        speed: float = 0.0,
        num_bits_per_symbol: BitsPerSym = BitsPerSym.QPSK,
        use_neural_rx: bool = False,
        num_conv2d_filters: int = 256,
        num_resnet_layers: int = 2,
        num_res_blocks: int = 4,
        name: str = "system",
    ):
        """
        Initialize end-to-end system with all components.

        Parameters
        ----------
        training : bool
            Enable training mode (affects Tx encoding and Rx decoding).

        perfect_csi : bool
            Enable perfect CSI for baseline receiver.

        cdl_model : str
            CDL channel model variant.

        delay_spread : float
            Channel delay spread in seconds.

        carrier_frequency : float
            Carrier frequency in Hz.

        speed : float
            UE speed for Doppler.

        num_bits_per_symbol : BitsPerSym
            Modulation order.

        use_neural_rx : bool
            Select neural vs baseline receiver.

        num_conv2d_filters : int
            Neural receiver width.

        num_resnet_layers : int
            Neural receiver block depth.

        num_res_blocks : int
            Neural receiver depth.

        name : str
            Keras model name.

        """
        super().__init__(name=name)

        self._training = training

        # Neural receiver architecture parameters
        self._use_neural_rx = bool(use_neural_rx)
        self._num_conv2d_filters = num_conv2d_filters
        self._num_resnet_layers = num_resnet_layers
        self._num_res_blocks = num_res_blocks

        # Build configuration with PHY parameters
        self._cfg = Config(
            perfect_csi=perfect_csi,
            cdl_model=cdl_model,
            delay_spread=delay_spread,
            carrier_frequency=carrier_frequency,
            speed=speed,
            num_bits_per_symbol=num_bits_per_symbol,
        )

        # =====================================================================
        # Component Instantiation
        # All components share the same Config and CSI for consistency
        # =====================================================================
        self._csi = CSI(self._cfg)

        # Tx: channel coding disabled during training to allow gradient flow
        self._tx = Tx(self._cfg, self._training)

        self._ch = Channel()

        # Baseline Rx: conventional LMMSE receiver (not used during training)
        self._rx = Rx(self._cfg, self._csi)

        # Neural Rx: CNN-based receiver with configurable architecture
        # Channel decoding disabled during training (BCE loss on raw LLRs)
        self._neural_rx = NeuralRx(
            self._cfg,
            self._training,
            self._num_conv2d_filters,
            self._num_resnet_layers,
            self._num_res_blocks,
        )

        # BCE loss for training: from_logits=True because neural Rx outputs LLRs
        # LLRs are log-odds ratios, equivalent to logits for binary classification
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=[], dtype=tf.int32),  # scalar batch size
            tf.TensorSpec(shape=[], dtype=tf.float32),  # scalar ebno_db
        ],
    )
    def call_scalar(self, batch_size, ebno_db_scalar):
        """
        Forward pass with scalar Eb/N0 (for PlotBER compatibility).

        This method broadcasts a single Eb/N0 value to all batch samples,
        providing compatibility with Sionna's ``PlotBER.simulate()`` which
        calls the model with scalar SNR values.

        Parameters
        ----------
        batch_size : tf.Tensor, int32, scalar
            Number of samples in the batch.

        ebno_db_scalar : tf.Tensor, float32, scalar
            Eb/N0 in dB, applied uniformly to all batch samples.

        Returns
        -------
        See ``__call__`` for return value documentation.

        Note
        ----
        This is a thin wrapper that expands the scalar to a vector
        and delegates to ``__call__``.
        """
        # Broadcast scalar Eb/N0 to vector of length batch_size
        ebno_vec = tf.fill([batch_size], ebno_db_scalar)
        return self.__call__(batch_size, ebno_vec)

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=[], dtype=tf.int32),  # batch_size (scalar)
            tf.TensorSpec(
                shape=[None], dtype=tf.float32
            ),  # ebno_db vector (len == batch_size)
        ],
    )
    def __call__(self, batch_size: tf.Tensor, ebno_db: tf.Tensor):
        """
        Forward pass through complete MIMO-OFDM system.

        Parameters
        ----------
        batch_size : tf.Tensor, int32, scalar
            Number of independent channel realizations to simulate.

        ebno_db : tf.Tensor, float32, [batch_size]
            Per-sample Eb/N0 in dB. Allows training across SNR range
            by varying Eb/N0 within each batch.

        Returns
        -------
        Training mode (``training=True`` and ``use_neural_rx=True``):
            loss : tf.Tensor, float32, scalar
                Binary cross-entropy loss between transmitted coded bits
                and predicted LLRs. Suitable for gradient descent.

        Inference mode (``training=False``):
            b : tf.Tensor, float32, [batch, 1, num_streams, k]
                Transmitted information bits (ground truth).

            b_hat : tf.Tensor, float32, [batch, 1, num_streams, k]
                Decoded information bits (receiver output).

        Pre-conditions
        --------------
        - ``len(ebno_db) == batch_size`` (vector length must match batch).
        - ``ebno_db`` values should be in reasonable range (e.g., -5 to 20 dB).

        Post-conditions
        ---------------
        - Training: Loss is scalar, suitable for optimizer.apply_gradients().
        - Inference: b and b_hat have matching shapes for BER computation.

        Note
        ----
        The Eb/N0 to noise power conversion accounts for:
        - Modulation order (bits per symbol)
        - Code rate
        - Resource grid structure (data vs pilot symbols)

        This ensures consistent SNR interpretation across configurations.

        Invariants
        ----------
        - Same channel realization (h_freq) used for transmission and reception.
        - Random bits are independent across batch samples.
        """
        # =====================================================================
        # Channel Generation
        # Generate new channel realization for this batch
        # =====================================================================
        h_freq = self._csi.build(batch_size)

        # Convert Eb/N0 (dB) to noise power spectral density
        # Accounts for modulation order, code rate, and resource grid efficiency
        no = ebnodb2no(
            ebno_db, self._cfg.num_bits_per_symbol, self._cfg.coderate, self._cfg.rg
        )

        # =====================================================================
        # Transmission and Channel
        # =====================================================================
        tx_out = self._tx(batch_size)
        y_out = self._ch(tx_out["x_rg"], h_freq, no)

        # =====================================================================
        # Reception
        # Select receiver and prepare arguments based on configuration
        # =====================================================================
        rx_to_use = self._neural_rx if self._use_neural_rx else self._rx

        # Neural Rx takes (y, no, batch_size); baseline Rx takes (y, h_freq, no)
        # This difference reflects that neural Rx doesn't use explicit CSI
        rx_args_to_pass = (
            (y_out["y"], no, batch_size)
            if self._use_neural_rx
            else (y_out["y"], h_freq, no)
        )
        rx_out = rx_to_use(*rx_args_to_pass)

        # =====================================================================
        # Output Selection
        # Training returns loss; inference returns bit tensors
        # =====================================================================
        if self._use_neural_rx and self._training:
            # BCE loss: compare predicted LLRs against transmitted coded bits
            # tx_out["c"] contains the ground-truth coded bits
            # rx_out["llr"] contains the predicted log-likelihood ratios
            loss = self.bce(tx_out["c"], rx_out["llr"])
            return loss

        # Inference: return transmitted and decoded information bits
        return tx_out["b"], rx_out["b_hat"]
