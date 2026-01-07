# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Central configuration for the MIMO-OFDM neural receiver demo.

Provides a centralized configuration dataclass that holds all
PHY-layer and system parameters for the MIMO-OFDM demo. The design
separates user-tunable parameters (e.g., channel model, CSI mode) from
hard-coded system constants (e.g., FFT size, antenna counts) to
enforce a single, validated system configuration.
"""

from dataclasses import dataclass, field
from typing import ClassVar, FrozenSet, Literal, Tuple
from enum import IntEnum
import numpy as np
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid

CDLModel = Literal["A", "B", "C", "D", "E"]


class BitsPerSym(IntEnum):
    """
    Enumeration of supported modulation orders.

    Maps modulation scheme names to their bits-per-symbol values.
    The integer value represents log2 of the constellation size.
    """

    BPSK = 1  # 2^1 = 2-QAM
    QPSK = 2  # 2^2 = 4-QAM
    QAM16 = 4  # 2^4 = 16-QAM


@dataclass(slots=True)
class Config:
    """
    Global configuration container for a MIMO-OFDM simulation setup.

    This dataclass centralizes all simulation parameters and derived objects
    (ResourceGrid, StreamManagement, LDPC code lengths) to ensure consistency
    across Tx, Channel, and Rx components. Parameters are divided into two
    categories:

    1. **User-settable**: Can be modified per experiment (e.g., ``cdl_model``,
       ``perfect_csi``). These control the simulation scenario.

    2. **Hard-coded (immutable)**: PHY/system constants validated for this
       demo (e.g., ``fft_size``, ``num_bs_ant``). Attempting to modify these
       after initialization raises ``AttributeError``.

    Parameters
    ----------
    perfect_csi : bool, (default False)
        If True, the receiver uses ground-truth channel state information
        instead of LS-estimated CSI. Useful for establishing performance
        upper bounds.

    cdl_model : {"A", "B", "C", "D", "E"}, (default "D")
        3GPP CDL channel model variant. Models A-C are NLOS with increasing
        delay spread; D-E are LOS. Model D provides moderate multipath
        suitable for neural receiver training.

    delay_spread : float, (default 300e-9)
        RMS delay spread in seconds. Controls the temporal dispersion of
        the channel. Typical urban values: 100-500 ns.

    carrier_frequency : float, (default 2.6e9)
        Carrier frequency in Hz. Affects Doppler spread and path loss
        characteristics in the CDL model.

    speed : float, (default 0.0)
        UE speed in m/s. Zero indicates a static channel
        (no Doppler). Non-zero enables time-varying fading.

    num_bits_per_symbol : BitsPerSym, (default BitsPerSym.QPSK)
        Modulation order. Accepts ``BitsPerSym`` enum or equivalent int.
        Higher orders increase spectral efficiency but require better SNR.

    Attributes
    ----------
    rg : ResourceGrid
        Configured OFDM resource grid with pilot pattern. Created by
        ``build()`` during initialization.

    sm : StreamManagement
        MIMO stream management object defining TX-RX stream associations.

    k : int
        Number of information bits per codeword (before LDPC encoding).

    n : int
        Number of coded bits per codeword (after LDPC encoding).
        Satisfies: ``n = k / coderate``.

    Note
    ----
    - The ``build()`` method is called automatically in ``__post_init__``.
      After initialization, immutable fields are locked and cannot be changed.
    - ``n == rg.num_data_symbols * num_bits_per_symbol``
    - ``k == n * coderate``
    - ``num_streams_per_tx == num_ut_ant`` (one stream per UT antenna)

    Example
    -------
    >>> cfg = Config(cdl_model="C", perfect_csi=True)
    >>> print(cfg.rg.num_data_symbols)
    >>> print(cfg.k, cfg.n)  # LDPC code dimensions
    """

    # =========================================================================
    # User-settable parameters
    # These may be adjusted per experiment to explore different scenarios.
    # =========================================================================
    perfect_csi: bool = False
    cdl_model: CDLModel = "D"
    delay_spread: float = 300e-9  # seconds
    carrier_frequency: float = 2.6e9  # Hz
    speed: float = 0.0  # m/s (UE speed)
    num_bits_per_symbol: BitsPerSym = BitsPerSym.QPSK

    # [phy-parameters-start]
    # =========================================================================
    # Hard-coded PHY/system parameters
    # These define the physical layer configuration and should not be modified
    # without re-validating the entire simulation chain.
    # =========================================================================

    # Uplink direction: UTs transmit to BS (affects antenna role assignment)
    _direction: str = field(init=False, default="uplink", repr=False)

    # OFDM numerology: 15 kHz SCS is standard for sub-6 GHz 5G NR
    _subcarrier_spacing: float = field(init=False, default=15e3, repr=False)

    # FFT size chosen to balance frequency resolution vs. complexity
    _fft_size: int = field(init=False, default=76, repr=False)

    # 14 OFDM symbols per slot (normal cyclic prefix, 5G NR standard)
    _num_ofdm_symbols: int = field(init=False, default=14, repr=False)

    # CP length must exceed max channel delay spread to prevent ISI
    _cyclic_prefix_length: int = field(init=False, default=6, repr=False)

    # Guard carriers prevent aliasing at band edges (asymmetric for DC null)
    _num_guard_carriers: Tuple[int, int] = field(init=False, default=(5, 6), repr=False)

    # DC subcarrier nulled to avoid LO leakage issues in practical systems
    _dc_null: bool = field(init=False, default=True, repr=False)

    # Kronecker pilot pattern: orthogonal in time-frequency, good for MIMO
    _pilot_pattern: str = field(init=False, default="kronecker", repr=False)

    # Pilot positions distributed across slot for channel tracking
    _pilot_ofdm_symbol_indices: Tuple[int, ...] = field(
        init=False, default=(2, 5, 8, 11), repr=False
    )

    # 4x8 MIMO: 4 UT antennas (Tx), 8 BS antennas (Rx) for uplink
    # This asymmetry favors the BS with more receive diversity
    _num_ut_ant: int = field(init=False, default=4, repr=False)
    _num_bs_ant: int = field(init=False, default=8, repr=False)

    # QAM modulation (constellation type, not order)
    _modulation: str = field(init=False, default="qam", repr=False)

    # Default bits per symbol (overridden by user-settable num_bits_per_symbol)
    _num_bits_per_symbol: int = field(init=False, default=2, repr=False)  # QPSK

    # Rate-1/2 LDPC provides good balance of coding gain vs. complexity
    _coderate: float = field(init=False, default=0.5, repr=False)

    # Fixed seed for reproducible channel/noise realizations
    _seed: int = field(init=False, default=42, repr=False)
    # [phy-parameters-end]

    # =========================================================================
    # Derived system parameters (computed by build())
    # =========================================================================
    _sm: StreamManagement = field(init=False, repr=False)
    _rg: ResourceGrid = field(init=False, repr=False)
    _k: int = field(init=False, repr=False)
    _n: int = field(init=False, repr=False)
    _num_streams_per_tx: int = field(init=False, repr=False)

    # =========================================================================
    # Immutability enforcement
    # Prevents accidental modification of validated PHY parameters after init.
    # =========================================================================
    _IMMUTABLE_FIELDS: ClassVar[FrozenSet[str]] = frozenset(
        {
            "_direction",
            "_subcarrier_spacing",
            "_fft_size",
            "_num_ofdm_symbols",
            "_cyclic_prefix_length",
            "_num_guard_carriers",
            "_dc_null",
            "_pilot_pattern",
            "_pilot_ofdm_symbol_indices",
            "_num_ut_ant",
            "_num_bs_ant",
            "_modulation",
            "_num_bits_per_symbol",
            "_coderate",
            "_seed",
        }
    )
    _immutable_locked: bool = field(init=False, default=False, repr=False)

    def __setattr__(self, name, value):
        """
        Intercept attribute assignment to enforce immutability constraints.

        Pre-conditions
        --------------
        - If ``_immutable_locked`` is False, all attributes can be set.

        Post-conditions
        ---------------
        - After ``_immutable_locked`` is True, fields in ``_IMMUTABLE_FIELDS``
          cannot be modified; attempting to do so raises ``AttributeError``.

        Raises
        ------
        AttributeError
            If attempting to modify an immutable field after initialization.
        """
        if getattr(self, "_immutable_locked", False) and name in self._IMMUTABLE_FIELDS:
            raise AttributeError(
                f"{name} is immutable (hard-coded PHY/system parameter)."
            )
        object.__setattr__(self, name, value)

    def build(self) -> "Config":
        """
        Construct derived objects (ResourceGrid, StreamManagement, LDPC lengths).

        This method computes all dependent configuration objects from the
        base parameters. It is called automatically during ``__post_init__``
        and should not typically be called directly.

        Pre-conditions
        --------------
        - All base parameters (``_fft_size``, ``_num_ofdm_symbols``, etc.)
          must be set to valid values.

        Post-conditions
        ---------------
        - ``_rg`` contains a fully configured ResourceGrid.
        - ``_sm`` contains StreamManagement for single-user MIMO.
        - ``_n`` and ``_k`` satisfy the coderate relationship.
        - ``_num_streams_per_tx == _num_ut_ant`` (spatial multiplexing).

        Returns
        -------
        Config
            Self reference for method chaining.

        Note
        ----
        The StreamManagement matrix ``[[1]]`` indicates a single TX-RX pair.
        Each UT antenna carries an independent data stream.
        """
        # Spatial multiplexing: one stream per UT antenna
        self._num_streams_per_tx = self._num_ut_ant

        # Stream matrix: single TX-RX link (no multi-user scheduling)
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)

        self._rg = ResourceGrid(
            num_ofdm_symbols=self._num_ofdm_symbols,
            fft_size=self._fft_size,
            subcarrier_spacing=self._subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=self._num_streams_per_tx,
            cyclic_prefix_length=self._cyclic_prefix_length,
            num_guard_carriers=list(self._num_guard_carriers),
            dc_null=self._dc_null,
            pilot_pattern=self._pilot_pattern,
            pilot_ofdm_symbol_indices=list(self._pilot_ofdm_symbol_indices),
        )

        # LDPC code dimensions derived from resource grid capacity
        # n = total bits that fit in data symbols; k = information bits
        self._n = int(self._rg.num_data_symbols * self.num_bits_per_symbol)
        self._k = int(self._n * self._coderate)
        return self

    def __post_init__(self):
        """
        Finalize configuration after dataclass field initialization.

        Performs type coercion for ``num_bits_per_symbol``, builds derived
        objects, and locks immutable fields to prevent later modification.

        Post-conditions
        ---------------
        - ``num_bits_per_symbol`` is guaranteed to be a ``BitsPerSym`` enum.
        - All derived objects (``_rg``, ``_sm``, ``_k``, ``_n``) are initialized.
        - ``_immutable_locked`` is True; immutable fields cannot be changed.
        """
        # Allow int input for num_bits_per_symbol, coerce to enum
        if not isinstance(self.num_bits_per_symbol, BitsPerSym):
            self.num_bits_per_symbol = BitsPerSym(self.num_bits_per_symbol)
        self.build()
        self._immutable_locked = True

    # =========================================================================
    # Read-only property accessors
    # Expose internal fields through properties to maintain encapsulation
    # while signaling that these values should not be modified.
    # =========================================================================

    @property
    def rg(self) -> ResourceGrid:
        """ResourceGrid: Configured OFDM resource grid with pilot pattern."""
        return self._rg

    @property
    def sm(self) -> StreamManagement:
        """StreamManagement: MIMO stream-to-TX/RX mapping configuration."""
        return self._sm

    @property
    def k(self) -> int:
        """int: Number of information bits per LDPC codeword."""
        return self._k

    @property
    def n(self) -> int:
        """int: Number of coded bits per LDPC codeword."""
        return self._n

    @property
    def num_streams_per_tx(self) -> int:
        """int: Number of spatial streams per transmitter (equals num_ut_ant)."""
        return self._num_streams_per_tx

    @property
    def direction(self) -> str:
        """str: Link direction, either 'uplink' or 'downlink'."""
        return self._direction

    @property
    def subcarrier_spacing(self) -> float:
        """float: OFDM subcarrier spacing in Hz."""
        return self._subcarrier_spacing

    @property
    def fft_size(self) -> int:
        """int: FFT size determining the number of subcarriers."""
        return self._fft_size

    @property
    def num_ofdm_symbols(self) -> int:
        """int: Number of OFDM symbols per slot/frame."""
        return self._num_ofdm_symbols

    @property
    def cyclic_prefix_length(self) -> int:
        """int: Cyclic prefix length in samples."""
        return self._cyclic_prefix_length

    @property
    def num_guard_carriers(self) -> Tuple[int, int]:
        """Tuple[int, int]: Number of guard subcarriers (lower, upper)."""
        return self._num_guard_carriers

    @property
    def dc_null(self) -> bool:
        """bool: Whether the DC subcarrier is nulled."""
        return self._dc_null

    @property
    def pilot_pattern(self) -> str:
        """str: Pilot pattern type (e.g., 'kronecker')."""
        return self._pilot_pattern

    @property
    def pilot_ofdm_symbol_indices(self) -> Tuple[int, ...]:
        """Tuple[int, ...]: OFDM symbol indices containing pilots."""
        return self._pilot_ofdm_symbol_indices

    @property
    def num_ut_ant(self) -> int:
        """int: Number of user terminal antennas (transmit side in uplink)."""
        return self._num_ut_ant

    @property
    def num_bs_ant(self) -> int:
        """int: Number of base station antennas (receive side in uplink)."""
        return self._num_bs_ant

    @property
    def modulation(self) -> str:
        """str: Modulation type (e.g., 'qam')."""
        return self._modulation

    @property
    def coderate(self) -> float:
        """float: LDPC code rate (k/n ratio)."""
        return self._coderate

    @property
    def seed(self) -> int:
        """int: Random seed for reproducible simulations."""
        return self._seed
