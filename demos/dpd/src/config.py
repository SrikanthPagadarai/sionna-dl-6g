# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Central configuration for the DPD demo.

Provides a configuration dataclass that ensures consistent
parameters across all DPD system components. The design uses immutable defaults
for RF/OFDM parameters to guarantee reproducible experiments, while allowing
batch_size and seed customization for training flexibility.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Config:
    """
    Central configuration for the DPD system.

    This dataclass enforces a separation between user-configurable parameters
    (seed, batch_size) and system constants (RF, OFDM, modulation settings).
    Immutable parameters use underscore-prefixed private fields with read-only
    properties to prevent accidental modification during experiments.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. Defaults to 42.
    batch_size : int
        Number of OFDM frames per batch. Defaults to 100.

    Attributes
    ----------
    num_ut : int
        Number of user terminals (fixed at 1 for single-user DPD).
    num_ut_ant : int
        Number of antennas per user terminal (fixed at 1).
    num_streams_per_tx : int
        Number of spatial streams per transmitter (fixed at 1).
    num_ofdm_symbols : int
        OFDM symbols per frame (fixed at 8).
    fft_size : int
        FFT size for OFDM modulation (fixed at 1024).
    subcarrier_spacing : float
        Subcarrier spacing in Hz (fixed at 15 kHz, LTE/5G compatible).
    num_guard_carriers : tuple of int
        Guard carriers (lower, upper) to reduce spectral leakage.
    dc_null : bool
        Whether DC subcarrier is nulled (True for practical systems).
    cyclic_prefix_length : int
        Cyclic prefix samples (fixed at 72, ~7% overhead).
    pilot_pattern : str
        Pilot arrangement strategy (fixed at "kronecker").
    pilot_ofdm_symbol_indices : list of int
        OFDM symbol indices containing pilots.
    num_bits_per_symbol : int
        Bits per QAM symbol (fixed at 4 for 16-QAM).
    coderate : float
        LDPC code rate (fixed at 0.5).
    signal_sample_rate : float
        Derived baseband sample rate in Hz.

    Notes
    -----
    **Design Rationale:**

    - Immutable RF parameters prevent mid-experiment changes that would
      invalidate DPD coefficient learning.
    - Guard carrier settings (200, 199) are asymmetric to account for
      DC null, yielding 624 usable subcarriers.

    **miscellaneous:**

    - signal_sample_rate = fft_size * subcarrier_spacing
    - Total subcarriers = fft_size - num_guard_carriers[0]
                        - num_guard_carriers[1] - dc_null

    Example
    -------
    >>> config = Config(seed=123, batch_size=32)
    >>> config.signal_sample_rate
    15360000.0
    >>> config.fft_size  # Immutable
    1024
    """

    # --- Mutable parameters (user-configurable at initialization) ---
    seed: int = field(default=42)
    batch_size: int = field(default=100)

    # --- System parameters (immutable) ---
    # Single-user SISO configuration
    _num_ut: int = field(init=False, default=1, repr=False)
    _num_ut_ant: int = field(init=False, default=1, repr=False)
    _num_streams_per_tx: int = field(init=False, default=1, repr=False)

    # --- Resource grid parameters (immutable) ---
    _num_ofdm_symbols: int = field(init=False, default=8, repr=False)
    _fft_size: int = field(init=False, default=1024, repr=False)
    _subcarrier_spacing: float = field(init=False, default=15000.0, repr=False)
    _num_guard_carriers: Tuple[int, int] = field(init=False, repr=False)
    _dc_null: bool = field(init=False, default=True, repr=False)
    _cyclic_prefix_length: int = field(init=False, default=72, repr=False)
    _pilot_pattern: str = field(init=False, default="kronecker", repr=False)
    _pilot_ofdm_symbol_indices: List[int] = field(init=False, repr=False)

    # --- Modulation and coding parameters (immutable) ---
    _num_bits_per_symbol: int = field(init=False, default=4, repr=False)
    _coderate: float = field(init=False, default=0.5, repr=False)

    def __post_init__(self):
        # Asymmetric guard bands: 200 lower + 199 upper + 1 DC = 400 nulled
        # This leaves 624 active subcarriers for data and pilots
        self._num_guard_carriers = (200, 199)
        self._pilot_ofdm_symbol_indices = [2, 6]

    # =========================================================================
    # System properties (read-only access to immutable parameters)
    # =========================================================================

    @property
    def num_ut(self) -> int:
        """int: Number of user terminals."""
        return self._num_ut

    @property
    def num_ut_ant(self) -> int:
        """int: Number of antennas per user terminal."""
        return self._num_ut_ant

    @property
    def num_streams_per_tx(self) -> int:
        """int: Number of spatial streams per transmitter."""
        return self._num_streams_per_tx

    # =========================================================================
    # Resource grid properties (read-only)
    # =========================================================================

    @property
    def num_ofdm_symbols(self) -> int:
        """int: Number of OFDM symbols per frame."""
        return self._num_ofdm_symbols

    @property
    def fft_size(self) -> int:
        """int: FFT size for OFDM modulation."""
        return self._fft_size

    @property
    def subcarrier_spacing(self) -> float:
        """float: Subcarrier spacing in Hz."""
        return self._subcarrier_spacing

    @property
    def num_guard_carriers(self) -> Tuple[int, int]:
        """tuple of int: Number of (lower, upper) guard carriers."""
        return self._num_guard_carriers

    @property
    def dc_null(self) -> bool:
        """bool: Whether the DC subcarrier is nulled."""
        return self._dc_null

    @property
    def cyclic_prefix_length(self) -> int:
        """int: Cyclic prefix length in samples."""
        return self._cyclic_prefix_length

    @property
    def pilot_pattern(self) -> str:
        """str: Pilot pattern type for channel estimation."""
        return self._pilot_pattern

    @property
    def pilot_ofdm_symbol_indices(self) -> List[int]:
        """list of int: OFDM symbol indices containing pilot symbols."""
        return self._pilot_ofdm_symbol_indices

    # =========================================================================
    # Modulation and coding properties (read-only)
    # =========================================================================

    @property
    def num_bits_per_symbol(self) -> int:
        """int: Number of bits per modulation symbol (4 = 16-QAM)."""
        return self._num_bits_per_symbol

    @property
    def coderate(self) -> float:
        """float: Forward error correction code rate."""
        return self._coderate

    # =========================================================================
    # Derived properties
    # =========================================================================

    @property
    def signal_sample_rate(self) -> float:
        """
        Baseband signal sample rate in Hz.

        Returns
        -------
        float
            Sample rate computed as ``fft_size * subcarrier_spacing``.
            With defaults: 1024 * 15000 = 15.36 MHz.

        Notes
        -----
        This is the native OFDM sample rate before any upsampling to the
        PA sample rate. The interpolator handles rate conversion between
        this rate and the PA's operating rate of 122.88 MHz.
        """
        return self._fft_size * self._subcarrier_spacing
