"""Central configuration for the DPD system."""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Config:
    """
    Central configuration for the DPD system.

    All parameters are fixed and immutable except for seed and batch_size.
    """

    # Mutable parameters (can be set at initialization)
    seed: int = field(default=42)
    batch_size: int = field(default=100)

    # System parameters (immutable)
    _num_ut: int = field(init=False, default=1, repr=False)
    _num_ut_ant: int = field(init=False, default=1, repr=False)
    _num_streams_per_tx: int = field(init=False, default=1, repr=False)

    # Resource grid parameters (immutable)
    _num_ofdm_symbols: int = field(init=False, default=8, repr=False)
    _fft_size: int = field(init=False, default=1024, repr=False)
    _subcarrier_spacing: float = field(init=False, default=15000.0, repr=False)
    _num_guard_carriers: Tuple[int, int] = field(init=False, repr=False)
    _dc_null: bool = field(init=False, default=True, repr=False)
    _cyclic_prefix_length: int = field(init=False, default=72, repr=False)
    _pilot_pattern: str = field(init=False, default="kronecker", repr=False)
    _pilot_ofdm_symbol_indices: List[int] = field(init=False, repr=False)

    # Modulation and coding parameters (immutable)
    _num_bits_per_symbol: int = field(init=False, default=4, repr=False)
    _coderate: float = field(init=False, default=0.5, repr=False)

    def __post_init__(self):
        self._num_guard_carriers = (200, 199)
        self._pilot_ofdm_symbol_indices = [2, 6]

    # Mutable properties (with setters) - no underscore prefix
    # Note: These don't need explicit properties since they're public fields

    # System properties (immutable - getters only)
    @property
    def num_ut(self) -> int:
        return self._num_ut

    @property
    def num_ut_ant(self) -> int:
        return self._num_ut_ant

    @property
    def num_streams_per_tx(self) -> int:
        return self._num_streams_per_tx

    # Resource grid properties (immutable - getters only)
    @property
    def num_ofdm_symbols(self) -> int:
        return self._num_ofdm_symbols

    @property
    def fft_size(self) -> int:
        return self._fft_size

    @property
    def subcarrier_spacing(self) -> float:
        return self._subcarrier_spacing

    @property
    def num_guard_carriers(self) -> Tuple[int, int]:
        return self._num_guard_carriers

    @property
    def dc_null(self) -> bool:
        return self._dc_null

    @property
    def cyclic_prefix_length(self) -> int:
        return self._cyclic_prefix_length

    @property
    def pilot_pattern(self) -> str:
        return self._pilot_pattern

    @property
    def pilot_ofdm_symbol_indices(self) -> List[int]:
        return self._pilot_ofdm_symbol_indices

    # Modulation and coding properties (immutable - getters only)
    @property
    def num_bits_per_symbol(self) -> int:
        return self._num_bits_per_symbol

    @property
    def coderate(self) -> float:
        return self._coderate

    # Derived properties
    @property
    def signal_sample_rate(self) -> float:
        """Signal sample rate in Hz (fft_size * subcarrier_spacing)."""
        return self._fft_size * self._subcarrier_spacing
