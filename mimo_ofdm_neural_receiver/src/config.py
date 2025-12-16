from dataclasses import dataclass, field
from typing import ClassVar, FrozenSet, Literal, Tuple
from enum import IntEnum
import numpy as np
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid

CDLModel = Literal["A", "B", "C", "D", "E"]

class BitsPerSym(IntEnum):
    BPSK  = 1   # 2^1 = 2-QAM
    QPSK  = 2   # 2^2 = 4-QAM
    QAM16 = 4   # 2^4 = 16-QAM

@dataclass(slots=True)
class Config:
    """
    Global configuration for one simulation setup.

    User-settable properties:
      - perfect_csi, cdl_model, delay_spread, carrier_frequency, speed, num_bits_per_symbol

    All other properties are hard-coded and exposed via read-only properties.
    On build(), creates:
      - ResourceGrid (rg)
      - StreamManagement (sm)
      - LDPC lengths (k, n)
    """

    # user-settable parameters
    perfect_csi: bool = False
    cdl_model: CDLModel = "D"
    delay_spread: float = 300e-9 # seconds
    carrier_frequency: float = 2.6e9 # Hz
    speed: float = 0.0 # m/s (UE speed)
    num_bits_per_symbol: BitsPerSym = BitsPerSym.QPSK

    # hard-coded PHY/system parameters
    _direction: str = field(init=False, default="uplink", repr=False)
    _subcarrier_spacing: float = field(init=False, default=15e3, repr=False)
    _fft_size: int = field(init=False, default=76, repr=False)
    _num_ofdm_symbols: int = field(init=False, default=14, repr=False)
    _cyclic_prefix_length: int = field(init=False, default=6, repr=False)
    _num_guard_carriers: Tuple[int, int] = field(init=False, default=(5, 6), repr=False)
    _dc_null: bool = field(init=False, default=True, repr=False)
    _pilot_pattern: str = field(init=False, default="kronecker", repr=False)
    _pilot_ofdm_symbol_indices: Tuple[int, ...] = field(init=False, default=(2, 5, 8, 11), repr=False)
    _num_ut_ant: int = field(init=False, default=4, repr=False)
    _num_bs_ant: int = field(init=False, default=8, repr=False)
    _modulation: str = field(init=False, default="qam", repr=False)
    _num_bits_per_symbol: int = field(init=False, default=2, repr=False)  # QPSK
    _coderate: float = field(init=False, default=0.5, repr=False)
    _seed: int = field(init=False, default=42, repr=False)

    # derived system parameters
    _sm: StreamManagement = field(init=False, repr=False)
    _rg: ResourceGrid = field(init=False, repr=False)
    _k: int = field(init=False, repr=False)
    _n: int = field(init=False, repr=False)
    _num_streams_per_tx: int = field(init=False, repr=False)

    # enforce immutability
    _IMMUTABLE_FIELDS: ClassVar[FrozenSet[str]] = frozenset({
        "_direction", "_subcarrier_spacing", "_fft_size", "_num_ofdm_symbols", "_cyclic_prefix_length",
        "_num_guard_carriers", "_dc_null", "_pilot_pattern", "_pilot_ofdm_symbol_indices",
        "_num_ut_ant", "_num_bs_ant", "_modulation", "_num_bits_per_symbol", "_coderate", "_seed",
    })
    _immutable_locked: bool = field(init=False, default=False, repr=False)

    def __setattr__(self, name, value):
        if getattr(self, "_immutable_locked", False) and name in self._IMMUTABLE_FIELDS:
            raise AttributeError(f"{name} is immutable (hard-coded PHY/system parameter).")
        super().__setattr__(name, value)
    
    # build/cache objects used across modules
    def build(self) -> "Config":
        # map one stream per UT antenna
        self._num_streams_per_tx = self._num_ut_ant
        
        # Stream matrix: one TX, one RX stream group
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

        # code lengths derived from RG payload size
        self._n = int(self._rg.num_data_symbols * self.num_bits_per_symbol)
        self._k = int(self._n * self._coderate)
        return self

    def __post_init__(self):
        if not isinstance(self.num_bits_per_symbol, BitsPerSym):
            self.num_bits_per_symbol = BitsPerSym(self.num_bits_per_symbol)
        self.build()
        self._immutable_locked = True

    # get-methods
    @property
    def rg(self) -> ResourceGrid:
        return self._rg

    @property
    def sm(self) -> StreamManagement:
        return self._sm

    @property
    def k(self) -> int:
        return self._k

    @property
    def n(self) -> int:
        return self._n

    @property
    def num_streams_per_tx(self) -> int:
        return self._num_streams_per_tx

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def subcarrier_spacing(self) -> float:
        return self._subcarrier_spacing

    @property
    def fft_size(self) -> int:
        return self._fft_size

    @property
    def num_ofdm_symbols(self) -> int:
        return self._num_ofdm_symbols

    @property
    def cyclic_prefix_length(self) -> int:
        return self._cyclic_prefix_length

    @property
    def num_guard_carriers(self) -> Tuple[int, int]:
        return self._num_guard_carriers

    @property
    def dc_null(self) -> bool:
        return self._dc_null

    @property
    def pilot_pattern(self) -> str:
        return self._pilot_pattern

    @property
    def pilot_ofdm_symbol_indices(self) -> Tuple[int, ...]:
        return self._pilot_ofdm_symbol_indices

    @property
    def num_ut_ant(self) -> int:
        return self._num_ut_ant

    @property
    def num_bs_ant(self) -> int:
        return self._num_bs_ant

    @property
    def modulation(self) -> str:
        return self._modulation

    @property
    def coderate(self) -> float:
        return self._coderate

    @property
    def seed(self) -> int:
        return self._seed
