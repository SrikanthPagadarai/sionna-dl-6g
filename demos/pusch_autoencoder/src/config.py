from dataclasses import dataclass, field
from typing import Tuple
from typing import List
import tensorflow as tf
from sionna.phy.nr.utils import MCSDecoderNR


@dataclass
class Config:
    """
    Central configuration for the PUSCH RT demo.
    """

    # Hard-coded system parameters
    _subcarrier_spacing: float = field(init=False, default=30e3, repr=False)  # Hz
    _num_time_steps: int = field(
        init=False, default=14, repr=False
    )  # OFDM symbols per slot

    _num_ue: int = field(init=False, default=4, repr=False)  # users
    _num_bs: int = field(init=False, default=1, repr=False)  # base-stations
    _num_ue_ant: int = field(init=False, default=4, repr=False)  # UE antennas
    _num_bs_ant: int = field(init=False, default=16, repr=False)  # BS antennas

    _batch_size_cir: int = field(
        init=False, default=500, repr=False
    )  # batch for CIR generation
    _target_num_cirs: int = field(
        init=False, default=5000, repr=False
    )  # total CIRs to generate

    _resource_grid: object = field(init=False, default=None, repr=False)
    _pusch_pilot_indices: List[int] = field(init=False, repr=False)
    _pusch_num_subcarriers: int = field(init=False, default=1, repr=False)
    _pusch_num_symbols_per_slot: int = field(init=False, default=1, repr=False)

    # Path solver / radio map
    _max_depth: int = field(init=False, default=5, repr=False)  # max reflections
    _min_gain_db: float = field(
        init=False, default=-130.0, repr=False
    )  # ignore below this path gain
    _max_gain_db: float = field(
        init=False, default=0.0, repr=False
    )  # ignore above this path gain
    _min_dist_m: float = field(
        init=False, default=5.0, repr=False
    )  # sampling annulus: inner radius
    _max_dist_m: float = field(
        init=False, default=400.0, repr=False
    )  # sampling annulus: outer radius

    # Radio map rendering
    _rm_cell_size: Tuple[float, float] = field(
        init=False, default=(1.0, 1.0), repr=False
    )
    _rm_samples_per_tx: int = field(init=False, default=10**7, repr=False)
    _rm_vmin_db: float = field(init=False, default=-110.0, repr=False)
    _rm_clip_at: float = field(init=False, default=12.0, repr=False)
    _rm_resolution: Tuple[int, int] = field(init=False, default=(650, 500), repr=False)
    _rm_num_samples: int = field(init=False, default=4096, repr=False)

    # BER/BLER simulation
    _batch_size: int = field(
        init=False, default=32, repr=False
    )  # must match CIRDataset batch size

    # Internal seed
    _seed: int = field(init=False, default=42, repr=False)

    # PUSCH-specific parameters
    _num_prb: int = field(
        init=False, default=16, repr=False
    )  # Number of physical resource blocks
    _mcs_index: int = field(
        init=False, default=14, repr=False
    )  # Modulation and coding scheme index
    _num_layers: int = field(init=False, default=1, repr=False)  # Number of MIMO layers
    _mcs_table: int = field(init=False, default=1, repr=False)  # MCS table selection
    _domain: str = field(init=False, default="freq", repr=False)  # Processing domain
    _num_bits_per_symbol: int = field(init=False, repr=False)
    _target_coderate: float = field(init=False, repr=False)

    def __post_init__(self):
        mcs_decoder = MCSDecoderNR()

        mcs_index = tf.constant(self._mcs_index, dtype=tf.int32)
        mcs_table_index = tf.constant(self._mcs_table, dtype=tf.int32)
        mcs_category = tf.constant(0, dtype=tf.int32)

        modulation_order, target_coderate = mcs_decoder(
            mcs_index,
            mcs_table_index,
            mcs_category,
            check_index_validity=True,
            transform_precoding=True,
            pi2bpsk=False,
        )

        # Convert to Python scalars
        self._num_bits_per_symbol = int(modulation_order.numpy())
        self._target_coderate = float(target_coderate.numpy())

        self._pusch_pilot_indices = [0, 0]

    # get-methods
    @property
    def subcarrier_spacing(self) -> float:
        return self._subcarrier_spacing

    @property
    def num_time_steps(self) -> int:
        return self._num_time_steps

    @property
    def num_ue(self) -> int:
        return self._num_ue

    @property
    def num_bs(self) -> int:
        return self._num_bs

    @property
    def num_ue_ant(self) -> int:
        return self._num_ue_ant

    @property
    def num_bs_ant(self) -> int:
        return self._num_bs_ant

    @property
    def batch_size_cir(self) -> int:
        return self._batch_size_cir

    @property
    def target_num_cirs(self) -> int:
        return self._target_num_cirs

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def min_gain_db(self) -> float:
        return self._min_gain_db

    @property
    def max_gain_db(self) -> float:
        return self._max_gain_db

    @property
    def min_dist_m(self) -> float:
        return self._min_dist_m

    @property
    def max_dist_m(self) -> float:
        return self._max_dist_m

    @property
    def rm_cell_size(self) -> Tuple[float, float]:
        return self._rm_cell_size

    @property
    def rm_samples_per_tx(self) -> int:
        return self._rm_samples_per_tx

    @property
    def rm_vmin_db(self) -> float:
        return self._rm_vmin_db

    @property
    def rm_clip_at(self) -> float:
        return self._rm_clip_at

    @property
    def rm_resolution(self) -> Tuple[int, int]:
        return self._rm_resolution

    @property
    def rm_num_samples(self) -> int:
        return self._rm_num_samples

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def num_prb(self) -> int:
        return self._num_prb

    @property
    def mcs_index(self) -> int:
        return self._mcs_index

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def mcs_table(self) -> int:
        return self._mcs_table

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def num_bits_per_symbol(self) -> float:
        return self._num_bits_per_symbol

    @property
    def target_coderate(self) -> float:
        return self._target_coderate

    @property
    def resource_grid(self):
        return self._resource_grid

    @property
    def pusch_pilot_indices(self):
        return self._pusch_pilot_indices

    @property
    def pusch_num_subcarriers(self):
        return self._pusch_num_subcarriers

    @property
    def pusch_num_symbols_per_slot(self):
        return self._pusch_num_symbols_per_slot

    # set methods
    @resource_grid.setter
    def resource_grid(self, rg):
        self._resource_grid = rg

    @pusch_pilot_indices.setter
    def pusch_pilot_indices(self, pusch_pilot_indices):
        self._pusch_pilot_indices = pusch_pilot_indices

    @pusch_num_subcarriers.setter
    def pusch_num_subcarriers(self, pusch_num_subcarriers):
        self._pusch_num_subcarriers = pusch_num_subcarriers

    @pusch_num_symbols_per_slot.setter
    def pusch_num_symbols_per_slot(self, pusch_num_symbols_per_slot):
        self._pusch_num_symbols_per_slot = pusch_num_symbols_per_slot
