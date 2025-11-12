from dataclasses import dataclass, field
from typing import Tuple
from enum import IntEnum
import numpy as np

class AntennaPattern(IntEnum):
    ISO = 0
    TR38901 = 1

@dataclass
class Config:
    """
    Central configuration for the PUSCH RT demo.

    Style intentionally mirrors your reference config.py:
      - dataclass container
      - user-tweakable fields grouped up top
      - read-only properties to access values elsewhere

    NOTE: No functionality changes vs your current script.
    """

    # -----------------------
    # User-settable parameters
    # -----------------------
    subcarrier_spacing: float = 30e3          # Hz
    num_time_steps: int = 14                  # OFDM symbols per slot

    num_ue: int = 4                           # users
    num_bs: int = 1                           # base-stations
    num_ue_ant: int = 4                       # UE antennas (cross-pol planar array uses half columns)
    num_bs_ant: int = 16                      # BS antennas (cross-pol planar array uses half columns)

    batch_size_cir: int = 100                 # batch for CIR generation
    target_num_cirs: int = 200                # total CIRs to generate

    # Path solver / radio map
    max_depth: int = 5                        # max reflections
    min_gain_db: float = -130.0               # ignore below this path gain
    max_gain_db: float = 0.0                  # ignore above this path gain
    min_dist_m: float = 5.0                   # sampling annulus: inner radius
    max_dist_m: float = 400.0                 # sampling annulus: outer radius

    # Radio map rendering (purely for visualization; values unchanged)
    rm_cell_size: Tuple[float, float] = (1.0, 1.0)
    rm_samples_per_tx: int = 10000
    rm_vmin_db: float = -110.0
    rm_clip_at: float = 12.0
    rm_resolution: Tuple[int, int] = (650, 500)
    rm_num_samples: int = 4096

    # BER/BLER simulation
    batch_size: int = 20                      # must match CIRDataset batch size
    plot_title: str = "PUSCH RT – BLER vs Eb/N0 (LMMSE)"

    # Internal seed (kept for parity with ref style)
    _seed: int = field(init=False, default=42, repr=False)

    # ---------------
    # Read-only props
    # ---------------
    @property
    def seed(self) -> int:
        return self._seed

    # Convenience aliases (to mirror reference style of accessing via properties)
    @property
    def SUBCARRIER_SPACING(self) -> float:
        return self.subcarrier_spacing

    @property
    def NUM_TIME_STEPS(self) -> int:
        return self.num_time_steps

    @property
    def NUM_UE(self) -> int:
        return self.num_ue

    @property
    def NUM_BS(self) -> int:
        return self.num_bs

    @property
    def NUM_UE_ANT(self) -> int:
        return self.num_ue_ant

    @property
    def NUM_BS_ANT(self) -> int:
        return self.num_bs_ant

    @property
    def BATCH_SIZE_CIR(self) -> int:
        return self.batch_size_cir

    @property
    def TARGET_NUM_CIRS(self) -> int:
        return self.target_num_cirs

    @property
    def MAX_DEPTH(self) -> int:
        return self.max_depth

    @property
    def MIN_GAIN_DB(self) -> float:
        return self.min_gain_db

    @property
    def MAX_GAIN_DB(self) -> float:
        return self.max_gain_db

    @property
    def MIN_DIST(self) -> float:
        return self.min_dist_m

    @property
    def MAX_DIST(self) -> float:
        return self.max_dist_m

    @property
    def RM_CELL_SIZE(self) -> Tuple[float, float]:
        return self.rm_cell_size

    @property
    def RM_SAMPLES_PER_TX(self) -> int:
        return self.rm_samples_per_tx

    @property
    def RM_VMIN_DB(self) -> float:
        return self.rm_vmin_db

    @property
    def RM_CLIP_AT(self) -> float:
        return self.rm_clip_at

    @property
    def RM_RESOLUTION(self) -> Tuple[int, int]:
        return self.rm_resolution

    @property
    def RM_NUM_SAMPLES(self) -> int:
        return self.rm_num_samples

    @property
    def BATCH_SIZE(self) -> int:
        return self.batch_size

    @property
    def PLOT_TITLE(self) -> str:
        return self.plot_title
