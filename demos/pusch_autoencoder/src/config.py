# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Central configuration for the PUSCH Autoencoder demo.

Provides a single source of truth for all system parameters used
across CIR generation, and autoencoder training and inference.
Using a centralized config ensures consistency between the
ray-tracing channel generation and the communication system simulation.
"""

from dataclasses import dataclass, field
from typing import Tuple
from typing import List
import tensorflow as tf
from sionna.phy.nr.utils import MCSDecoderNR


@dataclass
class Config:
    r"""
    Central configuration for the PUSCH-Autoencoder demo.

    This dataclass consolidates all hard-coded system parameters to ensure
    consistency across CIR generation, model instantiation, training, and
    inference. Parameters are intentionally non-configurable at instantiation
    (``init=False``) to enforce a single, validated system configuration.
    The number of BS antennas (``num_bs_ant``) is configurable
    to allow flexibility in array size.

    The configuration defines a MU-MIMO uplink scenario with:

    - 4 UEs, each with 4 antennas (cross-polarized)
    - 1 BS with configurable antennas (default 16, cross-polarized)
    - 16-QAM modulation (MCS index 14, table 1)
    - Site- (Munich-) specific ray-traced channel

    Parameters
    ----------
    num_bs_ant : int, optional
        Number of BS antennas (cross-polarized array). Must be even since
        the antenna array uses cross-polarization (num_cols = num_bs_ant // 2).
        Default is 16.

    Notes
    -----
    - After ``__post_init__``, ``num_bits_per_symbol`` and ``target_coderate``
      are populated from the 3GPP NR MCS tables.
    - All properties return consistent, validated values.
    - System dimensions (antennas, UEs) remain fixed after instantiation.
    - PUSCH-related properties (``resource_grid``, ``pusch_pilot_indices``, etc.)
      are set externally by ``PUSCHLinkE2E`` after transmitter construction.

    Example
    -------
    >>> cfg = Config()
    >>> print(cfg.num_bits_per_symbol)  # 4 for 16-QAM
    4
    >>> print(cfg.num_ue, cfg.num_bs_ant)
    4 16
    >>> cfg_32ant = Config(num_bs_ant=32)
    >>> print(cfg_32ant.num_bs_ant)
    32

    Notes
    -----
    The MCS decoder call in ``__post_init__`` uses ``transform_precoding=True``
    and ``pi2bpsk=False``, which is appropriate for standard PUSCH without
    DFT-s-OFDM transform precoding at the physical layer.
    """

    # =========================================================================
    # User-Configurable Parameters
    # =========================================================================
    # Number of BS antennas (cross-polarized). Default 16 provides sufficient
    # spatial DoF to separate 4 single-layer UE streams with reasonable margin.
    # Must be even since the antenna array uses cross-polarization.
    num_bs_ant: int = 16

    # =========================================================================
    # OFDM / Slot Structure
    # =========================================================================
    # Subcarrier spacing determines the slot duration and Doppler tolerance.
    # 30 kHz is standard for FR1 (sub-6 GHz) NR deployments.
    _subcarrier_spacing: float = field(init=False, default=30e3, repr=False)

    # 14 OFDM symbols per slot is the standard NR slot structure with
    # normal cyclic prefix. This must match the CIR time-domain sampling.
    _num_time_steps: int = field(init=False, default=14, repr=False)

    # =========================================================================
    # MIMO Configuration
    # =========================================================================
    # 4 UEs enables meaningful MU-MIMO interference while keeping
    # computational complexity manageable for the purpose of this demo.
    _num_ue: int = field(init=False, default=4, repr=False)

    # Single base station - multi-cell scenarios not currently supported.
    _num_bs: int = field(init=False, default=1, repr=False)

    # 4 UE antennas (2x2 cross-polarized) matches typical smartphone designs
    # and enables spatial multiplexing on the uplink.
    _num_ue_ant: int = field(init=False, default=4, repr=False)

    # =========================================================================
    # CIR Generation Parameters
    # =========================================================================
    # Batch size for ray-tracing CIR generation. Larger batches improve
    # GPU utilization but increase memory requirements.
    _batch_size_cir: int = field(init=False, default=500, repr=False)

    # Total CIRs to generate. 5000 provides reasonable channel diversity
    # for training while keeping generation time practical.
    _target_num_cirs: int = field(init=False, default=5000, repr=False)

    # =========================================================================
    # PUSCH Resource Grid (set externally after transmitter construction)
    # =========================================================================
    # These are populated by PUSCHLinkE2E after PUSCHTransmitter creates
    # the resource grid. They enable the neural detector to identify
    # pilot vs. data symbol positions.
    _resource_grid: object = field(init=False, default=None, repr=False)
    _pusch_pilot_indices: List[int] = field(init=False, repr=False)
    _pusch_num_subcarriers: int = field(init=False, default=1, repr=False)
    _pusch_num_symbols_per_slot: int = field(init=False, default=1, repr=False)

    # =========================================================================
    # Ray-Tracing Path Solver Configuration
    # =========================================================================
    # Maximum reflection depth. 5 captures dominant multipath while
    # avoiding excessive computation from high-order reflections.
    _max_depth: int = field(init=False, default=5, repr=False)

    # Path gain thresholds for UE position sampling. Positions with path
    # gain outside [-130, 0] dB are excluded to avoid dead zones and
    # unrealistic near-field scenarios.
    _min_gain_db: float = field(init=False, default=-130.0, repr=False)
    _max_gain_db: float = field(init=False, default=0.0, repr=False)

    # Sampling annulus distances from BS. Inner radius (5m) avoids
    # near-field effects; outer radius (400m) covers typical urban cell.
    _min_dist_m: float = field(init=False, default=5.0, repr=False)
    _max_dist_m: float = field(init=False, default=400.0, repr=False)

    # =========================================================================
    # Radio Map Visualization Parameters
    # =========================================================================
    # Cell size for radio map discretization (meters). 1m provides
    # sufficient resolution for urban canyon visualization.
    _rm_cell_size: Tuple[float, float] = field(
        init=False, default=(1.0, 1.0), repr=False
    )

    # Monte Carlo samples per TX for radio map computation. 10^7 samples
    # provides smooth coverage maps with low variance.
    _rm_samples_per_tx: int = field(init=False, default=10**7, repr=False)

    # Visualization thresholds. -110 dBm floor matches typical UE sensitivity;
    # clip_at=12 prevents outliers from dominating the colormap.
    _rm_vmin_db: float = field(init=False, default=-110.0, repr=False)
    _rm_clip_at: float = field(init=False, default=12.0, repr=False)

    # Output image resolution (width, height) in pixels.
    _rm_resolution: Tuple[int, int] = field(init=False, default=(650, 500), repr=False)

    # Samples per pixel for anti-aliased rendering.
    _rm_num_samples: int = field(init=False, default=4096, repr=False)

    # =========================================================================
    # Training / Simulation Parameters
    # =========================================================================
    # Batch size for BER/BLER simulation and training. Must be consistent
    # with CIRDataset batching to ensure proper tensor shapes.
    _batch_size: int = field(init=False, default=32, repr=False)

    # Global random seed for reproducibility across CIR generation,
    # weight initialization, and sampling.
    _seed: int = field(init=False, default=42, repr=False)

    # =========================================================================
    # PUSCH / 5G NR Physical Layer Parameters
    # =========================================================================
    # 16 PRBs = 192 subcarriers, suitable for moderate throughput scenario.
    _num_prb: int = field(init=False, default=16, repr=False)

    # MCS index 14 with table 1 yields 16-QAM with ~0.6 coderate,
    # providing a good balance between spectral efficiency and robustness.
    _mcs_index: int = field(init=False, default=14, repr=False)

    # Single layer per UE simplifies the autoencoder while still enabling
    # meaningful MU-MIMO scenarios with 4 co-scheduled UEs.
    _num_layers: int = field(init=False, default=1, repr=False)

    # MCS table 1 is the default for PUSCH without 256-QAM support.
    _mcs_table: int = field(init=False, default=1, repr=False)

    # Frequency-domain processing avoids OFDM modulation/demodulation
    # overhead and simplifies the neural network input structure.
    _domain: str = field(init=False, default="freq", repr=False)

    # Derived from MCS tables in __post_init__. Not set at field definition
    # since they require MCSDecoderNR computation.
    _num_bits_per_symbol: int = field(init=False, repr=False)
    _target_coderate: float = field(init=False, repr=False)

    def __post_init__(self):
        """
        Derive modulation order and coderate from 3GPP NR MCS tables.

        This method is called automatically after dataclass initialization.
        It uses Sionna's MCSDecoderNR to look up the modulation order
        (bits per symbol) and target coderate for the configured MCS index.

        The derived values are essential for:

        - Constellation initialization (num_bits_per_symbol determines QAM order)
        - Eb/N0 to noise variance conversion (requires coderate)
        - Transport block size computation

        Notes
        -----
        ``transform_precoding=True`` disables DFT-s-OFDM specific adjustments.
        ``pi2bpsk=False`` uses standard BPSK rather than pi/2-BPSK rotation.
        ``mcs_category=0`` selects the base MCS table without any offsets.
        """
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

        # Convert to Python scalars for use outside TensorFlow graphs
        self._num_bits_per_symbol = int(modulation_order.numpy())
        self._target_coderate = float(target_coderate.numpy())

        # Placeholder for DMRS symbol indices; actual values set by PUSCHLinkE2E
        self._pusch_pilot_indices = [0, 0]

    # =========================================================================
    # Read-Only Properties: OFDM / Slot Structure
    # =========================================================================
    @property
    def subcarrier_spacing(self) -> float:
        r"""float: Subcarrier spacing in Hz (30 kHz for FR1 NR)."""
        return self._subcarrier_spacing

    @property
    def num_time_steps(self) -> int:
        """int: Number of OFDM symbols per slot (14 with normal CP)."""
        return self._num_time_steps

    # =========================================================================
    # Read-Only Properties: MIMO Configuration
    # =========================================================================
    @property
    def num_ue(self) -> int:
        """int: Number of User Equipment (UE) devices in the MU-MIMO scenario."""
        return self._num_ue

    @property
    def num_bs(self) -> int:
        """int: Number of base stations (currently single-cell only)."""
        return self._num_bs

    @property
    def num_ue_ant(self) -> int:
        """int: Number of antennas per UE (4 = 2x2 cross-polarized array)."""
        return self._num_ue_ant

    # =========================================================================
    # Read-Only Properties: CIR Generation
    # =========================================================================
    @property
    def batch_size_cir(self) -> int:
        """int: Batch size for ray-tracing CIR generation."""
        return self._batch_size_cir

    @property
    def target_num_cirs(self) -> int:
        """int: Total number of CIR realizations to generate."""
        return self._target_num_cirs

    # =========================================================================
    # Read-Only Properties: Ray-Tracing Configuration
    # =========================================================================
    @property
    def max_depth(self) -> int:
        """int: Maximum number of ray reflections in path tracing."""
        return self._max_depth

    @property
    def min_gain_db(self) -> float:
        """float: Minimum path gain threshold (dB) for UE position sampling."""
        return self._min_gain_db

    @property
    def max_gain_db(self) -> float:
        """float: Maximum path gain threshold (dB) for UE position sampling."""
        return self._max_gain_db

    @property
    def min_dist_m(self) -> float:
        """float: Minimum UE-BS distance (m) for position sampling."""
        return self._min_dist_m

    @property
    def max_dist_m(self) -> float:
        """float: Maximum UE-BS distance (m) for position sampling."""
        return self._max_dist_m

    # =========================================================================
    # Read-Only Properties: Radio Map Visualization
    # =========================================================================
    @property
    def rm_cell_size(self) -> Tuple[float, float]:
        """Tuple[float, float]: Radio map cell size (x, y) in meters."""
        return self._rm_cell_size

    @property
    def rm_samples_per_tx(self) -> int:
        """int: Monte Carlo samples per transmitter for radio map."""
        return self._rm_samples_per_tx

    @property
    def rm_vmin_db(self) -> float:
        """float: Minimum value (dB) for radio map colormap."""
        return self._rm_vmin_db

    @property
    def rm_clip_at(self) -> float:
        """float: Clipping threshold for radio map visualization."""
        return self._rm_clip_at

    @property
    def rm_resolution(self) -> Tuple[int, int]:
        """Tuple[int, int]: Radio map image resolution (width, height) pixels."""
        return self._rm_resolution

    @property
    def rm_num_samples(self) -> int:
        """int: Anti-aliasing samples per pixel for rendering."""
        return self._rm_num_samples

    # =========================================================================
    # Read-Only Properties: Training / Simulation
    # =========================================================================
    @property
    def batch_size(self) -> int:
        """int: Batch size for training and BER/BLER simulation."""
        return self._batch_size

    @property
    def seed(self) -> int:
        """int: Global random seed for reproducibility."""
        return self._seed

    # =========================================================================
    # Read-Only Properties: PUSCH / 5G NR Configuration
    # =========================================================================
    @property
    def num_prb(self) -> int:
        """int: Number of Physical Resource Blocks (PRBs) allocated."""
        return self._num_prb

    @property
    def mcs_index(self) -> int:
        """int: Modulation and Coding Scheme index (0-28 for table 1)."""
        return self._mcs_index

    @property
    def num_layers(self) -> int:
        """int: Number of MIMO layers per UE (spatial streams)."""
        return self._num_layers

    @property
    def mcs_table(self) -> int:
        """int: MCS table index (1=64-QAM max, 2=256-QAM, 3=64-QAM low-SE)."""
        return self._mcs_table

    @property
    def domain(self) -> str:
        """str: Processing domain ('freq' or 'time') for transmitter output."""
        return self._domain

    @property
    def num_bits_per_symbol(self) -> float:
        """int: Bits per constellation symbol (derived from MCS)."""
        return self._num_bits_per_symbol

    @property
    def target_coderate(self) -> float:
        """float: Target code rate (derived from MCS, typically 0.3-0.9)."""
        return self._target_coderate

    # =========================================================================
    # Read/Write Properties: PUSCH Resource Grid (set by PUSCHLinkE2E)
    # =========================================================================
    @property
    def resource_grid(self):
        """
        ResourceGrid: Sionna OFDM resource grid object.

        Set externally by ``PUSCHLinkE2E`` after transmitter construction.
        Used by the neural detector to determine grid dimensions.
        """
        return self._resource_grid

    @property
    def pusch_pilot_indices(self):
        """
        List[int]: OFDM symbol indices containing DMRS pilots.

        Set externally by ``PUSCHLinkE2E`` from ``PUSCHConfig.dmrs_symbol_indices``.
        The neural detector uses this to separate pilot and data processing.
        """
        return self._pusch_pilot_indices

    @property
    def pusch_num_subcarriers(self):
        """
        int: Number of subcarriers in the PUSCH allocation.

        Set externally by ``PUSCHLinkE2E`` from ``PUSCHConfig.num_subcarriers``.
        """
        return self._pusch_num_subcarriers

    @property
    def pusch_num_symbols_per_slot(self):
        """
        int: Number of OFDM symbols per slot in the carrier configuration.

        Set externally by ``PUSCHLinkE2E`` from carrier settings.
        """
        return self._pusch_num_symbols_per_slot

    # =========================================================================
    # Setters: PUSCH Resource Grid (called by PUSCHLinkE2E during init)
    # =========================================================================
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
