# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Channel Impulse Response (CIR) management for ray-traced PUSCH channels.

Provides unified CIR generation, storage, and loading for the PUSCH
autoencoder demo. It handles the complete pipeline from 3D ray-tracing
in a realistic urban environment (Munich) to TFRecord serialization for
efficient training data loading.

Key Capabilities
----------------
1. **Scene Setup**: Configure Munich urban scene with BS antenna array and
   mobile UE positions within coverage-based sampling constraints.

2. **CIR Generation**: Ray-trace multipath channels for many UE positions,
   computing complex path gains and delays for each TX-RX link.

3. **TFRecord I/O**: Serialize/deserialize CIR tensors for efficient storage
   and fast training data loading via ``tf.data`` pipelines.

4. **MU-MIMO Grouping**: Combine individual UE CIRs into multi-user samples
   for joint processing in the neural detector.

Workflow
--------
Typical usage follows two phases:

**Generation Phase** (run once, offline)::

    # Generate CIR data for 16 BS antennas
    python -m demos.pusch_autoencoder.src.cir_manager --num_bs_ant 16 --seeds 0 1 2

    # Generate CIR data for 32 BS antennas
    python -m demos.pusch_autoencoder.src.cir_manager --num_bs_ant 32 --seeds 0 1 2

**Training Phase** (run many times)::

    cfg = Config(num_bs_ant=16)  # or 32
    manager = CIRManager(config=cfg)
    a, tau = manager.load_from_tfrecord(group_for_mumimo=True)
    # Loads from demos/pusch_autoencoder/cir_tfrecords_ant16 (or ant32)
    # a: [num_mu_samples, 1, 16, 4, 4, max_paths, 14]
    # tau: [num_mu_samples, 1, 4, max_paths]
"""

import os
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy.channel import CIRDataset
from sionna.rt import (
    load_scene,
    Camera,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
    RadioMapSolver,
)

from .config import Config
from .cir_generator import CIRGenerator

# get directory name of file
DEMO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# TensorFlow and GPU Configuration
# =============================================================================


def setup_tensorflow():
    """
    Configure TensorFlow environment for GPU-accelerated ray-tracing.

    This function performs three configuration tasks:

    1. **GPU Selection**: Defaults to GPU 0 if ``CUDA_VISIBLE_DEVICES`` is not set.
       This prevents accidental multi-GPU usage which can cause memory issues
       with Sionna's ray-tracer.

    2. **Log Suppression**: Reduces TensorFlow verbosity to ERROR level only,
       keeping console output clean during long CIR generation runs.

    3. **Memory Growth**: Enables dynamic GPU memory allocation instead of
       pre-allocating all GPU memory. This allows running other processes
       alongside CIR generation and prevents OOM errors when memory needs
       vary between ray-tracing batches.

    Notes
    -----
    This function is called automatically when the module is imported.
    It must run before any TensorFlow operations to take effect.

    The ``set_memory_growth`` call may fail if TensorFlow has already
    initialized the GPU context, in which case a warning is printed
    but execution continues.
    """
    # Default to GPU 0 if not specified (avoid multi-GPU complications)
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Suppress verbose TF logging during long generation runs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    # Enable dynamic memory allocation to avoid OOM with varying batch sizes
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # GPU context may already be initialized
            print(f"GPU configuration error: {e}")


# Run setup on import to ensure configuration happens before any TF operations
setup_tensorflow()


# =============================================================================
# CIRManager Class
# =============================================================================


class CIRManager:
    r"""
    Unified manager for CIR generation, storage, and loading.

    This class encapsulates the complete CIR data pipeline for the PUSCH
    autoencoder demo, including:

    - Munich scene setup with configurable antenna arrays
    - Radio map computation for coverage-aware UE position sampling
    - Ray-traced CIR generation with multipath propagation
    - TFRecord serialization for efficient data storage
    - MU-MIMO sample grouping for multi-user training

    Parameters
    ----------
    config : Config, optional
        Configuration object with system parameters. If ``None``, uses
        default ``Config()`` with standard MU-MIMO settings.

    Pre-conditions
    --------------
    - Sionna must be installed with ray-tracing support (``sionna.rt``).
    - GPU with sufficient memory for ray-tracing (recommended: 8GB+).
    - Munich scene assets must be available (bundled with Sionna).

    Post-conditions
    ---------------
    - After ``setup_scene()``: ``self.scene``, ``self.tx``, ``self.camera`` are set.
    - After ``compute_radio_map()``: ``self.rm`` contains path gain map.
    - After ``generate_cir_data()``: CIR arrays are in NumPy format.

    Invariants
    ----------
    - Config parameters (antenna counts, bandwidths) are fixed after init.
    - Scene geometry (TX position, array orientations) is fixed.

    Example
    -------
    >>> # Generation workflow (offline, run once)
    >>> manager = CIRManager()
    >>> manager.generate_and_save([0, 1, 2], tfrecord_dir="../cir_tfrecords")

    >>> # Loading workflow (online, run each training)
    >>> manager = CIRManager()
    >>> a, tau = manager.load_from_tfrecord(group_for_mumimo=True)
    >>> model = PUSCHLinkE2E((a, tau), ...)

    Notes
    -----
    The CIR generation process is computationally expensive but only
    needs to run once. Generated TFRecords can be reused across
    many training runs with different hyperparameters.

    The ``group_for_mumimo`` option in ``load_from_tfrecord()`` combines
    ``num_ue`` individual CIRs into single MU-MIMO samples, simulating
    co-scheduled uplink transmissions from multiple UEs.
    """

    def __init__(self, config=None):
        """
        Initialize CIRManager with system configuration.

        Parameters
        ----------
        config : Config, optional
            System configuration. Defaults to ``Config()`` if not provided.
        """
        self.cfg = config if config is not None else Config()

        # Cache frequently-used config values for cleaner code
        self.subcarrier_spacing = self.cfg.subcarrier_spacing
        self.num_time_steps = self.cfg.num_time_steps
        self.num_ue = self.cfg.num_ue
        self.num_bs = self.cfg.num_bs
        self.num_ue_ant = self.cfg.num_ue_ant
        self.num_bs_ant = self.cfg.num_bs_ant
        self.batch_size_cir = self.cfg.batch_size_cir
        self.target_num_cirs = self.cfg.target_num_cirs

        # Ray-tracing solver parameters
        self.max_depth = self.cfg.max_depth
        self.min_gain_db = self.cfg.min_gain_db
        self.max_gain_db = self.cfg.max_gain_db
        self.min_dist = self.cfg.min_dist_m
        self.max_dist = self.cfg.max_dist_m

        # Radio map visualization parameters
        self.rm_cell_size = self.cfg.rm_cell_size
        self.rm_samples_per_tx = self.cfg.rm_samples_per_tx
        self.rm_vmin_db = self.cfg.rm_vmin_db
        self.rm_clip_at = self.cfg.rm_clip_at
        self.rm_resolution = self.cfg.rm_resolution
        self.rm_num_samples = self.cfg.rm_num_samples

        self.batch_size = self.cfg.batch_size

        # Lazily initialized scene components
        self.scene = None
        self.tx = None
        self.camera = None
        self.rm = None  # Radio map for coverage-based sampling

    def setup_scene(self):
        """
        Initialize Munich scene with BS and UE antenna configurations.

        This method loads the Munich urban scene and configures:

        - **BS array**: 16x2 cross-polarized panel (32 elements) on a rooftop
        - **UE array**: 2x2 cross-polarized array (4 elements) with isotropic pattern
        - **Camera**: Overhead view for radio map visualization

        Returns
        -------
        scene : sionna.rt.Scene
            Configured scene object ready for ray-tracing.

        Notes
        -----
        The BS is positioned at ``[8.5, 21, 27]`` meters, which places it on
        a building rooftop in the Munich scene. The ``look_at`` direction
        points toward the main coverage area where UEs will be sampled.

        The BS uses ``tr38901`` antenna pattern (3GPP sector antenna) while
        UEs use ``iso`` (isotropic) pattern, reflecting realistic deployments
        where BS antennas are directional but UE antennas are omnidirectional.
        """
        # Load pre-built Munich urban environment
        self.scene = load_scene(sionna.rt.scene.munich)

        # BS antenna array: 8 columns × 2 polarizations = 16 elements
        # tr38901 pattern provides realistic sector antenna gain profile
        self.scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=self.num_bs_ant // 2,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="cross",
        )

        # Place BS on rooftop, oriented toward main street canyon
        self.tx = Transmitter(
            name="tx", position=[8.5, 21, 27], look_at=[45, 90, 1.5], display_radius=3.0
        )
        self.scene.add(self.tx)

        # Overhead camera for coverage map visualization
        self.camera = Camera(
            position=[0, 80, 500], orientation=np.array([0, np.pi / 2, -np.pi / 2])
        )

        # UE antenna array: 2 columns × 2 polarizations = 4 elements
        # Isotropic pattern models omnidirectional handheld device
        self.scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=self.num_ue_ant // 2,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="cross",
        )

        return self.scene

    def compute_radio_map(self, save_images=True):
        """
        Compute path gain radio map for coverage-based UE sampling.

        The radio map provides spatially-resolved path gain information used
        to sample UE positions in areas with valid coverage (avoiding dead
        zones and extreme near-field regions).

        Parameters
        ----------
        save_images : bool
            If ``True``, saves radio map visualization to PNG file.

        Returns
        -------
        rm : sionna.rt.RadioMap
            Radio map object with path gain values per spatial cell.

        Notes
        -----
        Radio map computation uses Monte Carlo ray-tracing with 10^7 samples
        per TX, which takes several minutes but provides smooth coverage maps.
        The result is cached in ``self.rm`` for subsequent UE position sampling.
        """
        if self.scene is None:
            self.setup_scene()

        # Monte Carlo radio map computation
        rm_solver = RadioMapSolver()
        self.rm = rm_solver(
            self.scene,
            max_depth=self.max_depth,
            cell_size=self.rm_cell_size,
            samples_per_tx=self.rm_samples_per_tx,
        )

        if save_images:
            # Render coverage map with path gain colorscale
            self.scene.render_to_file(
                camera=self.camera,
                radio_map=self.rm,
                rm_vmin=self.rm_vmin_db,
                clip_at=self.rm_clip_at,
                resolution=list(self.rm_resolution),
                filename="munich_radio_map.png",
                num_samples=self.rm_num_samples,
            )

        return self.rm

    def generate_cir_data(self, seed_offset=0, max_num_paths=0):
        """
        Generate ray-traced CIR data for multiple UE positions.

        This method performs the core ray-tracing loop:

        1. Sample UE positions from radio map (coverage-aware)
        2. Place receiver objects at sampled positions
        3. Run path solver to compute multipath propagation
        4. Extract CIR (path gains and delays) for each link

        Parameters
        ----------
        seed_offset : int
            Random seed offset for reproducible position sampling.
            Different offsets produce different UE position sets.
        max_num_paths : int
            Initial value for path count tracking. Updated during generation.

        Returns
        -------
        a : np.ndarray, complex64
            CIR coefficients with shape
            ``[num_samples, 1, num_bs_ant, 1, num_ue_ant, max_paths, num_time_steps]``
        tau : np.ndarray, float32
            Path delays with shape ``[num_samples, 1, 1, max_paths]``
        max_num_paths : int
            Maximum number of paths across all generated samples.

        Notes
        -----
        The generation loop processes ``batch_size_cir`` positions at a time,
        updating receiver positions and re-running the path solver. Progress
        is printed to console since this can take hours for large datasets.

        Empty CIRs (zero total power) are filtered out, as they represent
        positions with no valid propagation paths (e.g., inside buildings).

        The seed formula ``idx + seed_offset * 1000`` ensures different files
        (with different seed_offsets) produce non-overlapping position samples.
        """
        if self.rm is None:
            self.compute_radio_map(save_images=False)

        # Initial UE position sampling to create receiver objects
        ue_pos, _ = self.rm.sample_positions(
            num_pos=self.batch_size_cir,
            metric="path_gain",
            min_val_db=self.min_gain_db,
            max_val_db=self.max_gain_db,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
        )

        # Create receiver objects at sampled positions
        for i in range(self.batch_size_cir):
            p = ue_pos[0, i, :]
            if hasattr(p, "numpy"):
                p = p.numpy()
            p = np.asarray(p, dtype=np.float64)

            # Remove existing receiver if present (for re-runs)
            try:
                self.scene.remove(f"rx-{i}")
            except Exception:
                pass

            # Add receiver with mobility (3 m/s) for Doppler modeling
            rx = Receiver(
                name=f"rx-{i}",
                position=(float(p[0]), float(p[1]), float(p[2])),
                velocity=(3.0, 3.0, 0.0),
                display_radius=1.0,
                color=(1, 0, 0),
            )
            self.scene.add(rx)

        # Main CIR generation loop
        p_solver = PathSolver()
        a_list, tau_list = [], []
        num_runs = int(np.ceil(self.target_num_cirs / self.batch_size_cir))

        for idx in range(num_runs):
            print(f"Progress: {idx+1}/{num_runs}", end="\r", flush=True)

            # Sample fresh positions for each batch
            ue_pos, _ = self.rm.sample_positions(
                num_pos=self.batch_size_cir,
                metric="path_gain",
                min_val_db=self.min_gain_db,
                max_val_db=self.max_gain_db,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                seed=idx + seed_offset * 1000,  # Reproducible, non-overlapping
            )

            # Update receiver positions (faster than recreating objects)
            for rx in range(self.batch_size_cir):
                p = ue_pos[0, rx, :]
                if hasattr(p, "numpy"):
                    p = p.numpy()
                p = np.asarray(p, dtype=np.float64)
                self.scene.receivers[f"rx-{rx}"].position = (
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                )

            # Ray-trace all paths up to max_depth reflections
            paths = p_solver(
                self.scene, max_depth=self.max_depth, max_num_paths_per_src=10**7
            )

            # Extract CIR from path data
            a, tau = paths.cir(
                sampling_frequency=self.subcarrier_spacing,
                num_time_steps=self.num_time_steps,
                out_type="numpy",
            )
            a = a.astype(np.complex64)
            tau = tau.astype(np.float32)
            a_list.append(a)
            tau_list.append(tau)

            # Track maximum path count for padding
            num_paths = a.shape[-2]
            max_num_paths = max(max_num_paths, num_paths)

        # Pad all samples to common path dimension and concatenate
        a, tau = [], []
        for a_, tau_ in zip(a_list, tau_list):
            num_paths = a_.shape[-2]
            a.append(
                np.pad(
                    a_,
                    [
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, max_num_paths - num_paths],
                        [0, 0],
                    ],
                    constant_values=0,
                ).astype(np.complex64)
            )

            tau.append(
                np.pad(
                    tau_,
                    [[0, 0], [0, 0], [0, max_num_paths - num_paths]],
                    constant_values=0,
                ).astype(np.float32)
            )

        a = np.concatenate(a, axis=0)
        tau = np.concatenate(tau, axis=0)

        # Reorder dimensions to match Sionna CIRDataset expectations
        a = np.transpose(a, (2, 3, 0, 1, 4, 5))
        tau = np.transpose(tau, (1, 0, 2))

        a = np.expand_dims(a, axis=0)
        tau = np.expand_dims(tau, axis=0)

        a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])
        tau = np.transpose(tau, [2, 1, 0, 3])

        # Filter out samples with zero power (no valid paths)
        p_link = np.sum(np.abs(a) ** 2, axis=(1, 2, 3, 4, 5, 6))
        a = a[p_link > 0, ...]
        tau = tau[p_link > 0, ...]

        print("(in cir.py) a.shape: ", a.shape)
        print("(in cir.py) tau.shape: ", tau.shape)

        return a, tau, max_num_paths

    def save_to_tfrecord(self, a, tau, filename):
        """
        Serialize CIR data to TFRecord format for efficient storage.

        TFRecord provides efficient sequential reading, compression, and
        seamless integration with ``tf.data`` pipelines for training.

        Parameters
        ----------
        a : np.ndarray, complex64
            CIR coefficients, shape ``[num_samples, ...]``.
        tau : np.ndarray, float32
            Path delays, shape ``[num_samples, ...]``.
        filename : str
            Output TFRecord file path.

        Notes
        -----
        Each sample is stored as a separate TFRecord ``Example`` containing:

        - ``a``: Serialized complex64 tensor (path gains)
        - ``tau``: Serialized float32 tensor (path delays)
        - ``a_shape``: Shape metadata for deserialization
        - ``tau_shape``: Shape metadata for deserialization

        The shape metadata enables reconstruction of tensors with varying
        path counts across different TFRecord files.
        """

        def _bytes_feature(value):
            """Convert tensor bytes to TFRecord bytes feature."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_list_feature(value):
            """Convert shape tuple to TFRecord int64 list feature."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(len(a)):
                # Extract single sample
                a_sample = a[i]
                tau_sample = tau[i]

                # Serialize tensors to bytes
                a_bytes = tf.io.serialize_tensor(a_sample).numpy()
                tau_bytes = tf.io.serialize_tensor(tau_sample).numpy()

                # Store shape for deserialization
                a_shape = a_sample.shape
                tau_shape = tau_sample.shape

                # Build feature dictionary
                feature = {
                    "a": _bytes_feature(a_bytes),
                    "tau": _bytes_feature(tau_bytes),
                    "a_shape": _int64_list_feature(a_shape),
                    "tau_shape": _int64_list_feature(tau_shape),
                }

                # Write serialized Example
                features = tf.train.Features(feature=feature)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

        print(f"  Saved {len(a)} samples to {filename}")

    def load_from_tfrecord(self, tfrecord_dir=None, group_for_mumimo=False):
        """
        Load CIR data from TFRecord files with optional MU-MIMO grouping.

        Parameters
        ----------
        tfrecord_dir : str, optional
            Directory containing TFRecord files, relative to this module.
            If not provided, defaults to ``cir_tfrecords_ant{num_bs_ant}`` within the demo directory.
        group_for_mumimo : bool
            If ``True``, groups ``num_ue`` individual CIRs into MU-MIMO samples.
            This simulates co-scheduled uplink transmissions.

        Returns
        -------
        all_a : tf.Tensor, complex64
            CIR coefficients. Shape depends on ``group_for_mumimo``:

            - ``False``: ``[num_samples, 1, num_bs_ant, 1, num_ue_ant, max_paths, num_time_steps]``
            - ``True``: ``[num_mu_samples, 1, num_bs_ant, num_ue, num_ue_ant, max_paths, num_time_steps]``

        all_tau : tf.Tensor, float32
            Path delays. Shape depends on ``group_for_mumimo``:

            - ``False``: ``[num_samples, 1, 1, max_paths]``
            - ``True``: ``[num_mu_samples, 1, num_ue, max_paths]``

        Notes
        -----
        When ``group_for_mumimo=True``, consecutive samples are combined to
        form multi-user scenarios. For example, with ``num_ue=4``, samples
        0-3 become MU-MIMO sample 0, samples 4-7 become sample 1, etc.

        This grouping is valid because each original sample represents an
        independent UE position, and combining them simulates the realistic
        scenario where multiple UEs transmit simultaneously.
        """
        # Default directory includes antenna count
        if tfrecord_dir is None:
            cir_dir = os.path.join(DEMO_DIR, f"cir_tfrecords_ant{self.num_bs_ant}")
        elif os.path.isabs(tfrecord_dir):
            cir_dir = tfrecord_dir
        else:
            cir_dir = os.path.join(DEMO_DIR, tfrecord_dir)
        cir_files = tf.io.gfile.glob(os.path.join(cir_dir, "*.tfrecord"))

        if not cir_files:
            raise ValueError(f"No TFRecord files found in {cir_dir}")

        # TFRecord parsing specification
        feature_description = {
            "a": tf.io.FixedLenFeature([], tf.string),
            "tau": tf.io.FixedLenFeature([], tf.string),
            "a_shape": tf.io.VarLenFeature(tf.int64),
            "tau_shape": tf.io.VarLenFeature(tf.int64),
        }

        def _parse_example(example_proto):
            """Parse single TFRecord example to (a, tau) tensors."""
            parsed = tf.io.parse_single_example(example_proto, feature_description)

            # Deserialize tensor bytes back to tensors
            a = tf.io.parse_tensor(parsed["a"], out_type=tf.complex64)
            tau = tf.io.parse_tensor(parsed["tau"], out_type=tf.float32)

            # Recover shape from metadata (sparse -> dense conversion)
            a_shape = tf.sparse.to_dense(parsed["a_shape"])
            tau_shape = tf.sparse.to_dense(parsed["tau_shape"])

            # Reshape to original dimensions
            a = tf.reshape(a, a_shape)
            tau = tf.reshape(tau, tau_shape)

            return a, tau

        # Load and concatenate all samples from all files
        ds = tf.data.TFRecordDataset(cir_files)
        ds = ds.map(_parse_example)

        all_a = []
        all_tau = []
        for a, tau in ds:
            all_a.append(a)
            all_tau.append(tau)

        all_a = tf.concat(all_a, axis=0)
        all_tau = tf.concat(all_tau, axis=0)
        all_a = tf.expand_dims(all_a, axis=1)
        all_tau = tf.expand_dims(all_tau, axis=1)

        if group_for_mumimo:
            # Combine num_ue individual CIRs into MU-MIMO samples
            # This simulates co-scheduled uplink from multiple UEs
            num_ue = self.num_ue  # 4
            num_bs_ant = self.num_bs_ant
            num_samples = tf.shape(all_a)[0]
            num_mu_samples = num_samples // num_ue

            # Truncate to exact multiple of num_ue
            all_a = all_a[: num_mu_samples * num_ue]
            all_tau = all_tau[: num_mu_samples * num_ue]

            # Reshape: [N*4, 1, 16, 1, 4, paths, 14] -> [N, 1, 16, 4, 4, paths, 14]
            all_a = tf.reshape(
                all_a, [num_mu_samples, num_ue, 1, num_bs_ant, 1, 4, 51, 14]
            )
            all_a = tf.squeeze(all_a, axis=4)  # [N, 4, 1, 16, 4, 51, 14]
            all_a = tf.transpose(
                all_a, [0, 2, 3, 1, 4, 5, 6]
            )  # [N, 1, 16, 4, 4, 51, 14]

            # Reshape: [N*4, 1, 1, paths] -> [N, 1, 4, paths]
            all_tau = tf.reshape(all_tau, [num_mu_samples, num_ue, 1, 1, 51])
            all_tau = tf.squeeze(all_tau, axis=3)  # [N, 4, 1, 51]
            all_tau = tf.transpose(all_tau, [0, 2, 1, 3])  # [N, 1, 4, 51]

        return all_a, all_tau

    def build_channel_model(
        self,
        batch_size=None,
        num_bs=None,
        num_bs_ant=None,
        num_ue=None,
        num_ue_ant=None,
        num_time_steps=None,
        tfrecord_dir=None,
    ):
        """
        Build CIRDataset channel model from TFRecord files.

        This method creates a Sionna ``CIRDataset`` that can be used with
        ``OFDMChannel`` for baseline BER/BLER evaluation. The dataset
        provides on-demand CIR sampling during simulation.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for dataset. Default from config.
        num_bs : int, optional
            Number of base stations. Default from config.
        num_bs_ant : int, optional
            BS antenna count. Default from config.
        num_ue : int, optional
            Number of UEs. Default from config.
        num_ue_ant : int, optional
            UE antenna count. Default from config.
        num_time_steps : int, optional
            OFDM symbols per slot. Default from config.
        tfrecord_dir : str, optional
            Directory containing TFRecord files.
            If not provided, defaults to ``cir_tfrecords_ant{num_bs_ant}`` within the demo directory.

        Returns
        -------
        channel_model : CIRDataset
            Channel model compatible with Sionna's ``OFDMChannel``.

        Notes
        -----
        This method is primarily used for baseline evaluation where
        a ``CIRDataset`` object is needed. For autoencoder training, use
        ``load_from_tfrecord(group_for_mumimo=True)`` directly to get
        tensors that can be indexed during training.
        """
        # Use config values as defaults
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_bs = num_bs if num_bs is not None else self.num_bs
        num_bs_ant = num_bs_ant if num_bs_ant is not None else self.num_bs_ant
        num_ue = num_ue if num_ue is not None else self.num_ue
        num_ue_ant = num_ue_ant if num_ue_ant is not None else self.num_ue_ant
        num_time_steps = (
            num_time_steps if num_time_steps is not None else self.num_time_steps
        )

        # Load raw CIR data (not MU-MIMO grouped for CIRDataset)
        # tfrecord_dir=None will use the antenna-specific default in load_from_tfrecord
        all_a, all_tau = self.load_from_tfrecord(tfrecord_dir)
        max_num_paths = all_a.shape[-2]

        # Wrap in generator for CIRDataset consumption
        cir_generator = CIRGenerator(all_a, all_tau, num_ue)

        # Create Sionna CIRDataset
        channel_model = CIRDataset(
            cir_generator,
            batch_size,
            num_bs,
            num_bs_ant,
            num_ue,
            num_ue_ant,
            max_num_paths,
            num_time_steps,
        )

        return channel_model

    def save_visualization_ue_positions(self, filename="munich_ue_positions.png"):
        """
        Render radio map with current UE positions overlaid.

        Parameters
        ----------
        filename : str
            Output PNG file path.

        Raises
        ------
        RuntimeError
            If scene, radio map, or camera are not initialized.
        """
        if self.scene is None or self.rm is None or self.camera is None:
            raise RuntimeError(
                "Scene, radio map, or camera not initialized. "
                "Call setup_scene() and compute_radio_map() first."
            )

        self.scene.render_to_file(
            camera=self.camera,
            radio_map=self.rm,
            rm_vmin=self.rm_vmin_db,
            clip_at=self.rm_clip_at,
            resolution=list(self.rm_resolution),
            filename=filename,
            num_samples=self.rm_num_samples,
        )

    def generate_and_save(
        self,
        seed_offsets,
        tfrecord_dir=None,
        save_radio_map=True,
    ):
        """
        Complete CIR generation pipeline: generate, visualize, and save.

        This is the main entry point for offline CIR dataset creation.
        It handles scene setup, radio map computation, CIR generation,
        and TFRecord serialization in a single call.

        Parameters
        ----------
        seed_offsets : int or list of int
            Random seed offset(s) for UE position sampling.
            Each seed produces a separate TFRecord file.
        tfrecord_dir : str, optional
            Output directory for TFRecord files (relative to this module).
            If not provided, defaults to ``cir_tfrecords_ant{num_bs_ant}`` within the demo directory.
        save_radio_map : bool
            If ``True``, saves radio map and UE position visualizations.

        Example
        -------
        >>> manager = CIRManager()
        >>> # Generate 3 files with 5000 CIRs each (15000 total)
        >>> manager.generate_and_save([0, 1, 2])

        Notes
        -----
        Multiple seed offsets produce independent position samples, which
        can be beneficial for:

        1. Increasing total dataset size beyond single-file limits
        2. Enabling parallel generation on multiple machines
        3. Creating train/validation/test splits with different seeds

        The output directory includes the antenna count suffix to keep
        CIR data for different antenna configurations separate.
        """
        # Default directory includes antenna count for separation
        if tfrecord_dir is None:
            tfrecord_dir = os.path.join(DEMO_DIR, f"cir_tfrecords_ant{self.num_bs_ant}")
        # Normalize input to list
        if isinstance(seed_offsets, (int, np.integer)):
            seed_list = [int(seed_offsets)]
        elif isinstance(seed_offsets, (list, tuple, np.ndarray)):
            seed_list = [int(s) for s in seed_offsets]
        else:
            raise ValueError("seed_offsets must be an int or a list/tuple of ints")

        # Initialize scene and compute coverage map
        self.setup_scene()
        self.compute_radio_map(save_images=save_radio_map)

        # Flag to save UE position visualization (once, after first generation)
        need_ue_viz = save_radio_map

        # Create output directory (relative to this module file)
        if os.path.isabs(tfrecord_dir):
            cir_dir = tfrecord_dir
        else:
            cir_dir = os.path.join(DEMO_DIR, tfrecord_dir)
        os.makedirs(cir_dir, exist_ok=True)

        # Track maximum path count across all files for documentation
        max_num_paths_all = 0

        # Generate one TFRecord file per seed
        for idx, seed in enumerate(seed_list):
            print(
                f"\nGenerating CIR file {idx+1}/{len(seed_list)}  (seed_offset={seed})"
            )

            # Generate CIR data for this seed
            a, tau, max_num_paths = self.generate_cir_data(seed_offset=seed)

            max_num_paths_all = max(max_num_paths_all, max_num_paths)

            print(f"  a.shape={a.shape}, tau.shape={tau.shape}")
            print(f"  max_num_paths={max_num_paths}")

            # Save UE position visualization (once, after receivers are created)
            if need_ue_viz:
                ue_fig = os.path.join(
                    cir_dir, f"munich_ue_positions_seed_{seed:03d}.png"
                )
                try:
                    self.save_visualization_ue_positions(filename=ue_fig)
                    print(f"  Saved UE position visualization to '{ue_fig}'")
                except Exception as e:
                    print(f"  Warning: failed to save UE visualization: {e}")
                need_ue_viz = False

            # Save CIR data to TFRecord
            filename = os.path.join(cir_dir, f"cir_seed_{seed:03d}.tfrecord")
            self.save_to_tfrecord(a, tau, filename)

        print(f"\nSuccessfully generated {len(seed_list)} TFRecord files.")
        print(f"All files saved in '{cir_dir}' directory.")
        print(f"Global max_num_paths across all seeds = {max_num_paths_all}")


# =============================================================================
# Main Entry Point for Standalone CIR Generation
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CIR data for PUSCH autoencoder training."
    )
    parser.add_argument(
        "--num_bs_ant",
        type=int,
        default=16,
        choices=[16, 32],
        help="Number of BS antennas (default: 16)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Seed offset(s) for CIR generation (default: [0])",
    )

    args = parser.parse_args()

    print("\n CIR Generation Started")
    print(f"  num_bs_ant: {args.num_bs_ant}")
    print(f"  seeds: {args.seeds}")

    try:
        cfg = Config(num_bs_ant=args.num_bs_ant)
        cir_manager = CIRManager(config=cfg)
        cir_manager.generate_and_save(args.seeds)
        print("\n CIR Generation Completed Successfully \n")
    except Exception as e:
        print("\n!!! CIR Generation Failed !!!")
        print(f"Error: {e}\n")
        raise
