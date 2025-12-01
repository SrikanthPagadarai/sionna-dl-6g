import os
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy.channel import CIRDataset
from sionna.rt import (
    load_scene, Camera, Transmitter, Receiver,
    PlanarArray, PathSolver, RadioMapSolver
)

from .config import Config
from .cir_generator import CIRGenerator

# ============================================================================
# TensorFlow and GPU Configuration
# ============================================================================
def setup_tensorflow():
    """Configure TensorFlow and GPU settings."""
    # Set GPU device if not already specified
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Suppress TensorFlow info/warning logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

# Run setup on import
setup_tensorflow()


# ============================================================================
# CIRManager Class - Unified Channel Impulse Response Management
# ============================================================================
class CIRManager:
    """Unified class for CIRManager generation, storage, and loading."""
    
    def __init__(self, config=None):
        """Initialize CIRManager with configuration.
        
        Args:
            config: Config object. If None, uses default Config()
        """
        self.cfg = config if config is not None else Config()
        
        # Store frequently used config values
        self.subcarrier_spacing = self.cfg.subcarrier_spacing
        self.num_time_steps = self.cfg.num_time_steps
        self.num_ue = self.cfg.num_ue
        self.num_bs = self.cfg.num_bs
        self.num_ue_ant = self.cfg.num_ue_ant
        self.num_bs_ant = self.cfg.num_bs_ant
        self.batch_size_cir = self.cfg.batch_size_cir
        self.target_num_cirs = self.cfg.target_num_cirs
        
        # Solver parameters
        self.max_depth = self.cfg.max_depth
        self.min_gain_db = self.cfg.min_gain_db
        self.max_gain_db = self.cfg.max_gain_db
        self.min_dist = self.cfg.min_dist_m
        self.max_dist = self.cfg.max_dist_m
        
        # Radio map parameters
        self.rm_cell_size = self.cfg.rm_cell_size
        self.rm_samples_per_tx = self.cfg.rm_samples_per_tx
        self.rm_vmin_db = self.cfg.rm_vmin_db
        self.rm_clip_at = self.cfg.rm_clip_at
        self.rm_resolution = self.cfg.rm_resolution
        self.rm_num_samples = self.cfg.rm_num_samples
        
        self.batch_size = self.cfg.batch_size
        
        # Scene and related objects (will be initialized in setup_scene)
        self.scene = None
        self.tx = None
        self.camera = None
        self.rm = None  # Radio map
    
    def setup_scene(self):
        """Set up the scene with transmitter and receiver arrays.
        
        Returns:
            scene: Configured scene object
        """
        # Load scene
        self.scene = load_scene(sionna.rt.scene.munich)
        
        # Base station array
        self.scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=self.num_bs_ant // 2,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="cross"
        )
        
        # Base station (transmitter)
        self.tx = Transmitter(
            name="tx",
            position=[8.5, 21, 27],
            look_at=[45, 90, 1.5],
            display_radius=3.0
        )
        self.scene.add(self.tx)
        
        # Camera for visualization
        self.camera = Camera(
            position=[0, 80, 500],
            orientation=np.array([0, np.pi/2, -np.pi/2])
        )
        
        # UE arrays
        self.scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=self.num_ue_ant // 2,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="cross"
        )
        
        return self.scene
    
    def compute_radio_map(self, save_images=True):
        """Compute radio map and optionally save visualization.
        
        Args:
            save_images: If True, saves radio map images
            
        Returns:
            rm: Radio map object
        """
        if self.scene is None:
            self.setup_scene()
        
        # Compute radio map
        rm_solver = RadioMapSolver()
        self.rm = rm_solver(
            self.scene,
            max_depth=self.max_depth,
            cell_size=self.rm_cell_size,
            samples_per_tx=self.rm_samples_per_tx
        )
        
        if save_images:
            # Save radio map visualization
            self.scene.render_to_file(
                camera=self.camera,
                radio_map=self.rm,
                rm_vmin=self.rm_vmin_db,
                clip_at=self.rm_clip_at,
                resolution=list(self.rm_resolution),
                filename="munich_radio_map.png",
                num_samples=self.rm_num_samples
            )
        
        return self.rm
    
    def generate_cir_data(self, seed_offset=0, max_num_paths=0):
        """Generate CIR data for multiple UE positions.
        
        Args:
            seed_offset: Offset for random seed (used for generating multiple files)
            
        Returns:
            a: CIR coefficients array
            tau: Delay array
            max_num_paths: Maximum number of paths
        """
        if self.rm is None:
            self.compute_radio_map(save_images=False)
        
        # Sample initial UE positions
        ue_pos, _ = self.rm.sample_positions(
            num_pos=self.batch_size_cir,
            metric="path_gain",
            min_val_db=self.min_gain_db,
            max_val_db=self.max_gain_db,
            min_dist=self.min_dist,
            max_dist=self.max_dist
        )
        
        # Create receivers at sampled positions
        for i in range(self.batch_size_cir):
            p = ue_pos[0, i, :]
            if hasattr(p, "numpy"):
                p = p.numpy()
            p = np.asarray(p, dtype=np.float64)
            
            try:
                self.scene.remove(f"rx-{i}")
            except Exception:
                pass
            
            rx = Receiver(
                name=f"rx-{i}",
                position=(float(p[0]), float(p[1]), float(p[2])),
                velocity=(3.0, 3.0, 0.0),
                display_radius=1.0,
                color=(1, 0, 0)
            )
            self.scene.add(rx)
        
        # CIR generation
        p_solver = PathSolver()
        a_list, tau_list = [], []        
        num_runs = int(np.ceil(self.target_num_cirs / self.batch_size_cir))
        
        for idx in range(num_runs):
            print(f"Progress: {idx+1}/{num_runs}", end="\r", flush=True)
            
            # Sample new positions for each run
            ue_pos, _ = self.rm.sample_positions(
                num_pos=self.batch_size_cir,
                metric="path_gain",
                min_val_db=self.min_gain_db,
                max_val_db=self.max_gain_db,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                seed=idx + seed_offset * 1000  # Use seed_offset to vary between files
            )
            
            # Update receiver positions
            for rx in range(self.batch_size_cir):
                p = ue_pos[0, rx, :]
                if hasattr(p, "numpy"):
                    p = p.numpy()
                p = np.asarray(p, dtype=np.float64)
                self.scene.receivers[f"rx-{rx}"].position = (float(p[0]), float(p[1]), float(p[2]))
            
            # Compute paths
            paths = p_solver(
                self.scene,
                max_depth=self.max_depth,
                max_num_paths_per_src=10**7
            )
            
            # Get CIR
            a, tau = paths.cir(
                sampling_frequency=self.subcarrier_spacing,
                num_time_steps=self.num_time_steps,
                out_type="numpy"
            )
            a = a.astype(np.complex64)
            tau = tau.astype(np.float32)
            a_list.append(a)
            tau_list.append(tau)
            
            num_paths = a.shape[-2]
            max_num_paths = max(max_num_paths, num_paths)
        
        # Padding + stacking
        a, tau = [], []
        for a_, tau_ in zip(a_list, tau_list):
            num_paths = a_.shape[-2]
            a.append(
                np.pad(
                    a_,
                    [[0, 0], [0, 0], [0, 0], [0, 0],
                     [0, max_num_paths - num_paths], [0, 0]],
                    constant_values=0
                ).astype(np.complex64)
            )
            
            tau.append(
                np.pad(
                    tau_,
                    [[0, 0], [0, 0],
                     [0, max_num_paths - num_paths]],
                    constant_values=0
                ).astype(np.float32)
            )
        
        a = np.concatenate(a, axis=0)
        tau = np.concatenate(tau, axis=0)
        
        # Reorder dimensions
        a = np.transpose(a, (2, 3, 0, 1, 4, 5))
        tau = np.transpose(tau, (1, 0, 2))
        
        a = np.expand_dims(a, axis=0)
        tau = np.expand_dims(tau, axis=0)
        
        a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])
        tau = np.transpose(tau, [2, 1, 0, 3])
        
        # Remove empty CIRs
        p_link = np.sum(np.abs(a) ** 2, axis=(1, 2, 3, 4, 5, 6))
        a = a[p_link > 0, ...]
        tau = tau[p_link > 0, ...]
        
        print("(in cir.py) a.shape: ", a.shape)
        print("(in cir.py) tau.shape: ", tau.shape)
        
        return a, tau, max_num_paths
    
    def save_to_tfrecord(self, a, tau, filename):
        """Save CIR data to TFRecord file.
        
        Args:
            a: CIR coefficients array
            tau: Delay array
            filename: Output TFRecord filename
        """
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        def _int64_list_feature(value):
            """Returns an int64_list from a list or np.array of ints."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))
        
        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(len(a)):
                # Per-sample tensors
                a_sample = a[i]
                tau_sample = tau[i]
                
                # Serialize tensors
                a_bytes = tf.io.serialize_tensor(a_sample).numpy()
                tau_bytes = tf.io.serialize_tensor(tau_sample).numpy()
                
                # Shape metadata (per sample)
                a_shape = a_sample.shape
                tau_shape = tau_sample.shape
                
                # Create feature dictionary with data + shape metadata
                feature = {
                    "a": _bytes_feature(a_bytes),
                    "tau": _bytes_feature(tau_bytes),
                    "a_shape": _int64_list_feature(a_shape),
                    "tau_shape": _int64_list_feature(tau_shape),
                }
                
                # Create Example message
                features = tf.train.Features(feature=feature)
                example = tf.train.Example(features=features)
                
                # Write to file
                writer.write(example.SerializeToString())
        
        print(f"  Saved {len(a)} samples to {filename}")
    
    def load_from_tfrecord(self, tfrecord_dir="../cir_tfrecords", group_for_mumimo=False):
        """Load CIR data from TFRecord files.
        
        Args:
            tfrecord_dir: Directory containing TFRecord files
            
        Returns:
            all_a: Concatenated CIR coefficients with shape [num_samples, 1, num_rx_ant, num_ue, num_tx_ant, num_paths, num_time_steps]
            all_tau: Concatenated delay values with shape [num_samples, 1, num_ue, num_paths]
        """
        cir_dir = os.path.join(os.path.dirname(__file__), tfrecord_dir)
        cir_files = tf.io.gfile.glob(os.path.join(cir_dir, "*.tfrecord"))
        
        if not cir_files:
            raise ValueError(f"No TFRecord files found in {cir_dir}")
        
        feature_description = {
            "a": tf.io.FixedLenFeature([], tf.string),
            "tau": tf.io.FixedLenFeature([], tf.string),
            "a_shape": tf.io.VarLenFeature(tf.int64),
            "tau_shape": tf.io.VarLenFeature(tf.int64),
        }
        
        def _parse_example(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            
            # Deserialize tensors
            a = tf.io.parse_tensor(parsed["a"], out_type=tf.complex64)
            tau = tf.io.parse_tensor(parsed["tau"], out_type=tf.float32)
            
            # Read shape metadata (sparse -> dense)
            a_shape = tf.sparse.to_dense(parsed["a_shape"])
            tau_shape = tf.sparse.to_dense(parsed["tau_shape"])
            
            # Ensure correct shapes
            a = tf.reshape(a, a_shape)
            tau = tf.reshape(tau, tau_shape)
            
            return a, tau
        
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
            # Group num_ue individual CIRs into MU-MIMO samples
            num_ue = self.num_ue  # 4
            num_samples = tf.shape(all_a)[0]
            num_mu_samples = num_samples // num_ue
            
            # Truncate to multiple of num_ue
            all_a = all_a[:num_mu_samples * num_ue]
            all_tau = all_tau[:num_mu_samples * num_ue]
            
            # a: [N*4, 1, 16, 1, 4, 51, 14] -> [N, 1, 16, 4, 4, 51, 14]
            all_a = tf.reshape(all_a, [num_mu_samples, num_ue, 1, 16, 1, 4, 51, 14])
            all_a = tf.squeeze(all_a, axis=4)  # [N, 4, 1, 16, 4, 51, 14]
            all_a = tf.transpose(all_a, [0, 2, 3, 1, 4, 5, 6])  # [N, 1, 16, 4, 4, 51, 14]
            
            # tau: [N*4, 1, 1, 51] -> [N, 1, 4, 51]
            all_tau = tf.reshape(all_tau, [num_mu_samples, num_ue, 1, 1, 51])
            all_tau = tf.squeeze(all_tau, axis=3)  # [N, 4, 1, 51]
            all_tau = tf.transpose(all_tau, [0, 2, 1, 3])  # [N, 1, 4, 51]
        
        return all_a, all_tau
    
    def build_channel_model(self, batch_size=None, num_bs=None, num_bs_ant=None, 
                           num_ue=None, num_ue_ant=None, num_time_steps=None,
                           tfrecord_dir="../cir_tfrecords"):
        """Build channel model from TFRecord files.
        
        Args:
            batch_size: Batch size for the dataset (default: from config)
            num_bs: Number of base stations (default: from config)
            num_bs_ant: Number of BS antennas (default: from config)
            num_ue: Number of UEs (default: from config)
            num_ue_ant: Number of UE antennas (default: from config)
            num_time_steps: Number of time steps (default: from config)
            tfrecord_dir: Directory containing TFRecord files
            
        Returns:
            channel_model: CIRDataset object
        """
        # Use config values as defaults
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_bs = num_bs if num_bs is not None else self.num_bs
        num_bs_ant = num_bs_ant if num_bs_ant is not None else self.num_bs_ant
        num_ue = num_ue if num_ue is not None else self.num_ue
        num_ue_ant = num_ue_ant if num_ue_ant is not None else self.num_ue_ant
        num_time_steps = num_time_steps if num_time_steps is not None else self.num_time_steps
        
        # Load CIR data from TFRecord files
        all_a, all_tau = self.load_from_tfrecord(tfrecord_dir)
        max_num_paths = all_a.shape[-2]
        
        # Create CIR generator
        cir_generator = CIRGenerator(all_a, all_tau, num_ue)
        
        # Create channel model
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
        """Render and save the radio map with the current UE positions overlaid."""
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
        tfrecord_dir="../cir_tfrecords",
        save_radio_map=True,
    ):
        """
        Generate and save CIR data to TFRecord files.

        Args:
            seed_offsets : int or list[int]
                - If int - treat as a single seed offset.
                - If list/tuple - generate one file per seed.
            tfrecord_dir : str
                Directory (relative to this file) where TFRecord files will be saved.
            save_radio_map : bool
                If True, saves radio map and UE-position visualizations.
        """
        # Normalize input to a list
        if isinstance(seed_offsets, (int, np.integer)):
            seed_list = [int(seed_offsets)]
        elif isinstance(seed_offsets, (list, tuple, np.ndarray)):
            seed_list = [int(s) for s in seed_offsets]
        else:
            raise ValueError("seed_offsets must be an int or a list/tuple of ints")

        # Prepare scene and radio map
        self.setup_scene()
        self.compute_radio_map(save_images=save_radio_map)

        # Save UE positions visualization (once, after receivers are created)
        # Note: receivers will be created inside generate_cir_data on the first call.
        # We’ll call this after that first call so the image includes UEs.
        # To keep behavior simple, we’ll just mark that we still need to save it.
        need_ue_viz = save_radio_map

        # Make output directory relative to this file, consistent with load_from_tfrecord
        base_dir = os.path.dirname(__file__)
        cir_dir = os.path.join(base_dir, tfrecord_dir)
        os.makedirs(cir_dir, exist_ok=True)

        # Track the maximum number of paths across all files
        max_num_paths_all = 0

        # Generate a file per seed
        for idx, seed in enumerate(seed_list):
            print(f"\nGenerating CIR file {idx+1}/{len(seed_list)}  (seed_offset={seed})")

            # Generate CIR data
            a, tau, max_num_paths = self.generate_cir_data(seed_offset=seed)

            max_num_paths_all = max(max_num_paths_all, max_num_paths)

            print(f"  a.shape={a.shape}, tau.shape={tau.shape}")
            print(f"  max_num_paths={max_num_paths}")

            # Save UE-position visualization once, after first CIR generation
            if need_ue_viz:
                ue_fig = os.path.join(cir_dir, f"munich_ue_positions_seed_{seed:03d}.png")
                try:
                    self.save_visualization_ue_positions(filename=ue_fig)
                    print(f"  Saved UE position visualization to '{ue_fig}'")
                except Exception as e:
                    print(f"  Warning: failed to save UE visualization: {e}")
                need_ue_viz = False

            # File name uses the seed value for clarity
            filename = os.path.join(cir_dir, f"cir_seed_{seed:03d}.tfrecord")

            self.save_to_tfrecord(a, tau, filename)

        print(f"\nSuccessfully generated {len(seed_list)} TFRecord files.")
        print(f"All files saved in '{cir_dir}' directory.")
        print(f"Global max_num_paths across all seeds = {max_num_paths_all}")


if __name__ == "__main__":
    print("\n CIR Generation Started")
    try:
        cir_manager = CIRManager()
        cir_manager.generate_and_save([0])
        print("\n CIR Generation Completed Successfully \n")
    except Exception as e:
        print("\n!!! CIR Generation Failed !!!")
        print(f"Error: {e}\n")
        raise
