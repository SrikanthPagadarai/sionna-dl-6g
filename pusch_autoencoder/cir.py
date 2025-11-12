import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import or install Sionna
try:
    import sionna.phy
    import sionna.rt
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np

# For link-level simulations
from sionna.phy.channel import CIRDataset

# Import Sionna RT components
from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver

import os
import matplotlib.pyplot as plt

# ----------------------------
# New: central configuration
# ----------------------------
from config import Config
_cfg = Config()

# system parameters (read from config to avoid functionality changes)
subcarrier_spacing = _cfg.SUBCARRIER_SPACING
num_time_steps = _cfg.NUM_TIME_STEPS

num_ue = _cfg.NUM_UE
num_bs = _cfg.NUM_BS
num_ue_ant = _cfg.NUM_UE_ANT
num_bs_ant = _cfg.NUM_BS_ANT

batch_size_cir = _cfg.BATCH_SIZE_CIR

# radio-map / solver knobs
max_depth = _cfg.MAX_DEPTH

# sampling window
min_gain_db = _cfg.MIN_GAIN_DB
max_gain_db = _cfg.MAX_GAIN_DB
min_dist = _cfg.MIN_DIST
max_dist = _cfg.MAX_DIST

no_preview = False # unchanged toggle

# load an integrated scene
scene = load_scene(sionna.rt.scene.munich)

# bs has an antenna pattern from 3GPP 38.901
scene.tx_array = PlanarArray(num_rows=1, 
                             num_cols=num_bs_ant//2,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="cross")

# instantiate tx (BS)
tx = Transmitter(name="tx",position=[8.5,21,27],look_at=[45,90,1.5],display_radius=3.0)
scene.add(tx)

# create new camera
camera = Camera(position=[0,80,500],orientation=np.array([0,np.pi/2,-np.pi/2]))

# compute radio map for the instantiated tx (BS)
rm_solver = RadioMapSolver() # radio-map solver
rm = rm_solver(scene,
               max_depth=max_depth,
               cell_size=_cfg.RM_CELL_SIZE,
               samples_per_tx=_cfg.RM_SAMPLES_PER_TX)

scene.render_to_file(camera=camera,
                     radio_map=rm,
                     rm_vmin=_cfg.RM_VMIN_DB,
                     clip_at=_cfg.RM_CLIP_AT,
                     resolution=list(_cfg.RM_RESOLUTION),
                     filename="munich_radio_map.png",
                     num_samples=_cfg.RM_NUM_SAMPLES)

# sample random user positions from radio map

# sample batch-size random user positions from the radio-map
ue_pos, _ = rm.sample_positions(num_pos=batch_size_cir,
                                metric="path_gain",
                                min_val_db=min_gain_db,
                                max_val_db=max_gain_db,
                                min_dist=min_dist,
                                max_dist=max_dist)

# add UEs at the sampled positions
scene.rx_array = PlanarArray(num_rows=1, 
                             num_cols=num_ue_ant//2,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="cross")

# create batch_size receivers
for i in range(batch_size_cir):
    p = ue_pos[0, i, :]

    if hasattr(p, "numpy"):
        p = p.numpy()
    p = np.asarray(p, dtype=np.float64).reshape(3,)
    px, py, pz = float(p[0]), float(p[1]), float(p[2])

    # remove any existing receiver with same name
    try:
        scene.remove(f"rx-{i}")
    except Exception:
        pass

    rx = Receiver(
        name=f"rx-{i}",
        position=(px, py, pz),
        velocity=(3.0, 3.0, 0.0),
        display_radius=1.0,
        color=(1, 0, 0),
    )
    scene.add(rx)

scene.render_to_file(camera=camera,
                     radio_map=rm,
                     rm_vmin=_cfg.RM_VMIN_DB,
                     clip_at=_cfg.RM_CLIP_AT,
                     resolution=list(_cfg.RM_RESOLUTION),
                     filename="munich_radio_map_with_UEs.png",
                     num_samples=_cfg.RM_NUM_SAMPLES)

# create CIR dataset
target_num_cirs = _cfg.TARGET_NUM_CIRS  # unchanged value

# channel impulse responses
a_list, tau_list = [], []
max_num_paths = 0
p_solver = PathSolver()
num_runs = int(np.ceil(target_num_cirs/batch_size_cir))
for idx in range(num_runs):
    print(f"Progress: {idx+1}/{num_runs}", end="\r", flush=True)

    # sample random user positions from the radio-map
    ue_pos, _ = rm.sample_positions(num_pos=batch_size_cir,
                                    metric="path_gain",
                                    min_val_db=min_gain_db,
                                    max_val_db=max_gain_db,
                                    min_dist=min_dist,
                                    max_dist=max_dist,
                                    seed=idx)
    
    # update all receiver positions
    for rx in range(batch_size_cir):
        p = ue_pos[0, rx, :]

        if hasattr(p, "numpy"):
            p = p.numpy()
        p = np.asarray(p, dtype=np.float64).reshape(3,)
        px, py, pz = float(p[0]), float(p[1]), float(p[2])

        scene.receivers[f"rx-{rx}"].position = (px, py, pz)

    # simulate CIR
    paths = p_solver(scene, max_depth=max_depth, max_num_paths_per_src=10000)

    # from paths to CIRs
    a, tau = paths.cir(sampling_frequency=subcarrier_spacing, num_time_steps=14, out_type="numpy")
    a = a.astype(np.complex64, copy=False)
    tau = tau.astype(np.float32, copy=False)
    a_list.append(a)
    tau_list.append(tau)

    # update max number of paths over all batches of CIRs
    num_paths = a.shape[-2]
    if num_paths > max_num_paths:
        max_num_paths = num_paths

# concatenate all CIRs into a single tensor
a, tau = [], []
printed = False
for a_, tau_ in zip(a_list, tau_list):
    if not printed:
        print("a_ shape:", a_.shape)
        print("tau_ shape:", tau_.shape)
        printed = True
    num_paths = a_.shape[-2]
    a.append(np.pad(a_, [[0,0],[0,0],[0,0],[0,0],[0,max_num_paths-num_paths],[0,0]],
                constant_values=0).astype(np.complex64, copy=False))
    tau.append(np.pad(tau_, [[0,0],[0,0],[0,max_num_paths-num_paths]],
                  constant_values=0).astype(np.float32, copy=False))
a = np.concatenate(a, axis=0) # Concatenate along the num_rx dimension
tau = np.concatenate(tau, axis=0)
print('1)')
print('a.shape: ', a.shape)
print('tau.shape: ', tau.shape)

# reverse direction
a = np.transpose(a, (2,3,0,1,4,5))
tau = np.transpose(tau, (1,0,2))
print('2)')
print('a.shape: ', a.shape)
print('tau.shape: ', tau.shape)

# add a batch-size dimension
a = np.expand_dims(a, axis=0)
tau = np.expand_dims(tau, axis=0)
print('3)')
print('a.shape: ', a.shape)
print('tau.shape: ', tau.shape)

# switch batch-size and num_gggggggg dimensions
a = np.transpose(a,[3,1,2,0,4,5,6])
tau = np.transpose(tau,[2,1,0,3])
print('4)')
print('a.shape: ', a.shape)
print('tau.shape: ', tau.shape)

# remove CIRs that have no active link (i.e., a is all-zero)
p_link = np.sum(np.abs(a)**2, axis=(1,2,3,4,5,6))
a = a[p_link>0,...]
tau = tau[p_link>0,...]

print('5)')
print('a.shape: ', a.shape)
print('tau.shape: ', tau.shape)

class CIRGenerator:
    """(Unchanged)"""

    def __init__(self,
                 a,
                 tau,
                 num_tx):

        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]

        self._num_tx = num_tx

    def __call__(self):

        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample 4 random users and stack them together
            idx,_,_ = tf.random.uniform_candidate_sampler(
                            tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                            num_true=self._dataset_size,
                            num_sampled=self._num_tx,
                            unique=True,
                            range_max=self._dataset_size)

            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose to remove batch dimension
            a = tf.transpose(a, (3,1,2,0,4,5,6))
            tau = tf.transpose(tau, (2,1,0,3))

            # And remove batch-dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)

            yield a, tau


batch_size = _cfg.BATCH_SIZE  # Must equal CIRDataset batch size for BER sims

# Init CIR generator
cir_generator = CIRGenerator(a, tau, num_ue)
# Initialises a channel model that can be directly used by OFDMChannel layer
channel_model = CIRDataset(cir_generator, batch_size, num_bs, num_bs_ant,
                           num_ue, num_ue_ant, max_num_paths, num_time_steps)

