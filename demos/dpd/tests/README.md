# DPD (Digital Pre-Distortion) Tests

This directory contains comprehensive test suites for the DPD demo components.

## Test Files

### test_config.py
Tests for the `Config` dataclass:
- Configuration initialization and default values
- Immutable system parameters
- Resource grid parameters
- Derived properties (e.g., signal_sample_rate)
- Property getters verification

### test_tx.py
Tests for the `Tx` OFDM transmitter class:
- Transmitter initialization with Config
- Forward pass output shapes
- Resource grid creation
- LDPC encoder dimensions
- Time-domain signal generation

### test_power_amplifier.py
Tests for the `PowerAmplifier` class:
- PA initialization with default coefficients
- Forward pass with 1D and 2D (batched) inputs
- Memory polynomial model verification
- Gain estimation functionality
- Basis matrix construction

### test_interpolator.py
Tests for the `Interpolator` class:
- Initialization with different rate ratios
- Upsampling and downsampling factors
- Filter design parameters
- Forward pass output shape and rate
- Batched signal processing

### test_rx.py
Tests for the `Rx` OFDM receiver class:
- Receiver initialization
- OFDM demodulation
- Per-subcarrier equalization
- EVM computation
- Time synchronization

### test_ls_dpd.py
Tests for the `LeastSquaresDPD` class:
- Initialization with default and custom parameters
- Coefficient management (get/set)
- GMP basis matrix construction
- Predistortion forward pass
- Batched signal processing
- LS estimation

### test_nn_dpd.py
Tests for the `NeuralNetworkDPD` and `ResidualBlock` classes:
- ResidualBlock initialization and forward pass
- NN-DPD initialization with custom parameters
- Sliding window feature extraction
- Skip connection behavior
- Forward pass with 1D and 2D inputs
- Trainable variables verification

### test_ls_dpd_system.py
Tests for the `LS_DPDSystem` class:
- System initialization in training and inference modes
- Signal generation
- PA gain estimation
- Inference forward pass
- LS learning iterations
- Full LS learning procedure

### test_nn_dpd_system.py
Tests for the `NN_DPDSystem` class:
- System initialization in training and inference modes
- Training forward pass (returns loss)
- Inference forward pass
- Signal normalization
- Trainable variables for gradient-based optimization

## Running the Tests

To run all DPD tests:
```bash
pytest demos/dpd/tests/
```

To run a specific test file:
```bash
pytest demos/dpd/tests/test_power_amplifier.py
```

To run with verbose output:
```bash
pytest demos/dpd/tests/ -v
```

To run with printed output (see print statements):
```bash
pytest demos/dpd/tests/ -s
```

To run a specific test function:
```bash
pytest demos/dpd/tests/test_config.py::test_config_initialization -v
```
