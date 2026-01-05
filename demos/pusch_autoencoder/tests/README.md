# PUSCH Autoencoder Tests

This directory contains comprehensive test suites for the PUSCH autoencoder demo components.

## Test Files

### test_config.py
Tests for the `Config` class:
- Configuration initialization and default values
- MCS (Modulation and Coding Scheme) decoding
- PUSCH pilot indices setup
- Property getters and setters

### test_pusch_trainable_transmitter.py
Tests for the `PUSCHTrainableTransmitter` class:
- Initialization in training and inference modes
- Forward pass with different configurations
- Constellation normalization (unit power constraint)
- Trainable variable management
- Constellation updates during forward pass

### test_pusch_neural_detector.py
Tests for the `PUSCHNeuralDetector` class:
- Conv2D residual block functionality
- Neural detector initialization
- Forward pass with dummy inputs
- Custom constellation handling
- Trainable variables verification

### test_pusch_trainable_receiver.py
Tests for the `PUSCHTrainableReceiver` class:
- Initialization with different CSI modes (perfect/imperfect)
- Forward pass in training and inference modes
- Constellation retrieval from transmitter
- Trainable variable exposure
- Integration with neural detector

### test_system.py
Tests for the end-to-end `PUSCHLinkE2E` system:
- System initialization with different configurations
- Inference mode (returns bits and decoded bits)
- Training mode (returns loss for autoencoder)
- Trainable variables for baseline vs autoencoder
- Constellation properties (unit power, minimum distance)
- Performance at different SNR levels
- Batch size variations

### test_cir_generator.py
Tests for the `CIRGenerator` class:
- Generator initialization
- Single and multiple sample generation
- Randomness verification
- Integration with tf.data.Dataset
- Data type preservation
- Variable num_tx configurations

### test_cir_manager.py
Tests for the `CIRManager` class:
- Initialization with default and custom configs
- Solver and radio map parameter setup
- Scene attribute initialization
- TFRecord save and load functionality
- Channel model building
- Batching support

## Running the Tests

To run all tests:
```bash
pytest demos/pusch_autoencoder/tests/
```

To run a specific test file:
```bash
pytest demos/pusch_autoencoder/tests/test_system.py
```

To run with verbose output:
```bash
pytest demos/pusch_autoencoder/tests/ -v
```

To run with printed output (see print statements):
```bash
pytest demos/pusch_autoencoder/tests/ -s
```