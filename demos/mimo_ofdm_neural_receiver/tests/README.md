# MIMO OFDM Neural Receiver Tests

This directory contains comprehensive test suites for the MIMO OFDM neural receiver demo components.

## Test Files

### test_tx.py
Tests for the `Tx` OFDM transmitter class:
- Transmitter initialization with Config
- Forward pass output shapes for different modulation schemes (QPSK, 16-QAM)
- Information bits, coded bits, symbols, and resource grid dimensions
- Batch processing verification

### test_channel.py
Tests for the `Channel` class:
- Channel forward pass for different modulation schemes
- Received signal shape verification
- Integration with CSI (Channel State Information)
- Noise power handling

### test_rx.py
Tests for the `Rx` OFDM receiver class:
- Receiver initialization with different CSI modes (perfect/imperfect)
- Channel estimation output shapes
- Equalized symbols and effective noise dimensions
- LLR (Log-Likelihood Ratio) computation
- Decoded bits verification

### test_neural_rx.py
Tests for the `NeuralRx` neural network receiver class:
- Neural receiver initialization with different configurations
- Forward pass for various modulation schemes and CSI modes
- LLR output shape verification
- Decoded bits shape verification
- Integration with CSI subsystem

### test_system.py
Tests for the end-to-end `System` class:
- System initialization with different configurations
- Inference mode (returns transmitted and decoded bits)
- Training mode (returns loss for gradient-based optimization)
- BER computation verification
- Different CDL channel models
- Neural receiver vs baseline receiver comparison

## Running the Tests

To run all MIMO OFDM neural receiver tests:
```bash
pytest demos/mimo_ofdm_neural_receiver/tests/
```

To run a specific test file:
```bash
pytest demos/mimo_ofdm_neural_receiver/tests/test_system.py
```

To run with verbose output:
```bash
pytest demos/mimo_ofdm_neural_receiver/tests/ -v
```

To run with printed output (see print statements):
```bash
pytest demos/mimo_ofdm_neural_receiver/tests/ -s
```

To run a specific test function:
```bash
pytest demos/mimo_ofdm_neural_receiver/tests/test_system.py::test_system_inference -v
```
