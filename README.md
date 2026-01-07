# [Work-In-Progress] nextgen-wireless-dl-demos

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning demos for 5G/6G wireless systems using TensorFlow and [Sionna](https://nvlabs.github.io/sionna/).

> **Disclaimer:** This is an independent project and is not affiliated with, endorsed by, or sponsored by NVIDIA Corporation. [Sionna](https://nvlabs.github.io/sionna/) is an open-source library developed by NVIDIA.

⚠️ **Note:** This project is under active development and not accepting external contributions at this time.

## Author and Maintainer

**Srikanth Pagadarai <srikanth.pagadarai@gmail.com>**

## Overview

This repository contains neural network-based demos for 5G/6G communication systems:

| Demo | Description |
|------|-------------|
| **Digital Pre-Distortion** | Digital Pre-Distortion for power amplifier linearization |
| **MIMO-OFDM Neural Receiver** | Neural receiver for MIMO-OFDM systems with learned channel estimation and equalization |
| **Site-Specific PUSCH Autoencoder** | End-to-end autoencoder for 5G NR PUSCH with trainable constellation and neural detector |

## Project Structure

```
nextgen-wireless-dl-demos/
├── .github/
│   └── workflows/
│       ├── docs.yml                      # Documentation build workflow
│       ├── publish.yml                   # PyPI publish workflow
│       ├── test-publish.yml              # Test PyPI publish workflow
│       └── test.yml                      # CI test workflow
├── .dockerignore                         # Docker build exclusions
├── .flake8                               # Flake8 linter configuration
├── .gitignore                            # Git ignore rules
├── .gitmodules                           # Git submodule definitions
├── .pre-commit-config.yaml               # Pre-commit hooks configuration
│
├── demos/
│   ├── dpd/                              # Digital Pre-Distortion demo
│   │   ├── results/                      # Performance results
│   │   ├── src/
│   │   │   ├── config.py                 # System configuration
│   │   │   ├── tx.py                     # OFDM transmitter
│   │   │   ├── rx.py                     # OFDM receiver
│   │   │   ├── power_amplifier.py        # PA model with memory effects
│   │   │   ├── interpolator.py           # Sample rate conversion
│   │   │   ├── ls_dpd.py                 # Least-squares DPD
│   │   │   ├── ls_dpd_system.py          # LS-DPD end-to-end system
│   │   │   ├── nn_dpd.py                 # Neural network DPD
│   │   │   ├── nn_dpd_system.py          # NN-DPD end-to-end system
│   │   │   └── system.py                 # Base system class
│   │   ├── tests/                        # Unit tests
│   │   ├── training_ls.py                # LS-DPD training
│   │   ├── training_nn.py                # NN-DPD training
│   │   ├── inference.py                  # Model evaluation
│   │   ├── plots_ls.py                   # LS-DPD visualization
│   │   └── plots_nn.py                   # NN-DPD visualization
│   │
│   ├── mimo_ofdm_neural_receiver/        # Neural MIMO-OFDM receiver demo
│   │   ├── results/                      # Performance results
│   │   ├── src/
│   │   │   ├── config.py                 # System configuration
│   │   │   ├── tx.py                     # Transmitter chain
│   │   │   ├── rx.py                     # Baseline receiver
│   │   │   ├── channel.py                # CDL channel model
│   │   │   ├── csi.py                    # Channel state information
│   │   │   ├── neural_rx.py              # Neural receiver network
│   │   │   └── system.py                 # End-to-end system
│   │   ├── tests/                        # Unit tests
│   │   ├── training.py                   # Neural receiver training
│   │   ├── inference.py                  # Trained model evaluation
│   │   ├── baseline.py                   # Baseline receiver evaluation
│   │   └── plots.py                      # BER/BLER visualization
│   │
│   └── pusch_autoencoder/                # PUSCH autoencoder demo
│       ├── results/                      # Performance results
│       ├── src/
│       │   ├── config.py                 # System configuration
│       │   ├── pusch_trainable_transmitter.py  # Trainable constellation TX
│       │   ├── pusch_trainable_receiver.py     # Neural receiver
│       │   ├── pusch_neural_detector.py        # Conv2D-based detector
│       │   ├── cir_generator.py          # Channel impulse response generator
│       │   ├── cir_manager.py            # CIR dataset management
│       │   └── system.py                 # End-to-end PUSCH link
│       ├── tests/                        # Unit tests
│       ├── training.py                   # Autoencoder training
│       ├── inference.py                  # Trained model evaluation
│       ├── baseline.py                   # LMMSE baseline evaluation
│       └── plots.py                      # BLER and constellation plots
│
├── docs/                                 # Sphinx documentation
│   ├── api/                              # API reference
│   ├── demos/                            # Demo documentation
│   ├── conf.py                           # Sphinx configuration
│   └── *.rst                             # Documentation pages
│
├── docker/                               # Docker configuration
│   ├── docker-instructions.md            # Docker usage guide
│   └── entrypoint.sh                     # Container entrypoint
│
├── gcp-management/                       # GCP infrastructure (git submodule)
│
├── Dockerfile                            # Docker image definition
├── host_nvidia_runtime_setup.sh          # NVIDIA runtime setup script
├── pyproject.toml                        # Project configuration
├── poetry.lock                           # Dependency lock file
└── LICENSE                               # MIT license
```

## Installation

Requires Python 3.10–3.12.

```bash
pip install nextgen-wireless-dl-demos
```

Or install from source:

```bash
git clone https://github.com/SrikanthPagadarai/nextgen-wireless-dl-demos.git
cd nextgen-wireless-dl-demos
pip install .
```

## Quick Start

### DPD Demo

```bash
# Train Neural Network DPD
python demos/dpd/training_nn.py --iterations 10000

# Run inference
python demos/dpd/inference.py --dpd_method nn

# Generate plots
python demos/dpd/plots_nn.py
```

### MIMO OFDM Neural Receiver Demo

```bash
# Train neural receiver
python demos/mimo_ofdm_neural_receiver/training.py --iterations 10000

# Run inference
python demos/mimo_ofdm_neural_receiver/inference.py

# Generate comparison plots
python demos/mimo_ofdm_neural_receiver/plots.py
```

### PUSCH Autoencoder Demo

```bash
# Train autoencoder (16 BS antennas, default)
python demos/pusch_autoencoder/training.py

# Train autoencoder (32 BS antennas)
python demos/pusch_autoencoder/training.py --num_bs_ant 32

# Run inference
python demos/pusch_autoencoder/inference.py

# Generate plots
python demos/pusch_autoencoder/plots.py
```

## Development

### Setup

```bash
git clone https://github.com/SrikanthPagadarai/nextgen-wireless-dl-demos.git
cd nextgen-wireless-dl-demos
poetry install
poetry run pre-commit install
```

### Run Tests

```bash
# Run all tests
poetry run pytest demos/ -v

# Run tests for a specific demo
poetry run pytest demos/dpd/tests/ -v
poetry run pytest demos/mimo_ofdm_neural_receiver/tests/ -v
poetry run pytest demos/pusch_autoencoder/tests/ -v
```

### Code Formatting

```bash
poetry run black .
poetry run flake8 .
```

### Docker

```bash
# Build image
docker build -t nextgen-wireless-dl-demos .

# Run container with GPU support
docker run --gpus all -it nextgen-wireless-dl-demos
```

## Requirements

- Python 3.10–3.12
- TensorFlow 2.x
- Sionna ≥0.19.0
- CUDA (optional, for GPU acceleration)

## License

MIT

## References

- [Sionna: An Open-Source Library for Next-Generation Physical Layer Research](https://nvlabs.github.io/sionna/)
- 3GPP TS 38.211: NR Physical channels and modulation
- 3GPP TS 38.212: NR Multiplexing and channel coding