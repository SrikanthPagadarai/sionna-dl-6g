# sionna-dl-6g

Deep learning demos for B5G/6G wireless systems using TensorFlow and [Sionna](https://nvlabs.github.io/sionna/).

## Overview

This repository contains implementations of neural network-based systems for 5G/6G OFDM systems:

- **mimo_ofdm_neural_receiver** — Neural receiver for a MIMO-OFDM system
- **pusch_autoencoder** — Site-specific autoencoder design for 5G PUSCH

## Installation

Requires Python 3.10–3.12.

```bash
pip install sionna-dl-6g
```

Or install from source:

```bash
git clone https://github.com/SrikanthPagadarai/sionna-dl-6g.git
cd sionna-dl-6g
pip install .
```

## Quick Start


## Project Structure

```
sionna-dl-6g/
├── mimo_ofdm_neural_receiver/
│   ├── src/
│   │   ├── config.py      # System configuration
│   │   ├── tx.py          # Transmitter chain
│   │   ├── rx.py          # Receiver chain
│   │   ├── channel.py     # Channel model
│   │   ├── csi.py         # CSI management
│   │   └── neural_rx.py   # Neural receiver
│   ├── training.py
│   ├── inference.py
│   └── tests/
├── pusch_autoencoder/
│   ├── src/
│   │   ├── config.py
│   │   ├── pusch_trainable_transmitter.py
│   │   ├── pusch_trainable_receiver.py
│   │   └── pusch_neural_detector.py
│   ├── training.py
│   └── inference.py
└── pyproject.toml
```

## Configuration

Key parameters in `Config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_bits_per_symbol` | `QPSK` | Modulation: QPSK (2), QAM16 (4) |
| `perfect_csi` | `False` | Use perfect channel knowledge |
| `cdl_model` | `"D"` | CDL channel model (A, B, C, D, E) |
| `delay_spread` | `300e-9` | Channel delay spread in seconds |
| `carrier_frequency` | `2.6e9` | Carrier frequency in Hz |
| `speed` | `0.0` | UE speed in m/s |

## Development

### Setup

```bash
git clone https://github.com/SrikanthPagadarai/sionna-dl-6g.git
cd sionna-dl-6g
poetry install
poetry run pre-commit install
```

### Run Tests

```bash
poetry run pytest tests/ -v
```

### Code Formatting

```bash
poetry run black .
poetry run flake8 .
```

## Requirements

- Python 3.10–3.12
- TensorFlow 2.x
- Sionna ≥0.15.0
- CUDA (optional, for GPU acceleration)

## License

MIT

## References

- [Sionna: An Open-Source Library for Next-Generation Physical Layer Research](https://nvlabs.github.io/sionna/)
- 3GPP TS 38.211: NR Physical channels and modulation
- 3GPP TS 38.212: NR Multiplexing and channel coding