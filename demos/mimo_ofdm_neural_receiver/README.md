# MIMO OFDM Neural Receiver Demo

This demo implements a neural network-based receiver for MIMO-OFDM wireless communication systems. The neural receiver learns to perform joint channel estimation and equalization, outperforming traditional baseline receivers.

## Overview

The neural receiver replaces traditional OFDM receiver components (channel estimation, equalization, demapping) with a learned neural network that directly maps received signals to LLRs (Log-Likelihood Ratios) for channel decoding.

**Key features:**
- CDL (Clustered Delay Line) channel models
- Support for QPSK and 16-QAM modulation
- Perfect and imperfect CSI modes
- LDPC channel coding

## Training

Train the neural receiver using gradient-based optimization:

```bash
python demos/mimo_ofdm_neural_receiver/training.py
```

**Options:**
- `--iterations`: Number of training iterations (default: 10000)
- `--fresh`: Start fresh training, ignoring any existing checkpoint

**Example:**
```bash
python demos/mimo_ofdm_neural_receiver/training.py --iterations 20000
python demos/mimo_ofdm_neural_receiver/training.py --iterations 5000 --fresh
```

**Training details:**
- Batch size: 32
- Eb/N0 range: -3 dB to 7 dB (randomly sampled)
- Gradient accumulation: 4 steps
- Optimizer: Adam

**Outputs:**
- `results/mimo-ofdm-neuralrx-weights` - Trained model weights
- `results/loss.npy` - Training loss history
- `checkpoints/` - Training checkpoints for resumption

## Baseline Evaluation

Evaluate the traditional baseline receiver (LS channel estimation + LMMSE equalization):

```bash
python demos/mimo_ofdm_neural_receiver/baseline.py
```

This runs BER/BLER simulations for:
- Perfect CSI (known channel)
- Imperfect CSI (estimated channel)

**Outputs:**
- `results/all_baseline_results_cdlC.npz` - Baseline BER/BLER results

**Note:** Pre-computed baseline results are included in `results/all_baseline_results_cdlC.npz`.

## Inference

Run inference with the trained neural receiver:

```bash
python demos/mimo_ofdm_neural_receiver/inference.py
```

**Prerequisites:**
- Trained weights must exist at `results/mimo-ofdm-neuralrx-weights`

**Configuration:**
- Batch size: 32
- Eb/N0 range: -3 dB to 7 dB

**Outputs:**
- `results/inference_results.npz` - Neural receiver BER/BLER results

## Generating Plots

Generate comparison plots after running baseline and inference:

```bash
python demos/mimo_ofdm_neural_receiver/plots.py
```

**Prerequisites:**
- `results/all_baseline_results_cdlC.npz` (baseline results)
- `results/inference_results.npz` (neural receiver results)
- `results/loss.npy` (training loss, optional)

**Generated plots:**
- `results/loss.png` - Training loss curve
- `results/ber_cdlC.png` - BER comparison (baseline vs neural receiver)
- `results/bler_cdlC.png` - BLER comparison (baseline vs neural receiver)

## Complete Workflow Example

```bash
# 1. (Optional) Run baseline evaluation
python demos/mimo_ofdm_neural_receiver/baseline.py

# 2. Train neural receiver
python demos/mimo_ofdm_neural_receiver/training.py --iterations 10000

# 3. Run inference with trained model
python demos/mimo_ofdm_neural_receiver/inference.py

# 4. Generate comparison plots
python demos/mimo_ofdm_neural_receiver/plots.py
```

## Running Tests

```bash
pytest demos/mimo_ofdm_neural_receiver/tests/ -v
```

## Channel Configuration

The demo uses CDL-C/D channel models with configurable parameters:
- Delay spread: 100-300 ns
- Carrier frequency: 2.6 GHz
- User speed: 0 m/s (static)

These can be modified in `baseline.py` or by adjusting the `System` class parameters.
