# Digital Pre-Distortion (DPD) Demo

This demo implements Digital Pre-Distortion techniques for power amplifier (PA) linearization in wireless communication systems which use a 5G-like time-frequency resource grid. It supports two DPD methods: Neural Network (NN) DPD and Least-Squares (LS) DPD. In both methods, the DPD parameters are identified in an indirect learning architecture.

## Overview

Digital Pre-Distortion compensates for the nonlinear distortion introduced by power amplifiers, improving:
- **Out-of-band suppression** (ACLR - Adjacent Channel Leakage Ratio)
- **In-band distortion** (EVM - Error Vector Magnitude, NMSE - Normalized Mean Square Error)

## Training

### Least-Squares DPD Training

LS-DPD uses closed-form least-squares estimation:

```bash
python demos/dpd/training_ls.py
```

**Options:**
- `--iterations`: Number of LS iterations (default: 3)
- `--batch_size`: Batch size for signal generation (default: 16)
- `--order`: DPD polynomial order (default: 7)
- `--memory_depth`: DPD memory depth (default: 4)
- `--learning_rate`: LS learning rate (default: 0.75)
- `--learning_method`: Learning method - `newton` or `ema` (default: newton)

**Example:**
```bash
python demos/dpd/training_ls.py --iterations 5 --batch_size 32
```

**Outputs:**
- `results/ls-dpd-weights` - Trained DPD coefficients
- `results/ls-dpd-coeff-history.npy` - Coefficient history across iterations

### Neural Network DPD Training

NN-DPD uses gradient-based learning:

```bash
python demos/dpd/training_nn.py --iterations 10000
```

**Options:**
- `--iterations`: Number of training iterations (default: 10000)
- `--fresh`: Start fresh training, ignoring any existing checkpoint
- `--batch_size`: Batch size (default: 16)

**Example:**
```bash
python demos/dpd/training_nn.py --iterations 5000 --fresh
```

**Outputs:**
- `results/nn-dpd-weights` - Trained neural network weights
- `results/loss.npy` - Training loss history
- `checkpoints/` - Training checkpoints for resumption

## Inference

Run inference to evaluate the trained DPD model:

```bash
# For Neural Network DPD
python demos/dpd/inference.py --dpd_method nn

# For Least-Squares DPD
python demos/dpd/inference.py --dpd_method ls
```

**Options:**
- `--dpd_method`: DPD method - `nn` or `ls` (default: nn)
- `--batch_size`: Batch size for evaluation (default: 16)

**Outputs:**
- `results/inference_results_{method}.npz` - ACLR, NMSE, EVM metrics
- `results/psd_data_{method}.npz` - Power Spectral Density data
- `results/constellation_data_{method}.npz` - Constellation data for plotting

**Metrics reported:**
- ACLR (Adjacent Channel Leakage Ratio) - Lower and Upper
- NMSE (Normalized Mean Square Error)
- EVM (Error Vector Magnitude)

## Generating Plots

After running training and inference, generate visualization plots:

```bash
# For Neural Network DPD
python demos/dpd/plots_nn.py

# For Least-Squares DPD
python demos/dpd/plots_ls.py
```

**Generated plots:**
- `results/training_loss.png` - Training loss curve (NN-DPD only)
- `results/psd_comparison_{method}.png` - Power Spectral Density comparison
- `results/constellation_comparison_{method}.png` - Constellation diagram comparison

## Complete Workflow Example
For NN-DPD:

```bash
# 1. Train NN-DPD
python demos/dpd/training_nn.py --iterations 10000

# 2. Run inference
python demos/dpd/inference.py --dpd_method nn

# 3. Generate plots
python demos/dpd/plots_nn.py
```

For LS-DPD:

```bash
# 1. Train LS-DPD
python demos/dpd/training_ls.py --iterations 5

# 2. Run inference
python demos/dpd/inference.py --dpd_method ls

# 3. Generate plots
python demos/dpd/plots_ls.py
```

## Running Tests

```bash
pytest demos/dpd/tests/ -v
```
