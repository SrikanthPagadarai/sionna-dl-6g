import pickle
import numpy as np

WEIGHTS_FILE = "../results/mimo-ofdm-neuralrx-weights-uplink"  # adjust path if needed

with open(WEIGHTS_FILE, "rb") as f:
    weights = pickle.load(f)

# Helper: Conv2D kernel = rank-4 array
def is_conv_kernel(w):
    return isinstance(w, np.ndarray) and w.ndim == 4

conv_kernels = [w for w in weights if is_conv_kernel(w)]
if len(conv_kernels) < 2:
    raise RuntimeError("Not enough Conv2D kernels in this weights file.")

# Identify input conv: cin != cout
input_conv = None
for w in conv_kernels:
    _, _, cin, cout = w.shape
    if cin != cout:
        input_conv = w
        break
if input_conv is None:
    raise RuntimeError("Could not identify input Conv2D kernel (cin != cout).")

F = input_conv.shape[-1]  # num_conv2d_filters

# Identify output conv: cin == F and cout != F (maps F -> bits*streams)
output_conv = None
for w in conv_kernels:
    _, _, cin, cout = w.shape
    if cin == F and cout != F and w is not input_conv:
        output_conv = w
        break
if output_conv is None:
    raise RuntimeError("Could not identify output Conv2D kernel (cin==F and cout!=F).")

# Find indices of input/output kernels in the original weights list (by identity)
idx_in = next(i for i, w in enumerate(weights) if w is input_conv)
idx_out = next(i for i, w in enumerate(weights) if w is output_conv)

# Residual stack weights sit between input_conv block and output_conv block.
# input_conv has kernel + bias => start after idx_in+2
# output_conv has kernel + bias => ends before idx_out
residual_segment = weights[idx_in + 2 : idx_out]

if len(residual_segment) == 0:
    raise RuntimeError("Residual segment is empty; cannot infer residual structure.")

# Create a compact signature sequence for periodicity detection
# (ndim, shape) is enough; include exact shapes because they repeat per block.
sig = []
for w in residual_segment:
    if isinstance(w, np.ndarray):
        sig.append((w.ndim, tuple(w.shape)))
    else:
        # Unexpected type in get_weights(); represent by type name
        sig.append(("NON_NP", type(w).__name__))

N = len(sig)

# Find the smallest period p such that sig repeats every p elements
def smallest_period(seq):
    n = len(seq)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        ok = True
        for k in range(p, n):
            if seq[k] != seq[k % p]:
                ok = False
                break
        if ok:
            return p
    return None

p = smallest_period(sig)
if p is None:
    raise RuntimeError("Could not find a repeating pattern in residual weights.")

# Each residual “layer” contributes 4 arrays: LN(gamma,beta) + Conv(kernel,bias)
if p % 4 != 0:
    raise RuntimeError(
        f"Found residual block period p={p}, but p is not divisible by 4. "
        "This does not match the expected [LN gamma, LN beta, Conv kernel, Conv bias] pattern."
    )

num_resnet_layers = p // 4
num_res_blocks = N // p

print("\n=== NeuralRx architecture inferred from weights (pattern-based) ===")
print(f"num_conv2d_filters : {F}")
print(f"num_resnet_layers  : {num_resnet_layers}")
print(f"num_res_blocks     : {num_res_blocks}")
