#!/usr/bin/env bash
set -euo pipefail

cd /app

## OptiX / CPU selection for Sionna-RT
# We try the most common locations for libnvoptix.so.1
pick_nvoptix_path() {
  local cands=(
    "${DRJIT_LIBOPTIX_PATH:-}"
    "/usr/local/nvidia/lib64/libnvoptix.so.1"                  # injected by NVIDIA runtime
    "/usr/lib/x86_64-linux-gnu/nvidia/current/libnvoptix.so.1" # host bind-mount path
    "/usr/lib/x86_64-linux-gnu/libnvoptix.so.1"                # distro path
  )
  for p in "${cands[@]}"; do
    if [[ -n "$p" && -e "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

if nvoptix="$(pick_nvoptix_path)"; then
  export DRJIT_LIBOPTIX_PATH="$nvoptix"
  export MI_DEFAULT_VARIANT="cuda_ad_rgb"
  echo "[entrypoint] OptiX found at: $nvoptix  -> using CUDA (cuda_ad_rgb)"
else
  export MI_DEFAULT_VARIANT="llvm_ad_rgb"
  echo "[entrypoint] OptiX not found  -> using CPU (llvm_ad_rgb)"
fi

# If no command given, open bash; if a command was provided, run it.
if [[ $# -eq 0 ]]; then
  exec bash
else
  exec "$@"
fi
