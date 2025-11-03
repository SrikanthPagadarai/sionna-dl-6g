#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ -n "${NUM_TRAINING_ITERATIONS:-}" ]]; then
  sed -i -E "s/^(NUM_TRAINING_ITERATIONS\s*=\s*)[0-9]+/\1${NUM_TRAINING_ITERATIONS}/" \
    mimo_ofdm_neural_receiver/training.py
fi

if [[ "${1:-}" == "--iters" ]]; then
  iters="$2"
  sed -i -E "s/^(NUM_TRAINING_ITERATIONS\s*=\s*)[0-9]+/\1${iters}/" \
    mimo_ofdm_neural_receiver/training.py
  shift 2
fi

exec "$@"
