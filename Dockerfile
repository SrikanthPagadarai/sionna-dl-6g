# syntax=docker/dockerfile:1.7

# Build-time switches
ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ARG ENABLE_GPU=1                  # 1=GPU image, 0=CPU-only
ARG ENABLE_CUDA_CHECK=1           # 1=fail build if TF wheel CUDA != image CUDA
ARG TF_PACKAGE=tensorflow         # "tensorflow" or "tensorflow-cpu"
ARG TF_VERSION=2.15.1             # exact TF pin

FROM ${BASE_IMAGE} AS runtime

# Common environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    TF_CPP_MIN_LOG_LEVEL=2 \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

WORKDIR /app

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    build-essential curl git ca-certificates vim \
 && rm -rf /var/lib/apt/lists/*

# Python base + Poetry
RUN python3 -m venv "${VIRTUAL_ENV}" \
 && python -m pip install --upgrade pip wheel setuptools \
 && python -m pip install "poetry==1.8.3"

# App deps via Poetry (no TF here)
COPY pyproject.toml poetry.lock* /app/
ARG POETRY_WITH=""
RUN poetry config virtualenvs.create false \
 && (poetry lock --no-interaction --no-update || poetry lock --no-interaction) \
 && poetry install --no-interaction --no-root --only main ${POETRY_WITH:+--with ${POETRY_WITH}}

# Install TensorFlow last so it "wins"
ARG TF_PACKAGE
ARG TF_VERSION
RUN python -m pip install --upgrade --force-reinstall "${TF_PACKAGE}==${TF_VERSION}"

# (GPU only) Minimal CUDA toolchain for XLA JIT (nvvm/libdevice/ptxas)
ARG ENABLE_GPU
RUN if [ "${ENABLE_GPU}" = "1" ]; then \
      set -e; \
      CUDA_DIR="$(readlink -f /usr/local/cuda || true)"; \
      CUDA_MM=""; \
      if [ -f /usr/local/cuda/version.txt ]; then \
        CUDA_MM="$(awk '{print $3}' /usr/local/cuda/version.txt | cut -d. -f1-2)"; \
      fi; \
      if [ -z "${CUDA_MM}" ] && [ -n "${CUDA_DIR}" ]; then \
        CUDA_MM="$(basename "${CUDA_DIR}" | sed -n 's/^cuda-\([0-9]\+\.[0-9]\+\).*/\1/p')"; \
      fi; \
      if [ -z "${CUDA_MM}" ]; then echo "ERROR: Could not determine CUDA version"; exit 1; fi; \
      echo "Detected CUDA ${CUDA_MM}"; \
      apt-get update; \
      if [ "${CUDA_MM}" = "12.2" ]; then \
        apt-get install -y --no-install-recommends cuda-nvcc-12-2; \
      elif [ "${CUDA_MM}" = "11.8" ]; then \
        apt-get install -y --no-install-recommends cuda-nvcc-11-8; \
      else \
        apt-get install -y --no-install-recommends cuda-nvcc; \
      fi; \
      rm -rf /var/lib/apt/lists/*; \
    fi

# (GPU only) Build-time sanity check: TF wheel CUDA == image CUDA
ARG ENABLE_CUDA_CHECK
RUN if [ "${ENABLE_GPU}" = "1" ] && [ "${ENABLE_CUDA_CHECK}" = "1" ]; then \
      IMG_MM=""; \
      if [ -f /usr/local/cuda/version.txt ]; then \
        IMG_MM="$(awk '{print $3}' /usr/local/cuda/version.txt | cut -d. -f1-2)"; \
      fi; \
      if [ -z "${IMG_MM}" ]; then \
        CUDA_DIR="$(readlink -f /usr/local/cuda || true)"; \
        IMG_MM="$(basename "${CUDA_DIR}" | sed -n 's/^cuda-\([0-9]\+\.[0-9]\+\).*/\1/p')"; \
      fi; \
      if [ -z "${IMG_MM}" ]; then echo "ERROR: Could not determine image CUDA version"; exit 1; fi; \
      python -c "import sys,tensorflow as tf; tf_cuda=str(tf.sysconfig.get_build_info().get('cuda_version','')).strip(); print(f'[CHECK] Image CUDA: ${IMG_MM} | TF wheel CUDA: {tf_cuda}'); sys.exit(0 if tf_cuda.startswith('${IMG_MM}') else 1)"; \
    fi

# App code & entrypoint
COPY . /app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "mimo_ofdm_neural_receiver/training.py"]
