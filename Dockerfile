# syntax=docker/dockerfile:1.7

# Selectable BASE
# Default to a GPU-ready base (CUDA 12.2 + cuDNN 9, Ubuntu 22.04)
ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn9-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS runtime-base

# Common env
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    TF_CPP_MIN_LOG_LEVEL=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    build-essential curl git ca-certificates vim \
 && rm -rf /var/lib/apt/lists/*

# Python base
RUN python3 -m venv "${VIRTUAL_ENV}" \
 && python -m pip install --upgrade pip wheel setuptools

# Poetry
ARG POETRY_VERSION=1.8.3
RUN python -m pip install "poetry==${POETRY_VERSION}"

# TensorFlow flavor
# Choose TF package at build-time:
#   - GPU host image:  TF_PACKAGE=tensorflow
#   - CPU-only image:  TF_PACKAGE=tensorflow-cpu
ARG TF_VERSION=2.17.*
ARG TF_PACKAGE=tensorflow
RUN python -m pip install "${TF_PACKAGE}==${TF_VERSION}"

# App deps via Poetry
COPY pyproject.toml poetry.lock* /app/
ARG POETRY_WITH=""
RUN poetry config virtualenvs.create false \
 && (poetry lock --no-interaction --no-update || poetry lock --no-interaction) \
 && poetry install --no-interaction --no-root --only main ${POETRY_WITH:+--with ${POETRY_WITH}}

# Force the exact TF version
ARG TF_VERSION=2.15.*
RUN python -m pip install --upgrade --force-reinstall "tensorflow==${TF_VERSION}"

# App code
COPY . /app

# Entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "mimo_ofdm_over_cdl/training.py"]
