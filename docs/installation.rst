Installation
============

Prerequisites
-------------

- Python 3.10 - 3.12
- NVIDIA GPU with CUDA support (recommended)
- TensorFlow 2.x

Install via pip (PyPI)
----------------------

Install the latest release from PyPI:

.. code-block:: bash

   pip install nextgen-wireless-dl-demos

Install via pip (GitHub)
------------------------

Alternatively, install the latest development version directly from the repository:

.. code-block:: bash

   pip install git+https://github.com/SrikanthPagadarai/nextgen-wireless-dl-demos.git

Install via Poetry
------------------

For development, clone the repository and install dependencies using Poetry:

.. code-block:: bash

   git clone --recurse-submodules https://github.com/SrikanthPagadarai/nextgen-wireless-dl-demos.git
   cd nextgen-wireless-dl-demos
   poetry install

If you've already cloned without ``--recurse-submodules``, initialize the submodule separately:

.. code-block:: bash

   git submodule update --init --recursive

Dependencies
------------

The project depends on:

- `Sionna <https://nvlabs.github.io/sionna/>`_ (≥0.19.0) - NVIDIA's library for link-level simulations
- TensorFlow 2.x - Deep learning framework
- Matplotlib - Visualization

GPU Setup
---------

For optimal performance, ensure you have:

1. NVIDIA drivers installed
2. CUDA toolkit compatible with your TensorFlow version
3. cuDNN library

Refer to the `TensorFlow GPU guide <https://www.tensorflow.org/install/gpu>`_ for detailed setup instructions.


Cloud and Container Setup
=========================

For running GPU-accelerated workloads in the cloud using Docker containers, follow these three steps in order:

1. Provision cloud infrastructure (Step 1)
2. Configure the host NVIDIA runtime (Step 2)
3. Build and run the Docker container (Step 3)


Step 1: Cloud Platform Setup (GCP)
----------------------------------

This project provides helper scripts for provisioning GPU-enabled virtual machines on Google Cloud Platform (GCP). These scripts are included as a git submodule in the ``gcp-management/`` directory (source repository: `gcp-management <https://github.com/SrikanthPagadarai/gcp-management>`_).

.. note::

   While we provide GCP-specific tooling, you can use any cloud provider that offers GPU instances (AWS, Azure, etc.) or your own hardware. The key requirements are an NVIDIA GPU with CUDA support and Docker with the NVIDIA Container Toolkit.

GCP Management Files Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``gcp-management/`` submodule contains five files:

- ``.env.example`` - Template for environment variables (copy to ``.env`` and fill in your values)
- ``.gitignore`` - Ensures sensitive ``.env`` files are not committed
- ``README.md`` - Quick reference for common commands
- ``gcloud-setup.sh`` - Main script for creating CPU or GPU VMs
- ``gcloud-reset.sh`` - Script to reset gcloud configuration and credentials

Environment Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to the submodule directory and create a ``.env`` file from the template:

.. code-block:: bash

   cd gcp-management
   cp .env.example .env

Edit ``.env`` with your GCP project details:

.. code-block:: bash

   PROJECT_ID=your-gcp-project-id
   INSTANCE_NAME=your-vm-name
   BUCKET_NAME=your-bucket-name
   ZONE=us-central1-a
   SSH_USER=$(whoami)

Authentication
^^^^^^^^^^^^^^

Before running the setup script, authenticate with GCP:

.. code-block:: bash

   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID

Creating a GPU VM
^^^^^^^^^^^^^^^^^

Make the setup script executable and run it:

.. code-block:: bash

   chmod +x gcloud-setup.sh

   # Create a GPU VM with NVIDIA L4
   ./gcloud-setup.sh \
     --mode gpu \
     --project YOUR_PROJECT_ID \
     --name YOUR_VM_NAME \
     --zone us-central1-a \
     --gpu-type nvidia-l4 \
     --gpu-count 1 \
     --gpu-machine g2-standard-4

The script supports multiple GPU types and configurations:

.. code-block:: bash

   # T4 GPU on n1 machine
   ./gcloud-setup.sh --mode gpu --gpu-type nvidia-tesla-t4 --gpu-machine n1-standard-8

   # A100 GPU with spot pricing (cost savings)
   ./gcloud-setup.sh --mode gpu --gpu-type nvidia-a100 --gpu-machine a2-highgpu-2g --provisioning SPOT

   # CPU-only VM for development
   ./gcloud-setup.sh --mode cpu --cpu-type e2-standard-2

Connecting to the VM
^^^^^^^^^^^^^^^^^^^^

SSH into your newly created VM:

.. code-block:: bash

   gcloud compute ssh YOUR_USERNAME@YOUR_VM_NAME --zone=us-central1-a

Resetting GCP Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reset your gcloud credentials and configurations:

.. code-block:: bash

   # Soft reset (revoke credentials, clear configs)
   bash gcloud-reset.sh

   # Full reset (removes ~/.config/gcloud entirely)
   bash gcloud-reset.sh --nuke


Step 2: Host NVIDIA Runtime Setup
---------------------------------

Before running GPU-accelerated Docker containers, you must configure the NVIDIA Container Toolkit on your host machine. This step is required whether you're using a cloud VM or local hardware.

The ``host_nvidia_runtime_setup.sh`` script in the repository root automates this process:

.. code-block:: bash

   chmod +x host_nvidia_runtime_setup.sh
   ./host_nvidia_runtime_setup.sh

What the Script Does
^^^^^^^^^^^^^^^^^^^^

The script performs four operations:

1. **Installs NVIDIA Container Toolkit** - The toolkit allows Docker containers to access the host GPU.

   .. code-block:: bash

      sudo apt-get update
      sudo apt-get install -y nvidia-container-toolkit

2. **Configures Docker runtime** - Registers the NVIDIA runtime with Docker.

   .. code-block:: bash

      sudo nvidia-ctk runtime configure --runtime=docker

3. **Enables graphics capabilities** - Configures the runtime to expose compute, utility, and graphics capabilities to containers (required for Sionna's ray tracing features).

   .. code-block:: bash

      sudo sed -i \
        's/^#\?\s*env = .*NVIDIA_DRIVER_CAPABILITIES.*/env = ["NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"]/g' \
        /etc/nvidia-container-runtime/config.toml

4. **Restarts Docker** - Applies the configuration changes.

   .. code-block:: bash

      sudo systemctl restart docker

Verification
^^^^^^^^^^^^

After running the setup script, verify the installation:

.. code-block:: bash

   # Check NVIDIA driver is accessible
   nvidia-smi

   # Verify Docker can access the GPU
   docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi


Step 3: Docker Setup
--------------------

The project includes a multi-stage Dockerfile supporting both CPU and GPU builds. The Docker configuration files are located in the repository root (``Dockerfile``) and the ``docker/`` directory.

Docker Files Overview
^^^^^^^^^^^^^^^^^^^^^

- ``Dockerfile`` - Multi-stage build supporting CPU and GPU configurations
- ``docker/docker-instructions.md`` - Quick reference for build and run commands
- ``docker/entrypoint.sh`` - Container entrypoint script that auto-detects OptiX for GPU acceleration

Building Docker Images
^^^^^^^^^^^^^^^^^^^^^^

**GPU Build** (recommended for training and inference):

.. code-block:: bash

   docker build \
     --build-arg BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 \
     --build-arg ENABLE_GPU=1 \
     --build-arg TF_PACKAGE=tensorflow \
     --build-arg TF_VERSION=2.15.1 \
     -t nextgen-wireless-dl-demos:gpu-12.2 .

**CPU Build** (for development or systems without GPU):

.. code-block:: bash

   docker build \
     --build-arg BASE_IMAGE=ubuntu:24.04 \
     --build-arg ENABLE_GPU=0 \
     --build-arg ENABLE_CUDA_CHECK=0 \
     --build-arg TF_PACKAGE=tensorflow-cpu \
     --build-arg TF_VERSION=2.15.1 \
     -t nextgen-wireless-dl-demos:cpu .

Build Arguments Reference
^^^^^^^^^^^^^^^^^^^^^^^^^

The Dockerfile accepts several build-time arguments:

- ``BASE_IMAGE`` - Base Docker image (NVIDIA CUDA image for GPU, Ubuntu for CPU)
- ``ENABLE_GPU`` - Set to ``1`` for GPU support, ``0`` for CPU-only
- ``ENABLE_CUDA_CHECK`` - Set to ``1`` to verify TensorFlow CUDA version matches the base image
- ``TF_PACKAGE`` - TensorFlow package name (``tensorflow`` for GPU, ``tensorflow-cpu`` for CPU)
- ``TF_VERSION`` - TensorFlow version to install

Running Containers
^^^^^^^^^^^^^^^^^^

**Run with GPU access**:

.. code-block:: bash

   docker run -it --gpus all \
     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
     -e DRJIT_LIBOPTIX_PATH=/usr/lib/x86_64-linux-gnu/libnvoptix.so.1 \
     -v "$PWD":/app -w /app \
     nextgen-wireless-dl-demos:gpu-12.2

**Run CPU-only**:

.. code-block:: bash

   docker run -it \
     -v "$PWD":/app -w /app \
     nextgen-wireless-dl-demos:cpu bash

Entrypoint Script Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``docker/entrypoint.sh`` script automatically detects OptiX availability at container startup. It searches for ``libnvoptix.so.1`` in common locations and configures the Sionna/Mitsuba backend accordingly:

- **GPU mode** (``cuda_ad_rgb``): Used when OptiX is found, enabling GPU-accelerated ray tracing
- **CPU mode** (``llvm_ad_rgb``): Fallback when OptiX is not available

The script outputs which mode is selected at container startup, for example:

.. code-block:: text

   [entrypoint] OptiX found at: /usr/lib/x86_64-linux-gnu/libnvoptix.so.1 -> using CUDA (cuda_ad_rgb)

Running Demo Scripts in Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once inside the container, run demos using Python module syntax:

.. code-block:: bash

   # Training
   python -m demos.dpd.training_nn
   python -m demos.mimo_ofdm_neural_receiver.training
   python -m demos.pusch_autoencoder.training

   # Inference
   python -m demos.dpd.inference
   python -m demos.mimo_ofdm_neural_receiver.inference
   python -m demos.pusch_autoencoder.inference