Installation
============

Prerequisites
-------------

- Python 3.10 - 3.12
- NVIDIA GPU with CUDA support (recommended)
- TensorFlow 2.x

Install via Poetry
------------------

Clone the repository and install dependencies using Poetry:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/sionna-dl-6g-demos.git
   cd sionna-dl-6g-demos
   poetry install

Install via pip
---------------

Alternatively, install directly from the repository:

.. code-block:: bash

   pip install git+https://github.com/YOUR_USERNAME/sionna-dl-6g-demos.git

Dependencies
------------

The project depends on:

- `Sionna <https://nvlabs.github.io/sionna/>`_ (≥0.15.0) - NVIDIA's library for link-level simulations
- TensorFlow 2.x - Deep learning framework
- Matplotlib - Visualization

GPU Setup
---------

For optimal performance, ensure you have:

1. NVIDIA drivers installed
2. CUDA toolkit compatible with your TensorFlow version
3. cuDNN library

Refer to the `TensorFlow GPU guide <https://www.tensorflow.org/install/gpu>`_ for detailed setup instructions.
