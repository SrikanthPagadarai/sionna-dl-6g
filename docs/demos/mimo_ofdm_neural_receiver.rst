MIMO OFDM Neural Receiver
=========================

Overview
--------

This demo implements a neural network-based receiver for MIMO-OFDM systems,
replacing traditional signal processing blocks with learned components.

Background
----------

Traditional MIMO-OFDM receivers perform channel estimation, equalization, and
demapping as separate blocks. A neural receiver can learn to perform these
operations jointly, potentially achieving better performance especially in
challenging channel conditions.

Architecture
------------

.. todo:: Add architecture diagram

The neural receiver architecture consists of:

1. **Input processing**: Converts received resource grid to suitable tensor format
2. **Feature extraction**: Convolutional layers process the 2D time-frequency grid
3. **Neural demapper**: Outputs LLRs for the coded bits

Training
--------

The model is trained end-to-end using the binary cross-entropy loss between
predicted and true LLRs.

.. code-block:: python

   # Training configuration
   config = {
       "num_epochs": 100,
       "batch_size": 64,
       "snr_range": (-5, 15),
       "learning_rate": 1e-3,
   }

Results
-------

.. todo:: Add BER/BLER curves comparing neural vs. baseline receiver

References
----------

- Honkala et al., "DeepRx: Fully Convolutional Deep Learning Receiver," IEEE TCCN, 2021
- Sionna Neural Receiver Tutorial: https://nvlabs.github.io/sionna/examples/Neural_Receiver.html
