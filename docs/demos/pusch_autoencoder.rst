PUSCH Autoencoder
=================

Overview
--------

This demo implements an autoencoder for the 5G NR Physical Uplink Shared Channel (PUSCH),
learning joint transmitter-receiver optimization.

Background
----------

Autoencoders treat the entire communication system as an end-to-end learning problem.
The transmitter (encoder) and receiver (decoder) are jointly optimized to minimize
reconstruction error over the wireless channel.

Architecture
------------

.. todo:: Add architecture diagram

The autoencoder consists of:

1. **Encoder (Transmitter)**: Maps information bits to channel symbols
2. **Channel**: Simulates the 5G NR PUSCH transmission including:

   - OFDM modulation
   - MIMO precoding
   - Fading channel
   - AWGN noise

3. **Decoder (Receiver)**: Recovers transmitted bits from received signal

Configuration
-------------

The demo supports standard 5G NR PUSCH configurations:

.. code-block:: python

   pusch_config = {
       "num_rx_antennas": 4,
       "num_tx_antennas": 2,
       "num_layers": 2,
       "num_prbs": 48,
       "mcs_index": 10,
   }

Training Strategy
-----------------

Training proceeds in two phases:

1. **Pre-training**: Train over AWGN channel for stable initialization
2. **Fine-tuning**: Train over realistic fading channels

Results
-------

.. todo:: Add BLER curves comparing autoencoder vs. baseline PUSCH

References
----------

- O'Shea & Hoydis, "An Introduction to Deep Learning for the Physical Layer," IEEE TCCN, 2017
- Sionna 5G NR PUSCH Tutorial: https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html
