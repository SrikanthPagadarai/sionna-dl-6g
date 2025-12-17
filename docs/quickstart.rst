Quickstart
==========

This guide walks you through running your first demo.

Basic Usage
-----------

After installation, you can run the demos directly:

.. code-block:: bash

   python -m mimo_ofdm_neural_receiver.train
   python -m pusch_autoencoder.train

MIMO OFDM Neural Receiver
-------------------------

The neural receiver demo demonstrates learned signal processing for MIMO-OFDM systems:

.. code-block:: python

   from mimo_ofdm_neural_receiver import NeuralReceiver

   # Initialize and train
   receiver = NeuralReceiver()
   receiver.train(epochs=100)

   # Evaluate performance
   ber = receiver.evaluate(snr_range=range(-5, 20))

PUSCH Autoencoder
-----------------

The PUSCH autoencoder learns end-to-end transmission over 5G NR uplink:

.. code-block:: python

   from pusch_autoencoder import PUSCHAutoencoder

   # Create autoencoder
   autoencoder = PUSCHAutoencoder()
   autoencoder.train(epochs=50)

   # Compare with baseline
   autoencoder.plot_bler_comparison()

Next Steps
----------

- Explore the :doc:`demos/mimo_ofdm_neural_receiver` in detail
- Learn about the :doc:`demos/pusch_autoencoder` architecture
- Check the :doc:`api/index` for programmatic usage
