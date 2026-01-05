Site-Specific PUSCH Autoencoder
===============================

Overview
--------

This demo implements an end-to-end autoencoder for the 5G NR Physical Uplink Shared Channel (PUSCH), jointly optimizing a trainable constellation at the transmitter and a neural MIMO detector at the receiver. The system operates over site-specific ray-traced channels derived from a realistic Munich urban environment, enabling the autoencoder to learn communication strategies adapted to the propagation characteristics of a specific deployment.

The autoencoder extends concepts from classical communication autoencoders (O'Shea & Hoydis, 2017) to a multi-user MIMO uplink scenario with 4 UEs (each with 4 antennas) transmitting to a base station with 16 or 32 antennas. This builds upon the foundations established in the `Sionna 5G NR PUSCH Tutorial <https://nvlabs.github.io/sionna/phy/tutorials/5G_NR_PUSCH.html>`_, the `Link-Level Simulations with Ray Tracing Tutorial <https://nvlabs.github.io/sionna/phy/tutorials/Link_Level_Simulations_with_RT.html>`_, and the `Autoencoder Tutorial <https://nvlabs.github.io/sionna/phy/tutorials/Autoencoder.html>`_.

.. image:: /_static/pusch_autoencoder/pusch_autoencoder_light.svg
   :class: only-light
   :alt: PUSCH Autoencoder System Architecture

.. image:: /_static/pusch_autoencoder/pusch_autoencoder_dark.svg
   :class: only-dark
   :alt: PUSCH Autoencoder System Architecture


System Architecture
-------------------

The PUSCH link (:class:`~demos.pusch_autoencoder.src.system.PUSCHLinkE2E`) implements a complete uplink chain: TB encoding, symbol mapping via trainable constellation, OFDM modulation with DMRS pilots, ray-traced channel application, and neural detection. The system uses MCS index 14 (16-QAM, ~0.6 code rate) with 16 PRBs (192 subcarriers) and single-layer transmission per UE.

Channel impulse responses (CIRs) are pre-computed using Sionna's ray tracer on the Munich urban scene, with UE positions sampled within 5–400 m of the BS location. The :class:`~demos.pusch_autoencoder.src.cir_manager.CIRManager` handles TFRecord serialization and MU-MIMO grouping of individual UE channels into multi-user samples.

Trainable Transmitter
^^^^^^^^^^^^^^^^^^^^^

The transmitter (:class:`~demos.pusch_autoencoder.src.pusch_trainable_transmitter.PUSCHTrainableTransmitter`) extends Sionna's ``PUSCHTransmitter`` with learnable constellation points. The 16 complex constellation points are stored as separate real/imaginary ``tf.Variable`` tensors, initialized from standard Gray-coded 16-QAM. Normalization to unit average power is applied explicitly at each forward pass to maintain the SNR interpretation.

.. code-block:: python
   :caption: Constellation normalization (``pusch_trainable_transmitter.py:107-146``)

    def get_normalized_constellation(self):
        points = tf.complex(self._points_r, self._points_i)
        # Center: remove DC offset for balanced constellation
        points = points - tf.reduce_mean(points)
        # Normalize: scale to unit average power
        energy = tf.reduce_mean(tf.square(tf.abs(points)))
        points = points / tf.cast(tf.sqrt(energy), points.dtype)
        return points


Neural Detector
^^^^^^^^^^^^^^^

The neural detector (:class:`~demos.pusch_autoencoder.src.pusch_neural_detector.PUSCHNeuralDetector`) implements a hybrid classical/neural architecture that refines LS channel estimates and LMMSE soft symbols rather than learning detection from scratch. The architecture processes shared features through convolutional residual blocks before branching into channel estimation refinement and LLR prediction heads.

.. image:: /_static/pusch_autoencoder/pusch_autoencoder_neural_detector_toplevel_light.svg
   :class: only-light
   :alt: Neural Detector Top-Level Architecture

.. image:: /_static/pusch_autoencoder/pusch_autoencoder_neural_detector_toplevel_dark.svg
   :class: only-dark
   :alt: Neural Detector Top-Level Architecture

Three trainable correction scales control how much the neural network deviates from classical LMMSE processing. These scales start at zero (pure classical behavior) and are learned during training, providing interpretable indicators of where neural refinement helps most.

.. image:: /_static/pusch_autoencoder/pusch_autoencoder_res_block_light.svg
   :class: only-light
   :alt: Residual Block Architecture

.. image:: /_static/pusch_autoencoder/pusch_autoencoder_res_block_dark.svg
   :class: only-dark
   :alt: Residual Block Architecture


Training
--------

Training minimizes binary cross-entropy (BCE) loss between predicted LLRs and transmitted coded bits, plus a constellation regularization term that prevents collapse by penalizing constellation points closer than a minimum distance threshold. The system uses gradient accumulation over 16 micro-batches with separate Adam optimizers for transmitter variables (LR 1e-2), receiver correction scales (LR 1e-2), and neural network weights (LR 1e-4), all with cosine decay schedules over 5,000 iterations.

Eb/N0 is sampled uniformly from -2 to 10 dB for each batch, enabling the autoencoder to learn robust strategies across the operating SNR range. Training requires pre-generated CIR data via the :class:`~demos.pusch_autoencoder.src.cir_manager.CIRManager`.


Results
-------

Performance is evaluated using BER and BLER Monte Carlo simulation, comparing the trained autoencoder against baseline LMMSE detection with both perfect and imperfect (LS-estimated) CSI. The following figures show results for 16 and 32 BS antenna configurations.

.. figure:: ../../demos/pusch_autoencoder/results/munich_ue_positions.png
   :alt: Munich UE Positions
   :align: center
   :width: 80%

   Ray-traced Munich urban scene showing sampled UE positions for CIR generation.

.. figure:: ../../demos/pusch_autoencoder/results/ber_plot_1bs_16bs_ant_x_4ue_4ue_ant.png
   :alt: BER Plot 16 BS Antennas
   :align: center
   :width: 80%

   BER comparison: autoencoder vs. baseline LMMSE with 16 BS antennas.

.. figure:: ../../demos/pusch_autoencoder/results/bler_plot_1bs_16bs_ant_x_4ue_4ue_ant.png
   :alt: BLER Plot 16 BS Antennas
   :align: center
   :width: 80%

   BLER comparison: autoencoder vs. baseline LMMSE with 16 BS antennas.

.. figure:: ../../demos/pusch_autoencoder/results/ber_plot_1bs_32bs_ant_x_4ue_4ue_ant.png
   :alt: BER Plot 32 BS Antennas
   :align: center
   :width: 80%

   BER comparison: autoencoder vs. baseline LMMSE with 32 BS antennas.

.. figure:: ../../demos/pusch_autoencoder/results/bler_plot_1bs_32bs_ant_x_4ue_4ue_ant.png
   :alt: BLER Plot 32 BS Antennas
   :align: center
   :width: 80%

   BLER comparison: autoencoder vs. baseline LMMSE with 32 BS antennas.

.. figure:: ../../demos/pusch_autoencoder/results/constellation_normalized_ant16.png
   :alt: Learned Constellation 16 Antennas
   :align: center
   :width: 80%

   Learned constellation geometry (16 BS antennas) compared to standard 16-QAM.

.. figure:: ../../demos/pusch_autoencoder/results/constellation_normalized_ant32.png
   :alt: Learned Constellation 32 Antennas
   :align: center
   :width: 80%

   Learned constellation geometry (32 BS antennas) compared to standard 16-QAM.

The autoencoder demonstrates improved performance over imperfect-CSI LMMSE baseline, particularly at mid-to-high SNR where channel estimation errors dominate. The learned constellation points deviate moderately from standard 16-QAM, adapting to the site-specific channel statistics while maintaining sufficient minimum distance for reliable detection.


References
----------

- O'Shea & Hoydis, "An Introduction to Deep Learning for the Physical Layer," IEEE TCCN, 2017
- Sionna 5G NR PUSCH Tutorial: https://nvlabs.github.io/sionna/phy/tutorials/5G_NR_PUSCH.html
- Sionna Link-Level Simulations with Ray Tracing: https://nvlabs.github.io/sionna/phy/tutorials/Link_Level_Simulations_with_RT.html
- Sionna Autoencoder Tutorial: https://nvlabs.github.io/sionna/phy/tutorials/Autoencoder.html
- 3GPP TS 38.212: 5G NR Multiplexing and channel coding