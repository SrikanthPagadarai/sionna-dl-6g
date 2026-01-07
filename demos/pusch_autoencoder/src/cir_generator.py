# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
CIR sample generator for Sionna's CIRDataset integration.

Provides an infinite generator that yields random CIR samples from a
pre-loaded dataset. It serves as the data source for Sionna's
``CIRDataset`` class, which expects a callable that yields ``(a, tau)``
tuples on each iteration.

The generator implements random sampling without replacement within each
call, ensuring diverse UE combinations in each MU-MIMO sample while
maintaining uniform coverage over the dataset across many iterations.

Integration Pattern
-------------------
The generator is designed to work with Sionna's channel modeling pipeline::

    # Load CIR data from TFRecords
    a, tau = manager.load_from_tfrecord()

    # Create generator (callable that yields samples)
    cir_gen = CIRGenerator(a, tau, num_tx=4)

    # Wrap in CIRDataset for use with OFDMChannel
    channel_model = CIRDataset(cir_gen, batch_size, ...)

    # Use in simulation
    channel = OFDMChannel(channel_model, ...)
    y = channel([x, no])
"""

import tensorflow as tf


class CIRGenerator:
    r"""
    Infinite generator for random CIR sample selection.

    This class wraps a pre-loaded CIR dataset and provides an infinite
    generator interface compatible with Sionna's ``CIRDataset``. Each
    call to the generator yields a new random combination of UE channels,
    simulating the scenario where different UEs are co-scheduled in each
    transmission slot.

    Parameters
    ----------
    a : tf.Tensor or np.ndarray, complex64
        CIR path coefficients from ray-tracing. Shape:
        ``[num_samples, 1, num_bs_ant, 1, num_ue_ant, max_paths, num_time_steps]``
    tau : tf.Tensor or np.ndarray, float32
        Path delays in seconds. Shape:
        ``[num_samples, 1, 1, max_paths]``
    num_tx : int
        Number of transmitters (UEs) to sample for each yield.
        Typically equals ``num_ue`` in the system configuration.

    Example
    -------
    >>> a, tau = manager.load_from_tfrecord()  # [5000, 1, 16, 1, 4, 51, 14]
    >>> gen = CIRGenerator(a, tau, num_tx=4)
    >>> for a_sample, tau_sample in gen():
    ...     print(a_sample.shape)  # [16, 4, 4, 51, 14]
    ...     break

    Notes
    -----
    The generator uses ``tf.random.uniform_candidate_sampler`` for efficient
    random selection without replacement. This is more efficient than
    ``tf.random.shuffle`` for large datasets when only selecting a small
    subset.

    The dimension transpositions in ``__call__`` reorder from the storage
    format (sample-first) to Sionna's expected format (antenna-first):

    - Input ``a``: ``[num_tx, 1, num_bs_ant, 1, num_ue_ant, paths, time]``
    - Output ``a``: ``[num_bs_ant, num_tx, num_ue_ant, paths, time]``

    This reordering places the BS antenna dimension first, followed by the
    UE (TX) dimension, matching Sionna's channel tensor conventions.
    """

    def __init__(self, a, tau, num_tx):
        """
        Initialize the CIR generator with dataset and sampling parameters.

        Parameters
        ----------
        a : array-like, complex64
            CIR coefficients from ray-tracing dataset.
        tau : array-like, float32
            Path delays from ray-tracing dataset.
        num_tx : int
            Number of UEs to sample per iteration.
        """
        # Store as tf.constant for efficient repeated access
        # Complex64 and float32 match Sionna's internal precision
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]
        self._num_tx = num_tx

    def __call__(self):
        """
        Generate random CIR samples indefinitely.

        This method implements an infinite generator that yields new
        random UE combinations on each iteration. It is designed to be
        used as the data source for Sionna's ``CIRDataset``.

        Yields
        ------
        tuple of (a, tau)
            - ``a``: Complex path coefficients, shape
              ``[num_bs_ant, num_tx, num_ue_ant, max_paths, num_time_steps]``
            - ``tau``: Path delays, shape ``[1, num_tx, max_paths]``

        Notes
        -----
        The sampling uses ``uniform_candidate_sampler`` which efficiently
        selects ``num_tx`` unique indices from the dataset without
        replacement. This ensures each yielded sample contains channels
        from different UE positions, simulating realistic MU-MIMO scenarios
        where co-scheduled UEs are at different locations.

        The infinite loop means this generator never raises ``StopIteration``.
        Sionna's ``CIRDataset`` handles batch assembly and iteration limits.
        """
        # Infinite loop: Sionna's CIRDataset controls iteration count
        while True:
            # Sample num_tx unique indices without replacement
            # This simulates selecting different UEs for each MU-MIMO slot
            idx, _, _ = tf.random.uniform_candidate_sampler(
                # Dummy "true classes" tensor (required by API but not used)
                tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                num_true=self._dataset_size,
                num_sampled=self._num_tx,
                unique=True,  # Without replacement
                range_max=self._dataset_size,
            )

            # Gather CIR data for selected UE indices
            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose from storage format to Sionna's expected format:
            # Storage: [num_tx, 1, num_bs_ant, 1, num_ue_ant, paths, time]
            # Output:  [1, num_bs_ant, num_tx, num_ue_ant, paths, time]
            # After squeeze: [num_bs_ant, num_tx, num_ue_ant, paths, time]
            a = tf.transpose(a, (3, 1, 2, 0, 4, 5, 6))
            tau = tf.transpose(tau, (2, 1, 0, 3))

            # Remove singleton batch dimension (index 0 after transpose)
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)

            yield a, tau
