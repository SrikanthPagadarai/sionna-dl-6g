import tensorflow as tf


class CIRGenerator:
    """Generator for Channel Impulse Response (CIR) data.

    This class creates an infinite generator that yields random samples
    of CIR data for multiple transmitters.
    """

    def __init__(self, a, tau, num_tx):
        """Initialize the CIR generator.

        Args:
            a: CIR coefficients tensor
            tau: Delay values tensor
            num_tx: Number of transmitters to sample for each batch
        """
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]
        self._num_tx = num_tx

    def __call__(self):
        """Generate CIR samples indefinitely.

        Yields:
            Tuple of (a, tau) tensors with randomly sampled transmitters
        """
        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample random users and stack them together
            idx, _, _ = tf.random.uniform_candidate_sampler(
                tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                num_true=self._dataset_size,
                num_sampled=self._num_tx,
                unique=True,
                range_max=self._dataset_size,
            )

            # Gather the sampled data
            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose to rearrange dimensions for output format
            a = tf.transpose(a, (3, 1, 2, 0, 4, 5, 6))
            tau = tf.transpose(tau, (2, 1, 0, 3))

            # Remove batch dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)

            yield a, tau
