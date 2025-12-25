"""Batched signal processing utilities."""

import tensorflow as tf


def normalize_to_rms(data, target_rms):
    """
    Normalize batched signal to target RMS power.

    Computes global statistics across all batches (as if concatenated).

    Args:
        data: [B, num_samples] complex tensor
        target_rms: target RMS in dBm

    Returns:
        normalized data [B, num_samples], scale_factor
    """
    norm = tf.cast(tf.norm(data), tf.float32)
    n = tf.cast(tf.size(data), tf.float32)

    target_power = tf.constant(10 ** ((target_rms - 30) / 10), dtype=tf.float32)
    scale_factor = tf.sqrt(50.0 * n * target_power) / norm

    normalized = data * tf.cast(scale_factor, tf.complex64)

    return normalized, scale_factor
