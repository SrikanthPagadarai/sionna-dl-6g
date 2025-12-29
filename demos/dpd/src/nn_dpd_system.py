"""Neural Network DPD System for training and inference.

Extends base DPDSystem with Neural Network-based Digital Pre-Distortion.
Uses gradient-based optimization with indirect learning architecture.
"""

import tensorflow as tf

from .config import Config
from .system import DPDSystem
from .nn_dpd import NeuralNetworkDPD


class NN_DPDSystem(DPDSystem):
    """
    Neural Network DPD system for training and inference.

    Extends DPDSystem with NN-based predistortion using indirect learning.

    For NN-DPD:
        Returns indirect learning loss: ||DPD(PA_output/G) - predistorter_output||²

    The indirect learning approach:
    1. Generate baseband signal x
    2. Upsample to PA sample rate
    3. Apply predistorter: u = DPD(x)
    4. Pass through PA: y = PA(u)
    5. Normalize by PA gain: y_norm = y / G
    6. Train postdistorter: loss = ||DPD(y_norm) - u||²

    Args:
        training: Whether in training mode
        config: Config instance with system parameters
        dpd_memory_depth: DPD memory depth (default: 4)
        dpd_num_filters: DPD hidden layer size (default: 64)
        dpd_num_layers_per_block: Layers per residual block (default: 2)
        dpd_num_res_blocks: Number of residual blocks (default: 3)
        rms_input_dbm: Target input RMS power in dBm (default: 0.5)
        pa_sample_rate: PA sample rate in Hz (default: 122.88e6)
    """

    def __init__(
        self,
        training: bool,
        config: Config,
        dpd_memory_depth: int = 4,
        dpd_num_filters: int = 64,
        dpd_num_layers_per_block: int = 2,
        dpd_num_res_blocks: int = 3,
        rms_input_dbm: float = 0.5,
        pa_sample_rate: float = 122.88e6,
        **kwargs,
    ):
        super().__init__(
            training=training,
            config=config,
            rms_input_dbm=rms_input_dbm,
            pa_sample_rate=pa_sample_rate,
            **kwargs,
        )

        # Neural Network DPD layer
        self._dpd = NeuralNetworkDPD(
            memory_depth=dpd_memory_depth,
            num_filters=dpd_num_filters,
            num_layers_per_block=dpd_num_layers_per_block,
            num_res_blocks=dpd_num_res_blocks,
        )

        # Loss function with scaling for better gradient flow
        self._loss_fn = tf.keras.losses.MeanSquaredError()
        self._loss_scale = 1000.0  # Scale up loss for better monitoring

    def _forward_signal_path(self, x):
        """
        Forward signal through predistorter and PA (steps 1-3 of indirect learning).

        For NN-DPD: Input is normalized for better NN conditioning.

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            dict with:
                - u: predistorted signal (original scale)
                - u_norm: predistorted signal (normalized scale, for NN loss)
                - y_comp: gain-compensated PA output
                - x_scale: input normalization scale factor
        """
        # NN-DPD: Normalize input for better conditioning
        x_norm, x_scale = self._normalize_to_unit_power(x)
        u_norm = self._dpd(x_norm, training=False)
        u = u_norm * tf.cast(x_scale, u_norm.dtype)

        # Step 2: Pass through PA
        y = self._pa(u)

        # Step 3: Compensate for PA gain
        y_comp = y / tf.cast(self._pa_gain, y.dtype)

        return {
            "u": u,
            "u_norm": u_norm,
            "y_comp": y_comp,
            "x_scale": x_scale,
        }

    def _training_forward(self, x):
        """
        Training forward pass with indirect learning.

        Complete indirect learning architecture:
            Step 1: Apply predistorter: u = DPD(x)
            Step 2: Pass through PA: y = PA(u)
            Step 3: Compensate for PA gain: y_comp = y / G
            Step 4: Apply postdistorter: u_hat = DPD(y_comp)
            Step 5: Compute loss: ||u - u_hat||²

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            scalar MSE loss
        """
        # Steps 1-3: Forward through predistorter and PA
        signals = self._forward_signal_path(x)
        u_norm = signals["u_norm"]
        y_comp = signals["y_comp"]

        # Stop gradient on target (predistorter output)
        u_target = tf.stop_gradient(u_norm)

        # Normalize PA output for postdistorter
        y_norm, _ = self._normalize_to_unit_power(y_comp)

        # Step 4: Apply postdistorter (this is what we're training)
        u_hat_norm = self._dpd(y_norm, training=True)

        # Step 5: Compute loss in normalized domain
        u_target_ri = tf.stack(
            [tf.math.real(u_target), tf.math.imag(u_target)], axis=-1
        )
        u_hat_ri = tf.stack(
            [tf.math.real(u_hat_norm), tf.math.imag(u_hat_norm)], axis=-1
        )

        loss = self._loss_fn(u_target_ri, u_hat_ri) * self._loss_scale
        return loss

    def _inference_forward(self, x):
        """
        Inference forward pass.

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            dict with PA input and outputs
        """
        # PA output without DPD
        pa_output_no_dpd = self._pa(x)

        # NN-DPD: Normalize for DPD, apply DPD, scale back
        x_norm, x_scale = self._normalize_to_unit_power(x)
        x_predistorted_norm = self._dpd(x_norm, training=False)
        x_predistorted = x_predistorted_norm * tf.cast(
            x_scale, x_predistorted_norm.dtype
        )

        # Pass through PA
        pa_output_with_dpd = self._pa(x_predistorted)

        return {
            "pa_input": x,
            "pa_output_no_dpd": pa_output_no_dpd,
            "pa_output_with_dpd": pa_output_with_dpd,
            "predistorted": x_predistorted,
        }
