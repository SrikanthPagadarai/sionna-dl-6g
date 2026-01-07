# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
Neural network DPD system with gradient-based training.

This subclass provides the complete NN-DPD system that combines the base
DPDSystem infrastructure with a feedforward neural network predistorter.
Unlike LS-DPD which computes coefficients in closed form, NN-DPD uses
backpropagation to iteratively optimize the network weights.

This system implements the indirect learning architecture where the same
network serves as both predistorter (during inference) and postdistorter
(during training).

The normalization steps are critical for NN-DPD to ensure stable gradient
flow regardless of the actual signal power levels.
"""

import tensorflow as tf

from .config import Config
from .system import DPDSystem
from .nn_dpd import NeuralNetworkDPD


class NN_DPDSystem(DPDSystem):
    """
    Complete NN-DPD system with gradient-based training.

    Extends the base DPDSystem with a feedforward neural network predistorter
    trained via backpropagation. The indirect learning architecture trains
    the network to invert the PA, then uses the same network for predistortion.

    Parameters
    ----------
    training : bool
        Operating mode. True enables gradient computation for training.
    config : Config
        Configuration object with RF and OFDM parameters.
    dpd_memory_depth : int, optional
        Sliding window size for memory effects. Default: 4.
    dpd_num_filters : int, optional
        Hidden layer width (model capacity). Default: 64.
    dpd_num_layers_per_block : int, optional
        Dense layers per residual block. Default: 2.
    dpd_num_res_blocks : int, optional
        Number of residual blocks (network depth). Default: 3.
    rms_input_dbm : float, optional
        Target RMS power for PA input in dBm. Default: 0.5.
    pa_sample_rate : float, optional
        PA operating sample rate in Hz. Default: 122.88 MHz.
    **kwargs
        Additional keyword arguments passed to base DPDSystem.

    Attributes
    ----------
    dpd : NeuralNetworkDPD
        The neural network predistorter layer.

    Notes
    -----
    **Why Normalize Inputs for NN-DPD?**

    Normalizing to unit power:

    - Keeps activations in a well-behaved range
    - Ensures consistent gradient magnitudes
    - Makes hyperparameters (learning rate) transferable across power levels

    The scale factor is saved and reapplied after predistortion.

    **Loss Scaling:**

    The MSE loss is scaled by 1000 for better monitoring. Raw MSE values
    for normalized signals are typically very small (1e-4 to 1e-6), making
    progress hard to track. Scaling doesn't affect optimization (just
    scales gradients uniformly).

    **Gradient Flow:**

    During training, gradients flow only through the postdistorter path.
    The predistorter output ``u`` is treated as a fixed target via
    ``tf.stop_gradient()``. This is the standard indirect learning setup.

    **miscellaneous:**

    - ``estimate_pa_gain()`` must be called before training
    - For training, use with ``tf.GradientTape`` to compute gradients
    - Training forward returns scalar loss
    - Inference forward returns dict with PA outputs

    Example
    -------
    >>> config = Config()
    >>> system = NN_DPDSystem(training=True, config=config)
    >>> system.estimate_pa_gain()
    >>>
    >>> optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    >>> for step in range(1000):
    ...     with tf.GradientTape() as tape:
    ...         loss = system(batch_size=16)
    ...     grads = tape.gradient(loss, system.trainable_variables)
    ...     optimizer.apply_gradients(zip(grads, system.trainable_variables))

    See Also
    --------
    LS_DPDSystem : Least-squares DPD with closed-form training.
    NeuralNetworkDPD : The underlying neural network layer.
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

        # Instantiate the neural network predistorter.
        self._dpd = NeuralNetworkDPD(
            memory_depth=dpd_memory_depth,
            num_filters=dpd_num_filters,
            num_layers_per_block=dpd_num_layers_per_block,
            num_res_blocks=dpd_num_res_blocks,
        )

        # MSE loss for indirect learning objective.
        self._loss_fn = tf.keras.losses.MeanSquaredError()
        # Scale factor for readable loss values during training.
        # Does not affect optimization dynamics (uniform scaling).
        self._loss_scale = 1000.0

    def _forward_signal_path(self, x):
        """
        Forward signal through predistorter and PA with normalization.

        NN-DPD normalizes inputs to unit power for stable network behavior.
        The scale factor is preserved to restore original power after
        predistortion.

        Parameters
        ----------
        x : tf.Tensor
            Input signal at PA rate, shape ``[batch, num_samples]``.

        Returns
        -------
        dict
            Signal path outputs:

            - ``u`` : Predistorted signal at original scale
            - ``u_norm`` : Predistorted signal at normalized scale
            - ``y_comp`` : Gain-compensated PA output
            - ``x_scale`` : Scale factor to restore original power
        """
        # Normalize input to unit power for stable NN behavior.
        x_norm, x_scale = self._normalize_to_unit_power(x)

        # Apply predistorter in normalized domain.
        u_norm = self._dpd(x_norm, training=False)

        # Restore original scale for PA input.
        u = u_norm * tf.cast(x_scale, u_norm.dtype)

        # Pass predistorted signal through PA.
        y = self._pa(u)

        # Divide by PA gain to isolate nonlinear distortion.
        y_comp = y / tf.cast(self._pa_gain, y.dtype)

        return {
            "u": u,
            "u_norm": u_norm,
            "y_comp": y_comp,
            "x_scale": x_scale,
        }

    def _training_forward(self, x):
        """
        Compute indirect learning loss for gradient-based training.

        Implements the 5-step indirect learning architecture:

        1. Apply predistorter: ``u = DPD(x_norm) * scale``
        2. Pass through PA: ``y = PA(u)``
        3. Compensate for gain: ``y_comp = y / G``
        4. Apply postdistorter: ``u_hat = DPD(y_comp_norm)``
        5. Compute loss: ``MSE(u_norm, u_hat)``

        The loss trains the network to invert the PA. After training,
        the same network serves as a predistorter.

        Parameters
        ----------
        x : tf.Tensor
            Input signal at PA rate, shape ``[batch, num_samples]``.

        Returns
        -------
        tf.Tensor
            Scalar MSE loss (scaled for readability).

        Notes
        -----
        ``tf.stop_gradient()`` on the target ``u_norm`` ensures gradients
        only flow through the postdistorter path, not the predistorter.
        This is the standard indirect learning formulation.

        The loss is computed in the normalized domain to ensure consistent
        gradient magnitudes regardless of actual signal power.
        """
        # Steps 1-3: Forward through predistorter and PA.
        signals = self._forward_signal_path(x)
        u_norm = signals["u_norm"]
        y_comp = signals["y_comp"]

        # Target is predistorter output (gradient stopped).
        # The goal is to match postdistorter and predistorter outputs.
        u_target = tf.stop_gradient(u_norm)

        # Normalize PA output for postdistorter input.
        y_norm, _ = self._normalize_to_unit_power(y_comp)

        # Step 4: Apply postdistorter (this path receives gradients).
        u_hat_norm = self._dpd(y_norm, training=True)

        # Step 5: Compute MSE loss on real/imag components.
        # Split complex into [real, imag] for standard MSE computation.
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
        Run inference to compare PA output with and without DPD.

        Parameters
        ----------
        x : tf.Tensor
            Input signal at PA rate, shape ``[batch, num_samples]``.

        Returns
        -------
        dict
            Inference outputs:

            - ``pa_input`` : Original input signal
            - ``pa_output_no_dpd`` : PA output without predistortion
            - ``pa_output_with_dpd`` : PA output with predistortion
            - ``predistorted`` : DPD output (before PA)
        """
        # Baseline: PA output without any predistortion.
        pa_output_no_dpd = self._pa(x)

        # Apply predistorter with normalization.
        x_norm, x_scale = self._normalize_to_unit_power(x)
        x_predistorted_norm = self._dpd(x_norm, training=False)
        x_predistorted = x_predistorted_norm * tf.cast(
            x_scale, x_predistorted_norm.dtype
        )

        # Pass predistorted signal through PA.
        pa_output_with_dpd = self._pa(x_predistorted)

        return {
            "pa_input": x,
            "pa_output_no_dpd": pa_output_no_dpd,
            "pa_output_with_dpd": pa_output_with_dpd,
            "predistorted": x_predistorted,
        }
