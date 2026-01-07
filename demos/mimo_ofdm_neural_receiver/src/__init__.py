# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

"""
MIMO-OFDM Neural Receiver source package.

Exports: Config, CSI, Tx, Channel, Rx, NeuralRx, System.

Uses lazy imports to defer TensorFlow/Sionna loading until first access.
"""

from importlib import import_module

__all__ = ["Config", "CSI", "Tx", "Channel", "Rx", "NeuralRx", "System"]


def __getattr__(name):
    """Lazy import: class Foo is loaded from module foo.py on first access."""
    if name in __all__:
        return getattr(import_module(f".{name.lower()}", __name__), name)
    raise AttributeError(name)
