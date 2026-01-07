# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

from importlib import import_module

__all__ = [
    "Config",
    "CIRManager",
    "CIRGenerator",
    "PUSCHTrainableTransmitter",
    "PUSCHNeuralDetector",
    "PUSCHTrainableReceiver",
    "System",
]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module(f".{name.lower()}", __name__), name)
    raise AttributeError(name)
