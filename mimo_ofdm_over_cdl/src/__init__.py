from importlib import import_module

__all__ = ["Config", "CSI", "Tx", "Channel", "Rx"]

def __getattr__(name):
    if name in __all__:
        return getattr(import_module(f".{name.lower()}", __name__), name)
    raise AttributeError(name)
