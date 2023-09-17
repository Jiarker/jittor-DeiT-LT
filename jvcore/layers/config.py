"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
from typing import Any, Optional

__all__ = [
    'is_exportable', 'is_scriptable', 'is_no_jit',
    'set_exportable', 'set_scriptable', 'set_no_jit', 'set_layer_config'
]

# Set to True if prefer to have layers with no jit optimization (includes activations)
_NO_JIT = False

# Set to True if prefer to have activation layers with no jit optimization
# NOTE not currently used as no difference between no_jit and no_activation jit as only layers obeying
# the jit flags so far are activations. This will change as more layers are updated and/or added.
_NO_ACTIVATION_JIT = False

# Set to True if exporting a model with Same padding via ONNX
_EXPORTABLE = False

# Set to True if wanting to use torch.jit.script on a model
_SCRIPTABLE = False


def is_no_jit():
    return _NO_JIT


class set_no_jit:
    def __init__(self, mode: bool) -> None:
        global _NO_JIT
        self.prev = _NO_JIT
        _NO_JIT = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _NO_JIT
        _NO_JIT = self.prev
        return False


def is_exportable():
    return _EXPORTABLE


class set_exportable:
    def __init__(self, mode: bool) -> None:
        global _EXPORTABLE
        self.prev = _EXPORTABLE
        _EXPORTABLE = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _EXPORTABLE
        _EXPORTABLE = self.prev
        return False


def is_scriptable():
    return _SCRIPTABLE


class set_scriptable:
    def __init__(self, mode: bool) -> None:
        global _SCRIPTABLE
        self.prev = _SCRIPTABLE
        _SCRIPTABLE = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _SCRIPTABLE
        _SCRIPTABLE = self.prev
        return False
