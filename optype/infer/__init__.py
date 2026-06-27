"""Structurally infer the `optype` protocols required by a function."""

from ._api import infer
from ._backend import TERSE, Backend
from ._errors import InferError, InferWarning

__all__ = ("TERSE", "Backend", "InferError", "InferWarning", "infer")
