"""Structurally infer the `optype` protocols required by a function."""

from ._api import infer
from ._errors import InferError, InferWarning

__all__ = ("InferError", "InferWarning", "infer")
