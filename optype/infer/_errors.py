"""The `optype.infer` exceptions."""

__all__ = ("InferError", "InferWarning")


class InferError(NotImplementedError):
    """Raised when `infer` does not support the given function."""


class InferWarning(RuntimeWarning):
    """Emitted when `infer` could not explore the function exhaustively."""
