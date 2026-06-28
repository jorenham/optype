"""The `optype.infer` exceptions."""

from pathlib import Path

__all__ = ("InferError", "InferWarning")

# the infer package dir, skipped when attributing an `InferWarning` to user code
WARN_SKIP_PREFIX = str(Path(__file__).parent)


class InferError(NotImplementedError):
    """Raised when `infer` does not support the given function."""


class InferWarning(RuntimeWarning):
    """Emitted when `infer` could not explore the function exhaustively."""
