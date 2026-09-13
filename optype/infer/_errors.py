"""The `optype.infer` exceptions."""

from pathlib import Path

# the infer package dir, skipped when attributing an `InferWarning` to user code
WARN_SKIP_PREFIX = str(Path(__file__).parent)


class InferError(NotImplementedError):
    """Raised when `infer` does not support the given function."""


class InferWarning(RuntimeWarning):
    """Emitted when `infer` could not explore the function exhaustively."""


_MESSAGE_LIMIT = 200


def describe(exc: BaseException, /) -> str:
    """The message of `exc` cut to a readable length, or its type name when empty."""
    text = str(exc)
    if len(text) > _MESSAGE_LIMIT:
        text = f"{text[:_MESSAGE_LIMIT]} [...]"
    return text or type(exc).__name__
