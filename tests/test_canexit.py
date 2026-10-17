"""``CanExit.__exit__`` must accept one forwarding signature.

https://github.com/jorenham/optype/issues/651
"""

from types import TracebackType
from typing import assert_type

from optype import CanWith


def forward_exit[T: BaseException](
    wrapped: CanWith[object],
    exc_type: type[T] | None,
    exc: T | None,
    tb: TracebackType | None,
) -> None:
    wrapped.__exit__(exc_type, exc, tb)


def clean_exit(wrapped: CanWith[object]) -> None:
    assert_type(wrapped.__exit__(None, None, None), None)


def error_exit(
    wrapped: CanWith[object],
    exc: BaseException,
    tb: TracebackType,
) -> None:
    wrapped.__exit__(type(exc), exc, tb)


def test_canexit_exit_helpers_exist() -> None:
    assert callable(forward_exit)
    assert callable(clean_exit)
    assert callable(error_exit)
