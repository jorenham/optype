# experimental module for testing static types

from collections.abc import Callable
from typing import Self, assert_type, final

__all__ = ("UncheckedWarning", "assert_subtype", "assert_type")

class UncheckedWarning(UserWarning): ...

@final
class assert_subtype[T]:  # ruff: ignore[invalid-class-name]
    def __new__(cls, value: T, /) -> Self: ...
    @classmethod
    def __class_getitem__(cls, typ: object, /) -> Callable[[object], Self]: ...
