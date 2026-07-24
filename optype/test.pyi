# experimental module for testing static types

from typing import Self, final, type_check_only

__all__ = ("assert_subtype",)

@type_check_only
@final
class assert_subtype[T]:  # ruff: ignore[invalid-class-name]
    def __new__(cls, value: T, /) -> Self: ...
