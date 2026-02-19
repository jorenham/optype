# experimental module for testing static types

from typing import Generic, Self, TypeVar, final, type_check_only

__all__ = ("assert_subtype",)

_T = TypeVar("_T")

@type_check_only
@final
class assert_subtype(Generic[_T]):  # noqa: N801
    def __new__(cls, value: _T, /) -> Self: ...
