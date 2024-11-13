"""
Runtime-protocols for the `dataclasses` standard library.
https://docs.python.org/3/library/dataclasses.html
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol


if sys.version_info >= (3, 13):
    from typing import TypeVar, runtime_checkable
else:
    from typing_extensions import TypeVar, runtime_checkable


if TYPE_CHECKING:
    import dataclasses
    from collections.abc import Mapping


__all__ = ("HasDataclassFields",)


def __dir__() -> tuple[str, ...]:
    return __all__


_FieldsT = TypeVar(
    "_FieldsT",
    bound="Mapping[str, dataclasses.Field[object]]",
    default=dict[str, "dataclasses.Field[object]"],
)


@runtime_checkable
class HasDataclassFields(Protocol[_FieldsT]):
    """Can be used to check whether a type or instance is a dataclass."""

    __dataclass_fields__: _FieldsT
