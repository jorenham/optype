"""
Runtime-protocols for the `dataclasses` standard library.
https://docs.python.org/3/library/dataclasses.html
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import dataclasses
    from collections.abc import Mapping


if sys.version_info >= (3, 13):
    from typing import (
        Protocol,
        TypeVar,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        Protocol,
        TypeVar,
        runtime_checkable,
    )


__all__ = ('HasDataclassFields',)


_FieldsT = TypeVar(
    '_FieldsT',
    infer_variance=True,
    bound='Mapping[str, dataclasses.Field[Any]]',
    default=dict[str, 'dataclasses.Field[Any]'],
)


@runtime_checkable
class HasDataclassFields(Protocol[_FieldsT]):
    """Can be used to check whether a type or instance is a dataclass."""
    __dataclass_fields__: _FieldsT
