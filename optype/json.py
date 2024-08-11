"""
Type aliases for `json` standard library.
This assumes that the default encoder and decoder are used.
"""

from __future__ import annotations

import sys
from types import MappingProxyType
from typing import TypeAlias


if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

__all__ = (
    'AnyArray',
    'AnyObject',
    'AnyValue',
    'Array',
    'Object',
    'Value',
)

_Primitive: TypeAlias = None | bool | int | float | str

# Return types of `json.load[s]`
Value: TypeAlias = _Primitive | dict[str, 'Value'] | list['Value']
_VT = TypeVar('_VT', bound=Value, default=Value)
Array: TypeAlias = list[_VT]
Object: TypeAlias = dict[str, _VT]

# Input types of `json.dumps`
AnyValue: TypeAlias = (
    _Primitive
    # NOTE: `TypedDict` can't be included here, since it's not a sub*type* of
    # `dict[str, Any]` according to the typing docs and typeshed, even though
    # it **literally** is a subclass of `dict`...
    | dict[str, 'AnyValue']
    | MappingProxyType[str, 'AnyValue']
    | list['AnyValue']
    | tuple['AnyValue', ...]
)
_AVT = TypeVar('_AVT', bound=AnyValue, default=AnyValue)
AnyArray: TypeAlias = list[_AVT] | tuple[_AVT, ...]
AnyObject: TypeAlias = dict[str, _AVT]
