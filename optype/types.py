from __future__ import annotations

import sys
from typing import Literal


if sys.version_info >= (3, 13):
    from typing import LiteralString, Protocol, runtime_checkable
else:
    from typing_extensions import LiteralString, Protocol, runtime_checkable

from ._types_impl import AnnotatedAlias, GenericType, LiteralAlias, UnionAlias


__all__ = (
    'AnnotatedAlias',
    'GenericType',
    'LiteralAlias',
    'ProtocolType',
    'RuntimeProtocolType',
    'UnionAlias',
    'WrappedFinalType',
)


@runtime_checkable
class WrappedFinalType(Protocol):
    """
    A runtime-protocol that represents a type or method that's decorated with
    `@typing.final` or `@typing_extensions.final`.

    Note that the name `HasFinal` isn't used, because `__final__` is
    undocumented, and therefore not a part of the public API.
    """
    __final__: Literal[True]


@runtime_checkable
class ProtocolType(Protocol):
    _is_protocol: Literal[True]

    if sys.version_info >= (3, 12, 0):
        __protocol_attrs__: set[LiteralString]


@runtime_checkable
class RuntimeProtocolType(ProtocolType, Protocol):
    _is_runtime_protocol: Literal[True]

    if sys.version_info >= (3, 12, 2):
        __non_callable_proto_members__: set[LiteralString]
