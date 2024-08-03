# noqa: A005
from __future__ import annotations

import sys
from typing import ClassVar, Literal


if sys.version_info >= (3, 13):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


__all__ = ('FinalType',)


@runtime_checkable
class FinalType(Protocol):
    """
    A runtime-protocol that represents a type or method that's decorated with
    `@typing.final` or `@typing_extensions.final`.

    Note that the name `HasFinal` isn't used, because `__final__` is
    undocumented, and therefore not a part of the public API.
    """
    __final__: ClassVar[Literal[True]]
