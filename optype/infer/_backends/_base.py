"""The render-backend interface and the helpers its implementations share.

A `Backend` turns the structured `Signature`s that `_render` builds into text: the
default `TERSE` form (`[R](x: CanAdd[Literal[1], R]) -> R`), or valid `.pyi` Python.
"""

import enum
import sys
from collections.abc import Callable, Sequence
from typing import Protocol

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]


class Backend(Protocol):
    def render(self, sigs: Sequence[_ir.Signature], /) -> str: ...


def _enum_member_path(value: enum.Enum) -> tuple[str, str, str] | None:
    """The `(module, class, member)` names of an importable enum member, or `None`.

    A composite flag has no member name, and a local or nested class is not
    importable.
    """
    if not (name := value.name or "").isidentifier():
        return None

    cls = type(value)
    if getattr(sys.modules.get(cls.__module__), cls.__name__, None) is not cls:
        return None

    return cls.__module__, cls.__name__, name


def value_text(value: object) -> str:
    """A single value's source text: an enum member's bare path, else its `repr`.

    An inexpressible enum member falls back to its data value.
    """
    if not isinstance(value, enum.Enum):
        return repr(value)

    if (found := _enum_member_path(value)) is None:
        return repr(value.value)

    _, cls, member = found
    return f"{cls}.{member}"


def qualified_value_text(value: object, rec: Callable[[str], None]) -> str:
    """`value_text` with the enum class qualified; `rec` records its import path."""
    if not isinstance(value, enum.Enum) or (found := _enum_member_path(value)) is None:
        return value_text(value)

    module, cls, member = found
    rec(path := f"{module}.{cls}")
    return f"{path}.{member}"


def default_text(value: object) -> str:
    """The default's source text, in stub style: a literal `repr`, else `...`."""
    if isinstance(value, enum.Enum):
        return value_text(value) if _enum_member_path(value) else "..."

    simple = (
        value is None
        or _ir.is_sentinel(value)  # a sentinel's repr is its declared name
        or isinstance(value, (int, float, complex, str, bytes))
    )
    return repr(value) if simple else "..."


def qualified_default_text(value: object, rec: Callable[[str], None]) -> str:
    """`default_text` with an enum member's class qualified and recorded."""
    if isinstance(value, enum.Enum):
        return qualified_value_text(value, rec) if _enum_member_path(value) else "..."
    return default_text(value)
