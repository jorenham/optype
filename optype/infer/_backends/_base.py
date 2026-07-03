"""The render-backend interface and the helpers its implementations share.

A `Backend` turns the structured `Signature`s that `_render` builds into text: the
default `TERSE` form (`[R](x: CanAdd[Literal[1], R]) -> R`), or valid `.pyi` Python.
"""

import enum
import sys
from collections.abc import Callable, Sequence
from typing import Protocol

# `from optype.infer import _ir` would re-enter the package, which imports this module
import optype.infer._ir as _ir  # noqa: PLR0402


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


def value_text(value: object, rec: Callable[[str], None] | None = None) -> str:
    """A single value's source text: an enum member's path, else its `repr`.

    A `rec` records the enum class path to import and selects the qualified form;
    an inexpressible enum member falls back to its data value.
    """
    if not isinstance(value, enum.Enum):
        return repr(value)

    if (found := _enum_member_path(value)) is None:
        return repr(value.value)

    module, cls, member = found
    if rec is None:
        return f"{cls}.{member}"

    rec(path := f"{module}.{cls}")
    return f"{path}.{member}"


def default_text(value: object, rec: Callable[[str], None] | None = None) -> str:
    """The default's source text, in stub style: a literal `repr`, else `...`."""
    if isinstance(value, enum.Enum):
        return value_text(value, rec) if _enum_member_path(value) else "..."

    simple = (
        value is None
        or _ir.is_sentinel(value)  # a sentinel's repr is its declared name
        or isinstance(value, (int, float, complex, str, bytes))
    )
    return repr(value) if simple else "..."
