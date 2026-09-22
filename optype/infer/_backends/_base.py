"""The helpers the render backends share.

A backend turns the structured `Signature`s that `_render` builds into text: the
default terse form (`[R](x: CanAdd[Literal[1], R]) -> R`), or valid `.pyi` Python.
"""

import cmath
import enum
import math
import sys
from collections.abc import Callable

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]


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


type _Recorder = Callable[[str], None]


def value_text(value: object, rec: _Recorder | None = None) -> str:
    """A single value's source text: an enum member's path, else its `repr`.

    An inexpressible enum member falls back to its data value. With `rec`, the enum
    class is module-qualified and `rec` records that import path.
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


def default_text(value: object, rec: _Recorder | None = None) -> str:
    """The default's source text, in stub style: a literal `repr`, else `...`.

    With `rec`, an enum member's class is qualified and recorded, as in `value_text`.
    """
    if isinstance(value, enum.Enum):
        return value_text(value, rec) if _enum_member_path(value) else "..."

    simple = (
        value is None
        or _ir.is_sentinel(value)  # a sentinel's repr is its declared name
        or isinstance(value, (int, str, bytes))
        or (isinstance(value, float) and math.isfinite(value))
        or (isinstance(value, complex) and cmath.isfinite(value))
    )
    return repr(value) if simple else "..."
