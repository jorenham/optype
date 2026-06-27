"""The render-backend interface and the helpers its implementations share.

A `Backend` turns the structured `Signature`s that `_render` builds into text: the
default `TERSE` form (`[R](x: CanAdd[Literal[1], R]) -> R`), or valid `.pyi` Python.
"""

from collections.abc import Sequence
from typing import Protocol

# `from optype.infer import _ir` would re-enter the package, which imports this module
import optype.infer._ir as _ir  # noqa: PLR0402


class Backend(Protocol):
    def render(self, sigs: Sequence[_ir.Signature], /) -> str: ...


def default_text(value: object) -> str:
    """The default's source text, in stub style: a literal `repr`, else `...`."""
    simple = (
        value is None
        or _ir.is_sentinel(value)  # a sentinel's repr is its declared name
        or isinstance(value, (int, float, complex, str, bytes))
    )
    return repr(value) if simple else "..."
