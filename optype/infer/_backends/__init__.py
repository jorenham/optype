"""The render backends, each a function from signatures to text."""

from collections.abc import Callable, Sequence
from typing import Final, Literal

from . import _compat, _terse
from optype.infer._ir import Signature

type BackendName = Literal["terse", "compat"]

BACKENDS: Final[dict[BackendName, Callable[[Sequence[Signature]], str]]] = {
    "terse": _terse.render,
    "compat": _compat.render,
}
