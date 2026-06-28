"""NumPy-specific inference: NEP 13 ufuncs and NEP 18 `__array_function__`."""

from collections.abc import Iterable, Sequence
from inspect import Parameter, signature
from typing import cast

# `from . import` would import the package itself, which imports this module
import optype.infer._ir as _ir
from ._spy import _AnyFunc
from ._values import VARIADIC_KINDS

DUNDER_CAN_MAP = {
    "__array_ufunc__": "CanArrayUFunc",
    "__array_function__": "CanArrayFunction",
}

_DTYPE_KINDS = (
    ("?", "ToBoolND"),
    ("bhilqpBHILQP", "ToIntND"),
    ("efdg", "ToFloatND"),
    ("FDG", "ToComplexND"),
)
_DTYPE_RANK: dict[str, int] = {
    char: rank for rank, (chars, _) in enumerate(_DTYPE_KINDS) for char in chars
}
_DTYPE_ALIASES = [alias for _, alias in _DTYPE_KINDS]


def ufunc_nin(func: _AnyFunc) -> int | None:
    """The input arity if `func` looks like a ufunc, `None` otherwise."""
    nin, nout = getattr(func, "nin", None), getattr(func, "nout", None)
    if isinstance(nin, int) and isinstance(nout, int):
        return nin
    return None


def ufunc_params(nin: int) -> list[str]:
    """The conventional ufunc parameter names: `x`, or `x1`, `x2`, ..."""
    return ["x"] if nin == 1 else [f"x{i + 1}" for i in range(nin)]


def _ufunc_dtype(func: _AnyFunc, i: int) -> str | None:
    types = cast("Sequence[str]", getattr(func, "types", ()))
    ranks = [
        _DTYPE_RANK[ins[i]]
        for sig in types
        if i < len(ins := sig.partition("->")[0]) and ins[i] in _DTYPE_RANK
    ]
    return _DTYPE_ALIASES[max(ranks)] if ranks else None


def infer_ufunc(
    func: _AnyFunc,
    names: Sequence[str],
    selected: Iterable[str],
) -> list[_ir.Signature]:
    """The ufunc signature from its `.types` dtype table."""
    params: list[_ir.Param] = []
    for name in selected:
        i = names.index(name)
        arms: list[_ir.Node] = (
            [_ir.App("CanArrayUFunc", (_ir.Name("np.ufunc"), _ir.Name("R")))]
            if i == 0
            else []
        )
        if dtype := _ufunc_dtype(func, i):
            arms.append(_ir.Name(dtype))
        params.append(_ir.Param(name, _ir.union(arms) or _ir.Name("object")))
    return [_ir.Signature((_ir.TypeParam("R"),), tuple(params), _ir.Name("R"))]


def _required_args(func: _AnyFunc) -> int | None:
    """The positional arity of `func`, or `None` if it has none or is variadic."""
    try:
        params = signature(func).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind in VARIADIC_KINDS for p in params):
        return None
    return sum(
        p.kind is not Parameter.KEYWORD_ONLY and p.default is Parameter.empty
        for p in params
    )


def array_function_node(func: _AnyFunc, ret: _ir.Node) -> _ir.Node:
    """Render `CanArrayFunction` with the dispatched function's arity."""
    n = _required_args(func)
    sig: tuple[_ir.Node, ...] = (
        (_ir.Dots(),) if n is None else tuple(_ir.Name("Any") for _ in range(n))
    )
    return _ir.App("CanArrayFunction", (_ir.Fn(sig, ret), ret))
