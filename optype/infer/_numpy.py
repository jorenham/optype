"""NumPy-specific inference: NEP 13 ufuncs and NEP 18 `__array_function__`."""

from collections.abc import Iterable, Sequence
from inspect import Parameter, signature
from typing import cast

from ._spy import _AnyFunc

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

_PARAM_VAR = frozenset({Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD})


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


def infer_ufunc(func: _AnyFunc, names: Sequence[str], selected: Iterable[str]) -> str:
    """Render a ufunc signature from its `.types` dtype table."""
    parts: list[str] = []
    for name in selected:
        i = names.index(name)
        arms: list[str] = ["CanArrayUFunc[np.ufunc, R]"] if i == 0 else []
        if (dtype := _ufunc_dtype(func, i)) is not None:
            arms.append(dtype)
        parts.append(f"{name}: {' | '.join(arms) or 'object'}")
    return f"[R]({', '.join(parts)}) -> R"


def _required_args(func: _AnyFunc) -> int | None:
    try:
        params = signature(func).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind in _PARAM_VAR for p in params):
        return None
    return sum(
        p.kind is not Parameter.KEYWORD_ONLY and p.default is Parameter.empty
        for p in params
    )


def array_function_type(func: _AnyFunc, ret: str) -> str:
    """Render `CanArrayFunction` with the dispatched function's arity."""
    n = _required_args(func)
    args = ["Any"] * n if n is not None else ["..."]
    return f"CanArrayFunction[CanCall[{', '.join([*args, ret])}], {ret}]"


def type_name(tp: type) -> str:
    """The type's name, with the conventional `np.` prefix for numpy types."""
    name = tp.__name__
    return f"np.{name}" if tp.__module__.partition(".")[0] == "numpy" else name
