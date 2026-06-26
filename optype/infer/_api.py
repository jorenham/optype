"""The `infer` entry point: explore, probe the defaults, render."""

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from inspect import Parameter, signature

# `from . import` would import the package itself, which imports this module
import optype.infer._numpy as _numpy
from ._errors import InferError, InferWarning
from ._explore import explore_lenient, explore_tuple_params
from ._overloads import dispatch_overloads, resolve_defaults
from ._render import Names, signatures
from ._signature import probe_signatures
from ._spy import _AnyFunc
from ._values import GapKind


@dataclass(frozen=True, slots=True)
class _Gap:
    """A coverage gap at a call form."""

    kind: GapKind
    where: str  # the call form, e.g. "(x, y)"

    def message(self) -> str:
        return f"{self.kind} in {self.where}"


class _SelectError(ValueError):
    """A selected parameter or position is absent from the signature."""


def _select(params: Iterable[str | int], names: Names) -> Names:
    selected: list[str] = []
    for p in params:
        if isinstance(p, int):
            if not -len(names) <= p < len(names):
                msg = f"no parameter at position {p}"
                raise _SelectError(msg)
            selected.append(names[p])
        elif p in names:
            selected.append(p)
        else:
            msg = f"unknown parameter {p!r}"
            raise _SelectError(msg)
    return selected or names


def _form_signatures(
    func: _AnyFunc,
    parameters: Mapping[str, Parameter],
    params: tuple[str | int, ...],
    gaps: set[_Gap],
) -> list[str]:
    selected = _select(params, list(parameters))
    where = f"({', '.join(parameters)})"
    try:
        exploration, fallback = explore_lenient(func, parameters)
    except (IndexError, TypeError, ValueError) as exc:
        raise InferError(str(exc)) from exc
    gaps.update(_Gap(kind, where) for kind in exploration.gaps)
    if fallback:
        # the rejected parameters render from their defaults; skip the probing
        lines = signatures(exploration, parameters, selected, fallback)
        return list(dict.fromkeys(lines))
    exploration = exploration._replace(
        tuple_params=explore_tuple_params(func, parameters, exploration),
    )
    defaults, negate, overloads = resolve_defaults(
        func,
        parameters,
        selected,
        exploration,
    )
    lines = signatures(exploration, parameters, selected, defaults, negate=negate)
    if not defaults and not overloads:
        # dispatch only when defaults didn't already split the form
        lines = dispatch_overloads(func, parameters, selected, exploration, lines)
    return list(dict.fromkeys((*overloads, *lines)))


def _candidate_parameters(func: _AnyFunc) -> list[dict[str, Parameter]]:
    try:
        return [dict(signature(func).parameters)]
    except TypeError as exc:  # not callable
        raise InferError(str(exc)) from exc
    except ValueError as exc:  # callable but no signature (e.g. a C builtin)
        if (probed := probe_signatures(func)) is None:
            raise InferError(str(exc)) from exc
        return probed


def _infer(func: _AnyFunc, params: tuple[str | int, ...], gaps: set[_Gap]) -> str:
    if nin := _numpy.ufunc_nin(func):
        names = _numpy.ufunc_params(nin)
        return _numpy.infer_ufunc(func, names, _select(params, names))

    lines: list[str] = []
    last: Exception | None = None
    for parameters in _candidate_parameters(func):
        try:
            lines += _form_signatures(func, parameters, params, gaps)
        except (InferError, ValueError) as exc:
            # a candidate arity may fail exploration (e.g. `int`'s base); drop it,
            # re-raising below only if no candidate produced anything
            last = exc
    if not lines and last is not None:
        raise last
    return "\n".join(dict.fromkeys(lines))


def infer(func: _AnyFunc, /, *params: str | int, strict: bool = False) -> str:
    """Infer the `optype` protocol(s) required of `func`'s parameters.

    Pass parameter names or positions to report only those parameters.

    >>> print(infer(lambda x: x + 1))
    [R](x: CanAdd[Literal[1], R]) -> R

    When exploration is incomplete it emits an `InferWarning` naming the affected
    call form; pass `strict=True` to raise an `InferError` instead.

    Raises:
        InferError: If `func` is not supported, such as a non-callable, an
            operation without a matching protocol, or a parameter that requires
            a value that no placeholder can provide; or, with `strict=True`, if
            the function could not be explored exhaustively.
    """
    gaps: set[_Gap] = set()
    try:
        result = _infer(func, params, gaps)
    except RecursionError as exc:
        raise InferError("the result is nested too deeply") from exc
    if gaps:
        detail = "; ".join(sorted(g.message() for g in gaps))
        msg = f"incomplete exploration: {detail}"
        if strict:
            raise InferError(msg)
        warnings.warn(msg, InferWarning, stacklevel=2)
    return result
