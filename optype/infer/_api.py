"""The `infer` entry point: explore, probe the defaults, render."""

import warnings
from collections.abc import Iterable, Mapping
from inspect import Parameter

# `from . import` would import the package itself, which imports this module
import optype.infer._numpy as _numpy
from ._errors import InferError, InferWarning
from ._explore import (
    _declared_defaults,
    _Exploration,
    _explore_lenient,
    _explore_spies,
    _Gap,
    _parameter_forms,
)
from ._render import _Defaults, _Names, signatures
from ._spy import _AnyFunc, _TraceItem
from ._values import map_values


class _SelectError(ValueError):
    """A selected parameter or position is absent from a particular call form."""


def _select(params: Iterable[str | int], names: _Names) -> _Names:
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


def _bind(value: object, binding: Mapping[int, object]) -> object:
    """A deep copy of `value` with every bound spy replaced by its binding."""
    return map_values(value, lambda v: binding.get(id(v), v))


def _bind_exploration(exp: _Exploration, defaults: _Defaults) -> _Exploration:
    """The exploration as it would look with every defaulted parameter omitted."""
    spies = exp.spies
    binding = {id(spies[name]): value for name, value in defaults.items()}
    # so that a `type(spy)` result becomes `type(default)`
    binding |= {id(type(spies[name])): type(value) for name, value in defaults.items()}
    bound = {
        spy_id: [
            _TraceItem(
                item.attr,
                tuple(_bind(arg, binding) for arg in item.args),
                {key: _bind(val, binding) for key, val in item.kwargs.items()},
                item.return_,
            )
            for item in items
        ]
        for spy_id, items in exp.traces.items()
    }
    kept = {name: spy for name, spy in spies.items() if name not in defaults}
    bound_results = [_bind(result, binding) for result in exp.results]
    return _Exploration(
        kept,
        bound,
        bound_results,
        exp.var_count,
        exp.fixed,
        exp.deprecated,
        exp.gaps,
    )


def _defaults(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    selected: _Names,
    exploration: _Exploration,
) -> tuple[_Defaults, bool, list[str]]:
    """The parameter defaults if expressible as typevar defaults, else overloads.

    Omitting the defaulted parameters must behave like substituting their values
    into the generic signature; the function is rerun without them to check. On a
    mismatch the omitted calls are reported as separate overload lines, and a
    single defaulted parameter's type is excluded from the generic signature.
    """
    defaults = _declared_defaults(params)
    kinds = {p.kind for p in params.values()}
    if not defaults or (
        # `*args` placeholders would positionally fill an omitted default
        Parameter.VAR_POSITIONAL in kinds
        and any(params[n].kind is not Parameter.KEYWORD_ONLY for n in defaults)
    ):
        return {}, False, []

    required = {name: p for name, p in params.items() if name not in defaults}
    names = list(required)

    try:
        omitted = _explore_spies(func, params, omit=defaults)
        # the comparison must see every required parameter, regardless of selection
        observed = signatures(omitted, required, names)
    except Exception:  # noqa: BLE001
        return {}, False, []

    omitted_defaults = _bind_exploration(exploration, defaults)
    if signatures(omitted_defaults, required, names) == observed:
        return defaults, False, []

    overloads = signatures(omitted, params, selected, defaults)

    if len(defaults) == 1:
        return defaults, True, overloads

    for name, value in defaults.items():
        try:
            variant = _explore_spies(func, params, omit={name})
        except Exception:  # noqa: BLE001, S112
            continue
        overloads += signatures(variant, params, selected, {name: value})

    return {}, False, overloads


def _infer_form(
    func: _AnyFunc,
    parameters: Mapping[str, Parameter],
    params: tuple[str | int, ...],
    gaps: set[_Gap],
) -> list[str]:
    selected = _select(params, list(parameters))
    try:
        exploration, fallback = _explore_lenient(func, parameters)
    except (IndexError, TypeError, ValueError) as exc:
        raise InferError(str(exc)) from exc
    if exploration.gaps:
        where = f"({', '.join(parameters)})"
        gaps.update(_Gap(g.kind, g.where or where) for g in exploration.gaps)
    if fallback:
        # the rejected parameters render from their defaults; skip the probing
        lines = signatures(exploration, parameters, selected, fallback)
        return list(dict.fromkeys(lines))
    defaults, negate, overloads = _defaults(func, parameters, selected, exploration)
    lines = signatures(exploration, parameters, selected, defaults, negate=negate)
    return list(dict.fromkeys((*overloads, *lines)))


def _infer(func: _AnyFunc, params: tuple[str | int, ...], gaps: set[_Gap]) -> str:
    if nin := _numpy.ufunc_nin(func):
        names = _numpy.ufunc_params(nin)
        return _numpy.infer_ufunc(func, names, _select(params, names))

    forms = _parameter_forms(func)
    if len(forms) == 1:
        # a lone form's error is the result's; the loop below only tolerates a failed
        # form because another may still match
        return "\n".join(_infer_form(func, forms[0], params, gaps))

    # one overload per documented form: an unsatisfiable form is dropped and a
    # `_SelectError` filters one out, but an unexpected error still propagates
    lines: list[str] = []
    reasons: list[str] = []
    misses: list[str] = []
    for parameters in forms:
        try:
            lines += _infer_form(func, parameters, params, gaps)
        except _SelectError as exc:
            misses.append(str(exc))
        except InferError as exc:
            reasons.append(str(exc))
    if lines:
        return "\n".join(dict.fromkeys(lines))
    if reasons:
        raise InferError("; ".join(dict.fromkeys(reasons)))
    if misses:
        raise _SelectError("; ".join(dict.fromkeys(misses)))
    raise InferError("no inferable call form")


def infer(func: _AnyFunc, /, *params: str | int, strict: bool = False) -> str:
    """Infer the `optype` protocol(s) required of `func`'s parameters.

    Pass parameter names or positions to report only those parameters.

    >>> print(infer(lambda x: x + 1))
    [R](x: CanAdd[Literal[1], R]) -> R

    Because `infer` runs the function against placeholders, it can only observe
    the paths those placeholders drive. When exploration is incomplete it emits an
    `InferWarning` naming the affected call form; pass `strict=True` to raise an
    `InferError` instead of returning a provisional signature.

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
        if strict:
            msg = f"not exhaustively explored: {detail}"
            raise InferError(msg)
        msg = f"not every branch was explored: {detail}"
        warnings.warn(msg, InferWarning, stacklevel=2)
    return result
