"""The `infer` entry point: explore, probe the defaults, render."""

import builtins
import sys
from collections.abc import Collection, Iterable, Mapping
from inspect import Parameter
from typing import cast

# `from . import` would import the package itself, which imports this module
import optype.infer._numpy as _numpy
from ._errors import InferError
from ._explore import (
    _declared_defaults,
    _explore_lenient,
    _explore_spies,
    _parameters,
    _Recon,
)
from ._render import _Defaults, _Names, signatures
from ._spy import _AnyFunc, _TraceItem
from ._values import _Fn, _Gen


def _select(params: Iterable[str | int], names: _Names) -> _Names:
    selected: list[str] = []
    for p in params:
        if isinstance(p, int):
            if not -len(names) <= p < len(names):
                msg = f"no parameter at position {p}"
                raise ValueError(msg)
            selected.append(names[p])
        elif p in names:
            selected.append(p)
        else:
            msg = f"unknown parameter {p!r}"
            raise ValueError(msg)
    return selected or names


def _bind(value: object, binding: Mapping[int, object]) -> object:
    """A deep copy of `value` with every bound spy replaced by its binding."""
    if sys.version_info >= (3, 15):
        # this getattr is workaround for pyrefly (1.0.0)
        frozendict_ = getattr(builtins, "frozendict")  # noqa: B009
        if isinstance(value, frozendict_):
            return frozendict_({
                _bind(k, binding): _bind(v, binding)
                # this cast is a workaround for pyright  (1.1.410)
                for k, v in cast("Mapping[object, object]", value).items()
            })

    match value:
        case _Gen():
            yielded = [_bind(item, binding) for item in value.yielded]
            out: object = value._replace(yielded=yielded)
        case _Fn():
            bound = [_bind(item, binding) for item in value.results]
            out = value._replace(results=bound)
        case tuple() if type(value) is tuple:  # pyright: ignore[reportUnknownArgumentType]
            tup = cast("tuple[object, ...]", value)
            out = tuple(_bind(item, binding) for item in tup)
        case list():
            out = [_bind(item, binding) for item in cast("list[object]", value)]
        case set() | frozenset():
            items = {_bind(item, binding) for item in cast("Collection[object]", value)}
            out = frozenset(items) if isinstance(value, frozenset) else items
        case dict():
            mapping = cast("Mapping[object, object]", value)
            out = {_bind(k, binding): _bind(v, binding) for k, v in mapping.items()}
        case _:
            out = binding.get(id(value), value)
    return out


def _bind_recon(recon: _Recon, defaults: _Defaults) -> _Recon:
    """The recon as it would look with every defaulted parameter omitted."""
    spies, traces, results, count, fixed = recon
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
        for spy_id, items in traces.items()
    }
    kept = {name: spy for name, spy in spies.items() if name not in defaults}
    return kept, bound, [_bind(result, binding) for result in results], count, fixed


def _defaults(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    selected: _Names,
    recon: _Recon,
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

    if signatures(_bind_recon(recon, defaults), required, names) == observed:
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


def infer(func: _AnyFunc, /, *params: str | int) -> str:
    """Infer the `optype` protocol(s) required of `func`'s parameters.

    Pass parameter names or positions to report only those parameters.

    >>> print(infer(lambda x: x + 1))
    [R](x: CanAdd[Literal[1], R]) -> R

    Raises:
        InferError: If `func` is not supported, such as a non-callable, an
            operation without a matching protocol, or a parameter that requires
            a value that no placeholder can provide.
    """
    if nin := _numpy.ufunc_nin(func):
        names = _numpy.ufunc_params(nin)
        return _numpy.infer_ufunc(func, names, _select(params, names))

    parameters = _parameters(func)
    selected = _select(params, list(parameters))
    try:
        recon, fallback = _explore_lenient(func, parameters)
    except (IndexError, TypeError, ValueError) as exc:
        raise InferError(str(exc)) from exc
    if fallback:
        # the rejected parameters render from their defaults; skip the probing
        lines = signatures(recon, parameters, selected, fallback)
        return "\n".join(dict.fromkeys(lines))
    defaults, negate, overloads = _defaults(func, parameters, selected, recon)
    lines = signatures(recon, parameters, selected, defaults, negate=negate)
    return "\n".join(dict.fromkeys((*overloads, *lines)))
