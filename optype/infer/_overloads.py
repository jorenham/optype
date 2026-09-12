"""Synthesize the overload variants of a single call form.

Two situations split one generic signature into separate overload lines: parameter
defaults that cannot be expressed as typevar defaults, and a presence-test on a single
parameter's attribute.
"""

from collections.abc import Iterable, Mapping
from inspect import Parameter
from typing import NamedTuple

from ._analyze import (
    absent_verdict,
    dispatch_candidates,
    requires_only_presence,
    returns_concrete,
)
from ._explore import declared_defaults, explore_spies
from ._ir import Signature
from ._render import (
    Defaults,
    Names,
    render_all,
    renderers_of,
    signatures,
    union_signatures,
    widened_signatures,
)
from ._spy import _AnyFunc, _TraceItem
from ._values import Exploration, map_values


class _ResolvedDefaults(NamedTuple):
    defaults: Defaults
    negate: bool
    overloads: list[Signature]


def _bind(value: object, binding: Mapping[int, object]) -> object:
    """A deep copy of `value` with every bound spy replaced by its binding."""
    return map_values(value, lambda v: binding.get(id(v), v))


def _distinct(sigs: Iterable[Signature]) -> list[Signature]:
    # no set: a default value may be unhashable
    sigs = list(sigs)
    return [sig for i, sig in enumerate(sigs) if sig not in sigs[:i]]


def _bind_exploration(exp: Exploration, defaults: Defaults) -> Exploration:
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
    return exp._replace(
        spies={name: spy for name, spy in spies.items() if name not in defaults},
        traces=bound,
        results=[_bind(result, binding) for result in exp.results],
        tuple_params=exp.tuple_params - set(defaults),
    )


def resolve_defaults(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    selected: Names,
    exploration: Exploration,
) -> _ResolvedDefaults:
    """The parameter defaults if expressible as typevar defaults, else overloads.

    Omitting the defaulted parameters must behave like substituting their values
    into the generic signature; the function is rerun without them to check. On a
    mismatch the omitted calls are reported as separate overloads, and a single
    defaulted parameter's type is excluded from the generic signature.
    """
    defaults = declared_defaults(params)
    kinds = {p.kind for p in params.values()}
    if not defaults or (
        # `*args` placeholders would positionally fill an omitted default
        Parameter.VAR_POSITIONAL in kinds
        and any(params[n].kind is not Parameter.KEYWORD_ONLY for n in defaults)
    ):
        return _ResolvedDefaults({}, False, [])

    required = {name: p for name, p in params.items() if name not in defaults}
    names = list(required)

    try:
        omitted = explore_spies(func, params, omit=defaults)
        omitted_renderers = renderers_of(omitted, params)
        # the comparison must see every required parameter, regardless of selection
        observed = render_all(omitted_renderers, names, deprecated=omitted.deprecated)
    except Exception:  # ruff: ignore[blind-except]
        return _ResolvedDefaults({}, False, [])

    omitted_defaults = _bind_exploration(exploration, defaults)
    expected = signatures(omitted_defaults, required, names)
    if _distinct(expected) == _distinct(observed):
        return _ResolvedDefaults(defaults, False, [])

    overloads = render_all(
        omitted_renderers,
        selected,
        defaults,
        deprecated=omitted.deprecated,
    )

    if len(defaults) == 1:
        return _ResolvedDefaults(defaults, True, overloads)

    for name, value in defaults.items():
        try:
            variant = explore_spies(func, params, omit={name})
        except Exception:  # ruff: ignore[blind-except, try-except-continue]
            continue
        overloads += signatures(variant, params, selected, {name: value})

    return _ResolvedDefaults({}, False, overloads)


def dispatch_overloads(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    selected: Names,
    exploration: Exploration,
    baseline: list[Signature],
) -> list[Signature]:
    """The overloads for a presence-test on a single parameter's attribute.

    Forcing the attribute absent surfaces the branch a placeholder hides. If the return
    ignores the attribute's value, one overload covers it: the parameter widens to
    `object`, the return unions both branches. If the present branch returns the value,
    that overload stays over an `object` fallback. Otherwise the `baseline` holds.
    """
    candidates = dispatch_candidates(exploration) if len(params) == 1 else ()
    if len(candidates) != 1:
        return baseline
    ((param, name),) = candidates
    if not requires_only_presence(exploration, param, name):
        return baseline
    try:
        variant = explore_spies(func, params, absent={param: (name,)})
    except Exception:  # ruff: ignore[blind-except]
        return baseline
    widens = absent_verdict(variant, param, name)
    if widens is None:
        return baseline
    if (
        widens
        and returns_concrete(exploration.results)
        and returns_concrete(variant.results)
    ):
        return union_signatures(exploration, variant, params, selected)
    tail = widened_signatures(variant, params, selected)
    return baseline + tail if tail else baseline
