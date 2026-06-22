"""Synthesize the overload variants of a single call form.

Two situations split one generic signature into separate overload lines: parameter
defaults that cannot be expressed as typevar defaults, and a presence-test on a single
parameter's attribute.
"""

from collections.abc import Mapping
from inspect import Parameter

from ._analyze import (
    absent_verdict,
    dispatch_candidates,
    requires_only_presence,
    returns_concrete,
)
from ._explore import declared_defaults, explore_spies
from ._render import (
    Defaults,
    Names,
    signatures,
    union_signature,
    widened_signature,
)
from ._spy import _AnyFunc, _TraceItem
from ._values import Exploration, map_values


def _bind(value: object, binding: Mapping[int, object]) -> object:
    """A deep copy of `value` with every bound spy replaced by its binding."""
    return map_values(value, lambda v: binding.get(id(v), v))


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
    kept = {name: spy for name, spy in spies.items() if name not in defaults}
    bound_results = [_bind(result, binding) for result in exp.results]
    return Exploration(
        kept,
        bound,
        bound_results,
        exp.var_count,
        exp.fixed,
        exp.deprecated,
        exp.gaps,
    )


def resolve_defaults(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    selected: Names,
    exploration: Exploration,
) -> tuple[Defaults, bool, list[str]]:
    """The parameter defaults if expressible as typevar defaults, else overloads.

    Omitting the defaulted parameters must behave like substituting their values
    into the generic signature; the function is rerun without them to check. On a
    mismatch the omitted calls are reported as separate overload lines, and a
    single defaulted parameter's type is excluded from the generic signature.
    """
    defaults = declared_defaults(params)
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
        omitted = explore_spies(func, params, omit=defaults)
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
            variant = explore_spies(func, params, omit={name})
        except Exception:  # noqa: BLE001, S112
            continue
        overloads += signatures(variant, params, selected, {name: value})

    return {}, False, overloads


def dispatch_overloads(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    selected: Names,
    exploration: Exploration,
    baseline: list[str],
) -> list[str]:
    """The overloads for a presence-test on a single parameter's attribute.

    Forcing the attribute absent surfaces the branch a placeholder hides. If the return
    ignores the attribute's value, one overload covers it: the parameter widens to
    `object`, the return unions both branches. If the present branch returns the value,
    that overload stays over an `object` fallback. Otherwise the `baseline` holds.
    """
    candidates = (
        dispatch_candidates(exploration.spies, exploration.traces)
        if len(params) == 1
        else ()
    )
    if len(candidates) != 1:
        return baseline
    ((param, name),) = candidates
    if not requires_only_presence(exploration.spies, exploration.traces, param, name):
        return baseline
    try:
        variant = explore_spies(func, params, absent={param: (name,)})
    except Exception:  # noqa: BLE001
        return baseline
    widens = absent_verdict(variant.spies, variant.traces, param, name)
    if widens is None:
        return baseline
    if (
        widens
        and returns_concrete(exploration.results)
        and returns_concrete(variant.results)
    ):
        return union_signature(exploration, variant, params, selected)
    tail = widened_signature(variant, params, selected)
    return baseline + tail if tail else baseline
