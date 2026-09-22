"""Synthesize the overload variants of a single call form.

Two situations split one generic signature into separate overload lines: parameter
defaults that cannot be expressed as typevar defaults, and a presence-test on a single
parameter's attribute.
"""

from collections.abc import Iterable, Mapping
from inspect import Parameter
from itertools import starmap

from ._analyze import (
    absent_verdict,
    dispatch_candidates,
    requires_only_presence,
    returns_concrete,
)
from ._explore import declared_defaults, explore_spies
from ._ir import Signature, alpha_equal_signatures
from ._render import Defaults, Names, signatures, widened_signatures
from ._spy import AnyFunc, TraceItem
from ._values import Exploration, map_values


def _distinct(sigs: Iterable[Signature]) -> list[Signature]:
    out: list[Signature] = []
    for sig in sigs:
        if not any(alpha_equal_signatures(sig, seen) for seen in out):
            out.append(sig)
    return out


def _same(expected: Iterable[Signature], observed: Iterable[Signature]) -> bool:
    """Whether the two renders agree, up to how their type parameters are named."""
    expected, observed = _distinct(expected), _distinct(observed)
    return len(expected) == len(observed) and all(
        starmap(alpha_equal_signatures, zip(expected, observed, strict=True)),
    )


def _bind_exploration(exp: Exploration, defaults: Defaults) -> Exploration:
    """The exploration as it would look with every defaulted parameter omitted."""
    spies = exp.spies
    binding = {id(spies[name]): value for name, value in defaults.items()}
    # so that a `type(spy)` result becomes `type(default)`
    binding |= {id(type(spies[name])): type(value) for name, value in defaults.items()}
    bound = {
        spy_id: [
            TraceItem(
                item.attr,
                tuple(map_values(arg, binding) for arg in item.args),
                {key: map_values(val, binding) for key, val in item.kwargs.items()},
                item.ret,
            )
            for item in items
        ]
        for spy_id, items in exp.traces.items()
    }
    return exp._replace(
        spies={name: spy for name, spy in spies.items() if name not in defaults},
        traces=bound,
        results=[map_values(result, binding) for result in exp.results],
        tuple_params=exp.tuple_params - set(defaults),
    )


def resolve_defaults(
    func: AnyFunc,
    params: Mapping[str, Parameter],
    selected: Names,
    exploration: Exploration,
) -> list[Signature] | None:
    """The signatures with the defaults as typevar defaults, else with overloads, or
    `None` if the defaults need no handling.

    Omitting the defaulted parameters must behave like substituting their values
    into the generic signature; the function is rerun without them to check. On a
    mismatch the omitted calls are reported as separate overloads; a single
    defaulted parameter has its type excluded from the generic signature.
    """
    defaults = declared_defaults(params)
    kinds = {p.kind for p in params.values()}
    if not defaults or (
        # `*args` placeholders would positionally fill an omitted default
        Parameter.VAR_POSITIONAL in kinds
        and any(params[n].kind is not Parameter.KEYWORD_ONLY for n in defaults)
    ):
        return None

    required = {name: p for name, p in params.items() if name not in defaults}
    names = list(required)

    try:
        omitted = explore_spies(func, params, omit=defaults)
        # the comparison must see every required parameter, regardless of selection
        observed = signatures(omitted, params, names)
    except Exception:  # ruff: ignore[blind-except]
        return None

    omitted_defaults = _bind_exploration(exploration, defaults)
    expected = signatures(omitted_defaults, required, names)
    if _same(expected, observed):
        return signatures(exploration, params, selected, defaults)

    overloads = signatures(omitted, params, selected, defaults)
    if len(defaults) == 1:
        lines = signatures(exploration, params, selected, defaults, negate=True)
        return [*overloads, *lines]

    for name, value in defaults.items():
        try:
            variant = explore_spies(func, params, omit={name})
        except Exception:  # ruff: ignore[blind-except, try-except-continue]
            continue
        overloads += signatures(variant, params, selected, {name: value})

    return [*overloads, *signatures(exploration, params, selected)]


def dispatch_overloads(
    func: AnyFunc,
    params: Mapping[str, Parameter],
    selected: Names,
    exploration: Exploration,
    baseline: list[Signature],
) -> list[Signature]:
    """The overloads for a presence-test on a single parameter's attribute.

    Forcing the attribute absent surfaces the branch a placeholder hides. If the return
    ignores the attribute's value and both branches agree on deprecation, one overload
    covers it: the parameter widens to `object`, the return unions both branches.
    Otherwise the present overload stays over an `object` fallback, or, if that fallback
    would orphan a typevar, the `baseline` holds.
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
        and exploration.deprecated == variant.deprecated
        and returns_concrete(exploration.results)
        and returns_concrete(variant.results)
    ):
        combined = variant._replace(results=[*exploration.results, *variant.results])
        return signatures(combined, params, selected)
    tail = widened_signatures(variant, params, selected)
    return baseline + tail if tail else baseline
