"""Analysis passes over the recorded spy trace graph, ahead of rendering."""

from collections import Counter, defaultdict
from collections.abc import Generator, Iterable, Mapping, Sequence
from itertools import groupby
from typing import cast

from ._protocols import _DUNDER_ATTR, _DUNDER_CAN_R, _DUNDER_CLASS_ATTR
from ._spy import _Marker, _own_spy, _Spy, _SpyObject, _TraceItem, _Traces, as_spy
from ._values import _children, _Fn, _Gen, _Rec, _RecRef, _walk

type _Producer = Mapping[int, tuple[_SpyObject, _TraceItem]]


def return_spies(value: object) -> Generator[_SpyObject]:
    # an `_Fn`'s parameter spies are not returns; they are named like parameters
    for node in _walk(value):
        if (spy := as_spy(node)) is not None:
            yield spy


def dispatch_candidates(
    spies: Mapping[str, _SpyObject],
    traces: _Traces,
) -> list[tuple[str, str]]:
    """The `(parameter, attribute)` reads on `spies`, to probe for presence-dispatch."""
    return list(
        dict.fromkeys(
            (param, name)
            for param, spy in spies.items()
            for item in traces.get(id(spy), ())
            if item.attr == "__getattr__" and item.args
            if isinstance(name := item.args[0], str) and not isinstance(name, _Spy)
        ),
    )


def _param_traces(
    spies: Mapping[str, _SpyObject],
    traces: _Traces,
    param: str,
) -> Sequence[_TraceItem]:
    """The recorded ops on `param`'s spy, empty if it has none."""
    spy = spies.get(param)
    return () if spy is None else traces.get(id(spy), ())


def absent_verdict(
    spies: Mapping[str, _SpyObject],
    traces: _Traces,
    param: str,
    name: str,
) -> bool | None:
    """How forcing `name` absent resolves `param`, from one scan of its trace.

    `None`: absence not tolerated, no dispatch. `True`: only the absence remains, widen
    to `object`. `False`: a present-branch read survives, keep an `object` fallback.
    """
    items = _param_traces(spies, traces, param)
    marker = "__getattr__", name
    if not any(item.attr == _Marker.ABSENT and item.args == marker for item in items):
        return None
    return all(item.attr == _Marker.ABSENT for item in items)


def requires_only_presence(
    spies: Mapping[str, _SpyObject],
    traces: _Traces,
    param: str,
    name: str,
) -> bool:
    """Whether `param`'s sole requirement is the bare presence of `name`.

    Every op must read `name` and never constrain its value, so widening the parameter
    past `Has[name]` drops nothing a caller would have to satisfy.
    """
    items = _param_traces(spies, traces, param)
    return bool(items) and all(
        item.attr == "__getattr__"
        and item.args == (name,)
        and not traces.get(id(item.return_))
        for item in items
    )


def returns_concrete(results: Iterable[object]) -> bool:
    """Whether no result carries a spy, so the return is a concrete type."""
    return all(next(return_spies(r), None) is None for r in results)


def _trace_order(params: Sequence[_SpyObject], traces: _Traces) -> list[_SpyObject]:
    """The reachable spies: `params` first, then each op's return spy, depth-first."""
    order: list[_SpyObject] = []
    seen: set[int] = set()
    stack = list(reversed(params))
    while stack:
        spy = stack.pop()
        if id(spy) in seen:
            continue
        seen.add(id(spy))
        order.append(spy)
        stack.extend(
            op.return_ for op in traces[id(spy)] if isinstance(op.return_, _SpyObject)
        )
    return order


def analyze(
    params: Sequence[_SpyObject],
    results: Iterable[object],
    traces: _Traces,
) -> tuple[list[_SpyObject], dict[int, int]]:
    order = _trace_order(params, traces)
    ops = [op for spy in order for op in traces[id(spy)]]

    appear: Counter[int] = Counter(id(spy) for spy in params)
    appear.update(id(spy) for result in results for spy in return_spies(result))
    appear.update(
        id(arg)
        for op in ops
        for value in (*op.args, *op.kwargs.values())
        if (arg := as_spy(value)) is not None
    )
    appear.update(id(op.return_) for op in ops if isinstance(op.return_, _SpyObject))

    return order, appear


def _shape(spy: _SpyObject, made_by: _Producer, keys: dict[int, object]) -> object:
    """A structural key over op-shape, not operands: `x[y]` and `x[z]` share one."""
    sid = id(spy)
    if sid in keys:
        return keys[sid]
    keys[sid] = sid  # a parameter or leaf is its own key
    if (made := made_by.get(sid)) is not None:
        owner, item = made
        # an attribute name replaces the fixed arity: `x.spam` is not `x.ham`
        attr = item.attr
        arity: object = (
            item.args[0]
            if attr in _DUNDER_ATTR or attr in _DUNDER_CLASS_ATTR
            else len(item.args)
        )
        owner_key = _shape(owner, made_by, keys)
        keys[sid] = "op", owner_key, attr, arity, tuple(sorted(item.kwargs))
    return keys[sid]


def representatives(order: Sequence[_SpyObject], traces: _Traces) -> dict[int, int]:
    """Map each spy to a representative: the same operation on an owner of the same
    shape shares one, so the fresh placeholder each forked run allocates for a
    repeated subexpression collapses onto it instead of spawning a type parameter.
    """
    # a result is always a fresh spy, so a parameter never appears here and stays a leaf
    made_by: dict[int, tuple[_SpyObject, _TraceItem]] = {}
    for owner in order:
        for item in traces[id(owner)]:
            if isinstance(item.return_, _SpyObject):
                made_by.setdefault(id(item.return_), (owner, item))

    keys: dict[int, object] = {}
    rep: dict[object, int] = {}  # structural key -> the first spy that had it
    reps: dict[int, int] = {}
    for spy in order:
        reps[id(spy)] = rep.setdefault(_shape(spy, made_by, keys), id(spy))
    return reps


def group_traces(
    named: Iterable[int],
    reps: dict[int, int],
    traces: _Traces,
) -> _Traces:
    """Each representative's bound: the traces of every named spy that shares it.

    Only named spies merge; an inline one still renders its constraints where used.
    """
    merged: _Traces = {}
    for spy_id in named:
        merged.setdefault(reps.get(spy_id, spy_id), []).extend(traces.get(spy_id, ()))
    return merged


def spy_runs(items: Iterable[object], spy: _SpyObject) -> list[int]:
    """Lengths of each consecutive run of `spy` within `items`."""
    groups = groupby(items, key=lambda item: item is spy)
    return [sum(1 for _ in group) for is_spy, group in groups if is_spy]


def _packed_uses(value: object, spy: _SpyObject, count: int) -> Generator[bool]:
    # yields each use of `spy`: True if packed (one full placeholder run in a tuple)
    match value:
        case _SpyObject():
            if value is spy:
                yield False
            return
        case tuple() if not isinstance(value, _Gen | _Fn | _Rec | _RecRef):
            tup = cast("tuple[object, ...]", value)
            if runs := spy_runs(tup, spy):
                yield runs == [count]
            items = (item for item in tup if item is not spy)
        case _:
            items = _children(value)

    for item in items:
        yield from _packed_uses(item, spy, count)


def all_packed(
    spy: _SpyObject,
    results: Iterable[object],
    traces: _Traces,
    count: int,
) -> bool:
    """Whether `spy` is used at least once, but only ever packed.

    PEP 646 cannot express bare element uses, and traces on `spy` are operations on its
    elements, so a variadic parameter renders as a `TypeVarTuple` only when this holds.
    """
    if traces[id(spy)]:
        return False

    trace_values = (
        value
        for items in traces.values()
        for item in items
        for value in (*item.args, *item.kwargs.values())
    )
    uses = [
        use
        for value in (*results, *trace_values)
        for use in _packed_uses(value, spy, count)
    ]
    return bool(uses) and all(uses)


def reflect(params: Sequence[_SpyObject], traces: _Traces) -> _Traces:
    """A copy of `traces` with each spy-spy binary op reflected onto its RHS."""
    kept: _Traces = dict(traces)
    added: defaultdict[int, list[_TraceItem]] = defaultdict(list)
    for spy in _trace_order(params, traces):
        keep: list[_TraceItem] = []
        for item in traces[id(spy)]:
            rhs = item.args[0] if item.args else None
            if item.attr in _DUNDER_CAN_R and isinstance(rhs, _SpyObject):
                reflected = _TraceItem("__r" + item.attr[2:], (spy,), {}, item.return_)
                added[id(_own_spy(rhs))].append(reflected)
            else:
                keep.append(item)
        kept[id(spy)] = keep
    return {spy_id: keep + added[spy_id] for spy_id, keep in kept.items()}
