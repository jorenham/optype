"""Turn a loop the explorer ran several times back into one recursive typevar."""

from collections import defaultdict
from collections.abc import Mapping

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]

type _Typars = list[_ir.TypeParam]
type _Params = list[_ir.Param]
type _Sig = tuple[_Typars, _Params, _ir.Node]

_LOOP_MIN = 3  # copies a chain needs before it counts as a loop


def _shift_edges(bounded: Mapping[str, _ir.Node]) -> dict[str, str]:
    """Link `x -> y` when `y` is what `x` becomes on the next pass through the loop.

    All three must hold: `y` appears in `x`'s bound, the two bounds are the same apart
    from which names they use, and that renaming sends `y` to another bounded name.
    The last one rules out a `y` that the two bounds merely have in common.
    """
    edge: dict[str, str] = {}
    for x, bound in bounded.items():
        for y in dict.fromkeys(_ir.names(bound)):
            if y == x or y not in bounded:
                continue

            mapping = _ir.alpha_equal(bound, bounded[y])
            if mapping is not None and mapping.get(y) in bounded:
                edge[x] = y
                break
    return edge


def _gc_typars(typars: _Typars, params: _Params, ret: _ir.Node) -> _Typars:
    """Drop the type parameters that nothing mentions any more."""
    by_name = {typar.name: typar for typar in typars}
    reach: set[str] = set()
    stack = [name for p in params for name in _ir.names(p.node)]
    stack += _ir.names(ret)
    while stack:
        if (name := stack.pop()) in reach:
            continue

        reach.add(name)
        if (typar := by_name.get(name)) is not None:
            stack += _ir.names(typar.bound) if typar.bound is not None else ()
            stack += _ir.names(typar.default) if typar.default is not None else ()

    return [typar for typar in typars if typar.name in reach]


def _rename_sig(
    typars: _Typars,
    params: _Params,
    ret: _ir.Node,
    remap: dict[str, str],
) -> _Sig:
    """Apply a `Name` remap across a signature's type params, params, and return."""
    tps = [
        _ir.TypeParam(
            remap.get(typar.name, typar.name),
            None if typar.bound is None else _ir.rename(typar.bound, remap),
            None if typar.default is None else _ir.rename(typar.default, remap),
            typar.unpack,
        )
        for typar in typars
    ]
    args = [
        _ir.Param(
            p.name,
            _ir.rename(p.node, remap),
            p.prefix,
            p.nameless,
            p.default,
        )
        for p in params
    ]
    return tps, args, _ir.rename(ret, remap)


def _renumber_tyvars(typars: _Typars, params: _Params, ret: _ir.Node) -> _Sig:
    """Rename what is left back to `T, U, V, ...`, in order and without gaps."""
    remap: dict[str, str] = {}
    n = 0
    for typar in typars:
        if _ir.tyvar_index(typar.name) is not None:
            if (new := _ir.tyvar_name(n)) != typar.name:
                remap[typar.name] = new
            n += 1

    return _rename_sig(typars, params, ret, remap) if remap else (typars, params, ret)


def _collapse_renaming(
    bounded: Mapping[str, _ir.Node],
    edge: Mapping[str, str],
) -> dict[str, str]:
    """The renaming that points every name in a loop at the first copy of it.

    A name links to at most one other, so following the links from any name ends
    somewhere; the names that end in the same place are copies of one another.
    """
    order = {name: i for i, name in enumerate(bounded)}

    def terminal(name: str) -> str:
        seen: set[str] = set()
        while name in edge and name not in seen:
            seen.add(name)
            name = edge[name]
        return name

    runs: defaultdict[str, list[str]] = defaultdict(list)
    for name in bounded:
        runs[terminal(name)].append(name)

    remap: dict[str, str] = {}
    for members in runs.values():
        if len(members) < _LOOP_MIN:
            continue
        lead = min(members, key=order.__getitem__)
        remap.update({name: lead for name in members if name != lead})
    return remap


def collapse_recursive(typars: _Typars, params: _Params, ret: _ir.Node) -> _Sig:
    """Replace a chain of repeated typevars with the recursive one it stands for.

    Exploring a loop runs its body several times, and every pass gets a typevar of its
    own, so `sum` or `-x + x` comes out as `T, U, V, ...` where each bound looks like
    the one before it. Pointing them all at the first copy gives the recursive type the
    loop really has, instead of a record of how many times it happened to run. Two
    variables that go round the loop together stay recursive in terms of each other.
    """
    bounded = {
        typar.name: typar.bound
        for typar in typars
        if typar.bound is not None and not typar.unpack
    }
    if len(bounded) < _LOOP_MIN or not (edge := _shift_edges(bounded)):
        return typars, params, ret

    remap = _collapse_renaming(bounded, edge)
    if not remap:
        return typars, params, ret

    kept = [typar for typar in typars if typar.name not in remap]
    typars, params, ret = _rename_sig(kept, params, ret, remap)
    return _renumber_tyvars(_gc_typars(typars, params, ret), params, ret)
