"""The lowered target model and the pure `_ir.Node` helpers shared across lowering.

`Lowerer` builds these definitions; `_print` emits them.
"""

import graphlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]


@dataclass(frozen=True, slots=True)
class Attr:
    """A protocol attribute: `name: T`, a `@property` (read-only, or with a setter of
    another type), or a `ClassVar`."""

    name: str
    type: _ir.Node
    classvar: bool = False
    readonly: bool = False
    setter: _ir.Node | None = None
    # the setter type is not one the getter returns, which basedpyright reports
    mismatch: bool = False


@dataclass(frozen=True, slots=True)
class Method:
    """A protocol method, e.g. a `Has['name', () -> +R]` or a callable's `__call__`."""

    name: str
    params: _ir.Terms
    ret: _ir.Node


type Member = Attr | Method


@dataclass(frozen=True, slots=True)
class ProtocolDef:
    """A synthesized helper `Protocol`: extra `bases` (intersection) or `members`."""

    name: str
    type_params: tuple[_ir.TypeParam, ...]
    bases: tuple[_ir.Node, ...]
    members: tuple[Member, ...]


@dataclass(frozen=True, slots=True)
class Alias:
    """A (possibly recursive) `type` alias, for a self-referential concrete bound."""

    name: str
    type_params: tuple[_ir.TypeParam, ...]
    value: _ir.Node


type Helper = ProtocolDef | Alias  # a synthesized helper definition


@dataclass(frozen=True, slots=True)
class Module:
    helpers: tuple[Helper, ...]
    funcs: tuple[_ir.Signature, ...]


def free_tyvars(nodes: Iterable[_ir.Term], tyvars: frozenset[str]) -> list[str]:
    """The signature typevars referenced across `nodes`, in first-appearance order."""
    seen: dict[str, None] = {}
    for node in nodes:
        for name in _ir.names(node):
            if name in tyvars:
                seen.setdefault(name, None)
    return list(seen)


def is_generic(node: _ir.Node, tyvars: frozenset[str]) -> bool:
    """Whether `node` references any of the signature's type variables."""
    return not frozenset(_ir.names(node)).isdisjoint(tyvars)


def is_protocol_node(node: _ir.Node) -> bool:
    return isinstance(node, _ir.App) and node.origin.startswith(("Can", "Has", "Just"))


def combine_name(bases: Sequence[str]) -> str:
    """The combined-protocol name, e.g. `CanNeg` + `CanRAdd` -> `CanNegRAdd`."""
    for prefix in ("Can", "Has", "Just"):
        if bases and all(b.startswith(prefix) for b in bases):
            return prefix + "".join(b.removeprefix(prefix) for b in bases)
    return "".join(bases)


def bound_name(bound: _ir.Node, tyvar: str) -> str:
    return bound.origin if isinstance(bound, _ir.App) else f"Bound{tyvar}"


def strip_variance(node: _ir.Node) -> _ir.Node:
    return node.part if isinstance(node, _ir.Variance) else node


def member_nodes(members: Iterable[Member]) -> Iterable[_ir.Term]:
    for member in members:
        if isinstance(member, Attr):
            yield member.type
            if member.setter is not None:
                yield member.setter
        else:
            yield from member.params
            yield member.ret


def subst_member(member: Member, m: Mapping[str, _ir.Node]) -> Member:
    if isinstance(member, Attr):
        setter = None if member.setter is None else _ir.subst(member.setter, m)
        return replace(member, type=_ir.subst(member.type, m), setter=setter)
    params = tuple(_ir.subst_term(p, m) for p in member.params)
    return replace(member, params=params, ret=_ir.subst(member.ret, m))


def _reachable(deps: Mapping[str, frozenset[str]], start: str) -> set[str]:
    seen: set[str] = set()
    stack = list(deps.get(start, ()))
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(deps.get(node, ()))
    return seen


def cyclic_names(deps: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """The nodes that lie on a cycle (a self-loop or a mutual reference)."""
    return frozenset(node for node in deps if node in _reachable(deps, node))


def components(
    cyclic: frozenset[str],
    deps: Mapping[str, frozenset[str]],
) -> list[frozenset[str]]:
    """The mutually-reachable groups within the cyclic nodes."""
    groups: list[frozenset[str]] = []
    seen: set[str] = set()
    for node in sorted(cyclic):
        if node in seen:
            continue
        reach = _reachable(deps, node)
        group = frozenset(
            {node} | {m for m in cyclic if m in reach and node in _reachable(deps, m)},
        )
        seen |= group
        groups.append(group)
    return groups


def resolution_order(
    groups: Iterable[frozenset[str]],
    singles: Iterable[str],
    deps: Mapping[str, frozenset[str]],
) -> list[frozenset[str]]:
    """The cyclic `groups` and acyclic `singles` as units, in dependency order."""
    unit_of = {name: group for group in groups for name in group}
    unit_of |= {name: frozenset({name}) for name in singles}
    # sorted insertion keeps the order deterministic; `deps` is the predecessor map
    units = sorted(set(unit_of.values()), key=sorted)
    graph = {
        unit: sorted(
            {
                unit_of[dep]
                for name in unit
                for dep in deps.get(name, ())
                if dep not in unit
            },
            key=sorted,
        )
        for unit in units
    }
    return list(graphlib.TopologicalSorter(graph).static_order())
