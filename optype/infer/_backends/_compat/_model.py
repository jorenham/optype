"""The lowered target model and the pure `_ir.Node` helpers shared across lowering.

`_Lowerer` builds these definitions; `_print` emits them.
"""

import graphlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

# `from optype.infer import _ir` would re-enter the package, which imports this module
import optype.infer._ir as _ir  # noqa: PLR0402

__all__ = (
    "_Alias",
    "_Attr",
    "_Func",
    "_Member",
    "_Method",
    "_Module",
    "_Protocol",
    "_bound_name",
    "_combine_name",
    "_components",
    "_cyclic",
    "_free_tyvars",
    "_is_generic",
    "_is_protocol_node",
    "_member_nodes",
    "_strip_variance",
    "_subst_member",
    "_toposort",
    "_value",
)


@dataclass(frozen=True, slots=True)
class _Attr:
    """A protocol attribute: `name: T`, a read-only `@property`, or a `ClassVar`."""

    name: str
    type: _ir.Node
    classvar: bool = False
    readonly: bool = False


@dataclass(frozen=True, slots=True)
class _Method:
    """A protocol method, e.g. a `Has['name', () -> +R]` or a callable's `__call__`."""

    name: str
    params: tuple[_ir.Node | _ir.Arg, ...]
    ret: _ir.Node


type _Member = _Attr | _Method


@dataclass(frozen=True, slots=True)
class _Protocol:
    """A synthesized helper `Protocol`: extra `bases` (intersection) or `members`."""

    name: str
    type_params: tuple[_ir.TypeParam, ...]
    bases: tuple[_ir.Node, ...]
    members: tuple[_Member, ...]


@dataclass(frozen=True, slots=True)
class _Alias:
    """A (possibly recursive) `type` alias, for a self-referential concrete bound."""

    name: str
    type_params: tuple[_ir.TypeParam, ...]
    value: _ir.Node


@dataclass(frozen=True, slots=True)
class _Func:
    """One overload: a `def` with PEP 695 type parameters and an optional marker."""

    type_params: tuple[_ir.TypeParam, ...]
    params: tuple[_ir.Param, ...]
    ret: _ir.Node
    deprecated: str | None


@dataclass(frozen=True, slots=True)
class _Module:
    helpers: tuple[_Protocol | _Alias, ...]
    funcs: tuple[_Func, ...]


def _value(arg: _ir.Node | _ir.Arg) -> _ir.Node:
    return arg.value if isinstance(arg, _ir.Arg) else arg


def _free_tyvars(
    nodes: Iterable[_ir.Node | _ir.Arg],
    tyvars: frozenset[str],
) -> list[str]:
    """The signature typevars referenced across `nodes`, in first-appearance order."""
    seen: dict[str, None] = {}
    for node in nodes:
        for name in _ir.names(node):
            if name in tyvars:
                seen.setdefault(name, None)
    return list(seen)


def _is_generic(node: _ir.Node, tyvars: frozenset[str]) -> bool:
    """Whether `node` references any of the signature's type variables."""
    return not frozenset(_ir.names(node)).isdisjoint(tyvars)


def _is_protocol_node(node: _ir.Node) -> bool:
    return isinstance(node, _ir.App) and node.origin.startswith(("Can", "Has", "Just"))


def _combine_name(bases: Sequence[str]) -> str:
    """The combined-protocol name, e.g. `CanNeg` + `CanRAdd` -> `CanNegRAdd`."""
    for prefix in ("Can", "Has", "Just"):
        if bases and all(b.startswith(prefix) for b in bases):
            return prefix + "".join(b.removeprefix(prefix) for b in bases)
    return "".join(bases)


def _bound_name(bound: _ir.Node, tyvar: str) -> str:
    return bound.origin if isinstance(bound, _ir.App) else f"Bound{tyvar}"


def _strip_variance(node: _ir.Node) -> _ir.Node:
    return node.part if isinstance(node, _ir.Variance) else node


def _member_nodes(members: Iterable[_Member]) -> Iterable[_ir.Node | _ir.Arg]:
    for member in members:
        if isinstance(member, _Attr):
            yield member.type
        else:
            yield from member.params
            yield member.ret


def _subst_member(member: _Member, m: Mapping[str, _ir.Node]) -> _Member:
    if isinstance(member, _Attr):
        return replace(member, type=_ir.subst(member.type, m))
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


def _cyclic(deps: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """The nodes that lie on a cycle (a self-loop or a mutual reference)."""
    return frozenset(node for node in deps if node in _reachable(deps, node))


def _components(
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


def _toposort(
    nodes: frozenset[str] | set[str],
    deps: Mapping[str, frozenset[str]],
) -> list[str]:
    """The acyclic `nodes` ordered so each follows the others it depends on."""
    # sorted insertion keeps the order deterministic; `deps` is the predecessor map
    graph = {node: deps.get(node, frozenset()) & nodes for node in sorted(nodes)}
    return list(graphlib.TopologicalSorter(graph).static_order())
