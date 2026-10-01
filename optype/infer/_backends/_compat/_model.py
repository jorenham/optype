"""The lowered target model and the pure `_ir.Node` helpers shared across lowering.

`Lowerer` builds these definitions; `_print` emits them.
"""

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
class Module:
    helpers: tuple[ProtocolDef, ...]
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


def combine_name(bases: Sequence[str]) -> str:
    """The combined-protocol name, e.g. `CanNeg` + `CanRAdd` -> `CanNegRAdd`."""
    for prefix in ("Can", "Has", "Just"):
        if bases and all(b.startswith(prefix) for b in bases):
            return prefix + "".join(b.removeprefix(prefix) for b in bases)
    return "".join(bases)


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
