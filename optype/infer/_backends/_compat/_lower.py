"""`Lowerer`: rewrite the `Signature` IR into a printable `Module`.

It synthesizes helper `Protocol`s (and recursive aliases) for the constructs the typing
spec cannot express: intersections, the inline `Has[...]` form, typevar-referencing
bounds, and keyword/defaulted callables. `docs/infer.md` is the source of truth.
"""

import builtins
import functools
import keyword
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from itertools import count, islice, product
from typing import final

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]
from ._model import (
    Alias,
    Attr,
    Helper,
    Member,
    Method,
    Module,
    ProtocolDef,
    bound_name,
    combine_name,
    components,
    cyclic_names,
    free_tyvars,
    is_generic,
    is_protocol_node,
    member_nodes,
    strip_variance,
    subst_member,
    toposort,
)
from ._print import OPTYPE, import_of, member_key, type_text
from optype.infer._errors import InferError
from optype.inspect import is_union_type


def _bound_node(bound: object) -> _ir.Node | None:
    """A typevar bound as an IR node: a type, a union, or a shipped protocol."""
    if isinstance(bound, type):
        return _ir.Type(bound)
    args: list[_ir.Node] = []
    for arg in typing.get_args(bound):
        if (node := _bound_node(arg)) is None:
            return None
        args.append(node)
    if not args:
        return None
    if is_union_type(bound):
        return _ir.union(args)
    origin = typing.get_origin(bound)
    if not isinstance(origin, type):
        return None
    return _ir.App(_origin_name(origin), tuple(args))


def _origin_name(origin: type) -> str:
    """The name the printer imports `origin` by: bare if importable, else qualified."""
    name = origin.__name__
    found = import_of(name)
    if (
        found is not None
        and found[1] == name
        and origin.__module__.startswith(found[0])
    ):
        return name
    return _ir.type_name(origin)


@functools.cache
def _protocol_bounds(origin: str) -> tuple[_ir.Node | None, ...] | None:
    """The per-argument typevar bounds of an `optype` protocol, or `None` if unknown.

    Reconciles the inferred `App` with the shipped generic: a non-generic protocol
    reports `()` (so excess arguments drop), and a bounded argument lets the matching
    inferred typevar pick up that bound.
    """
    if origin not in OPTYPE:
        return None

    import optype  # ruff: ignore[import-outside-top-level]

    cls = getattr(optype, origin, None)
    params = getattr(cls, "__parameters__", None)
    if params is None:
        return None
    return tuple(_bound_node(getattr(p, "__bound__", None)) for p in params)


def _inherited_bounds(
    bases: Sequence[_ir.Node],
    typars: frozenset[str],
) -> dict[str, _ir.Node]:
    """Each helper type parameter's bound, taken from the protocol base it fills."""
    result: dict[str, _ir.Node] = {}
    for base in bases:
        if (
            not isinstance(base, _ir.App)
            or (bounds := _protocol_bounds(base.origin)) is None
        ):
            continue
        for arg, bound in zip(base.args, bounds, strict=False):
            if bound is not None and isinstance(arg, _ir.Name) and arg.name in typars:
                result.setdefault(arg.name, bound)
    return result


type _Constraints = dict[str, list[_ir.Node]]

# the canonical dedup keys: equal definitions serialize to equal keys
type _ProtoKey = tuple[tuple[str, ...], tuple[str, ...]]  # bases, then members
type _GroupKey = tuple[tuple[bool, str], ...]  # one entry per cyclic bound


@final
class _HelperRegistry:
    """The helper definitions shared across every signature of one render."""

    defs: dict[str, Helper]
    groups: dict[_GroupKey, list[str]]
    _keys: dict[_ProtoKey, str]
    _names: set[str]  # every claimed helper name

    def __init__(self) -> None:
        self.defs = {}
        self.groups = {}
        self._keys = {}
        self._names = set()

    def register(
        self,
        candidate: str,
        key: _ProtoKey,
        build: Callable[[str], Helper],
    ) -> str:
        if key in self._keys:
            return self._keys[key]

        name = self.claim(candidate)
        self._keys[key] = name
        self.defs[name] = build(name)
        return name

    def claim(self, candidate: str) -> str:
        """A fresh helper name that collides with no real import or other helper."""
        name = candidate
        i = 2
        while (
            name in self._names
            or import_of(name) is not None
            or hasattr(builtins, name)
        ):
            name = f"{candidate}{i}"
            i += 1
        self._names.add(name)
        return name


@final
class Lowerer:
    """Rewrite a sequence of `Signature`s into a printable `Module`."""

    _registry: _HelperRegistry

    def __init__(self) -> None:
        self._registry = _HelperRegistry()

    def module(self, sigs: Sequence[_ir.Signature]) -> Module:
        funcs = [_SigLowerer(self._registry, sig).func() for sig in sigs]
        return Module(tuple(self._registry.defs.values()), tuple(funcs))


def _signed(args: tuple[_ir.Node, ...]) -> tuple[_ir.Node, ...] | None:
    """`args` as reads and writes; a lone method is a read of its callable type."""
    if all(isinstance(arg, _ir.Variance) for arg in args):
        return args
    if len(args) == 1 and isinstance(args[0], _ir.Fn):
        return (_ir.Variance(_ir.COVARIANT, args[0]),)
    return None


def _merge_has(parts: Sequence[_ir.Node]) -> list[_ir.Node]:
    """Join the `Has` members of one attribute, so its reads and writes share one."""
    out: list[_ir.Node] = []
    index: dict[str, int] = {}
    for part in parts:
        if isinstance(part, _ir.Has) and (signed := _signed(part.args)) is not None:
            if (i := index.get(part.attr)) is not None:
                prev = out[i]
                assert isinstance(prev, _ir.Has)
                merged = _signed(prev.args)
                assert merged is not None
                out[i] = _ir.Has(part.attr, merged + signed)
                continue
            index[part.attr] = len(out)
        out.append(part)
    return out


@final
class _SigLowerer:
    """Lower one `Signature`; its type variables are fixed for the whole traversal."""

    _registry: _HelperRegistry
    _sig: _ir.Signature
    _tyvars: frozenset[str]  # the signature's type parameter names

    def __init__(self, registry: _HelperRegistry, sig: _ir.Signature) -> None:
        self._registry = registry
        self._sig = sig
        self._tyvars = frozenset(typar.name for typar in sig.type_params)

    def func(self) -> _ir.Signature:
        sig = self._sig
        constraints: _Constraints = {}
        params = [self._param(p, constraints) for p in sig.params]
        ret = self._node(sig.ret, constraints)
        kept, subst = self._resolve_typars(sig.type_params, constraints)
        params = [self._subst_param(p, subst) for p in params]
        return replace(
            sig,
            type_params=tuple(kept),
            params=tuple(params),
            ret=_ir.subst(ret, subst),
        )

    def _resolve_typars(
        self,
        typars: Sequence[_ir.TypeParam],
        constraints: _Constraints,
    ) -> tuple[list[_ir.TypeParam], dict[str, _ir.Node]]:
        """Eliminate typevar-referencing bounds, which PEP 695 forbids.

        An acyclic bound substitutes in place; a cyclic (self- or mutually-referential)
        one becomes a helper `Protocol`.
        """
        bound: dict[str, _ir.Node | None] = {}
        default: dict[str, _ir.Node | None] = {}
        for typar in typars:
            merged = self._merge_bound(
                typar.bound,
                constraints.get(typar.name),
            )
            bound[typar.name] = None if merged == _ir.OBJECT else merged
            default[typar.name] = (
                None if typar.default is None else self._node(typar.default, {})
            )

        elim = {
            tyvar: b for tyvar, b in bound.items() if b and is_generic(b, self._tyvars)
        }
        deps = {tyvar: frozenset(_ir.names(b)) & set(elim) for tyvar, b in elim.items()}
        cyclic = cyclic_names(deps)

        subst: dict[str, _ir.Node] = {}
        for group in components(cyclic, deps):
            subst |= self._hoist_group(group, elim)
        for tyvar in toposort(set(elim) - cyclic, deps):
            subst[tyvar] = _ir.subst(elim[tyvar], subst)

        kept = [
            replace(typar, bound=bound[typar.name], default=default[typar.name])
            for typar in typars
            if typar.name not in elim
        ]
        return kept, subst

    def _param(
        self,
        param: _ir.Param,
        constraints: _Constraints,
    ) -> _ir.Param:
        return replace(param, node=self._node(param.node, constraints))

    @staticmethod
    def _subst_param(param: _ir.Param, subst: Mapping[str, _ir.Node]) -> _ir.Param:
        return replace(param, node=_ir.subst(param.node, subst))

    def _node(  # ruff: ignore[too-many-return-statements]
        self,
        node: _ir.Node,
        constraints: _Constraints,
    ) -> _ir.Node:
        match node:
            case _ir.Has(attr, signed):
                return self._has(attr, signed, constraints)
            case _ir.App(origin, args):
                return self._app(origin, args, constraints)
            case _ir.Fn(params, ret):
                return self._fn(params, ret, constraints)
            case _ir.Union(parts):
                lowered = [self._node(p, constraints) for p in parts]
                return _ir.union(lowered) or _ir.OBJECT
            case _ir.Intersection(parts):
                return self._inter(parts, constraints)
            case _ir.Not(_):
                return _ir.OBJECT
            case _ir.Unpack(part):
                return _ir.Unpack(self._node(part, constraints))
            case _ir.Variance(_, part):
                return self._node(part, constraints)
            case _:
                return node

    def _arg(self, arg: _ir.Term, constraints: _Constraints) -> _ir.Term:
        if isinstance(arg, _ir.Arg):
            return replace(arg, value=self._node(arg.value, constraints))

        return self._node(arg, constraints)

    def _app(self, origin: str, args: _ir.Terms, constraints: _Constraints) -> _ir.App:
        lowered = [self._arg(a, constraints) for a in args]
        if (bounds := _protocol_bounds(origin)) is not None:
            lowered = lowered[: len(bounds)]
            for arg, bound in zip(lowered, bounds, strict=False):
                if bound and isinstance(arg, _ir.Name) and arg.name in self._tyvars:
                    constraints.setdefault(arg.name, []).append(bound)
        return _ir.App(origin, tuple(lowered))

    def _inter(self, parts: Sequence[_ir.Node], constraints: _Constraints) -> _ir.Node:
        parts = _merge_has(parts)
        # a typevar member lifts the others into that typevar's bound, which is sound
        tyvar = next(
            (
                p.name
                for p in parts
                if isinstance(p, _ir.Name) and p.name in self._tyvars
            ),
            None,
        )
        if tyvar is not None:
            extra = [
                p
                for p in parts
                if not isinstance(p, _ir.Not)
                and not (isinstance(p, _ir.Name) and p.name == tyvar)
            ]
            constraints.setdefault(tyvar, []).extend(extra)
            return _ir.Name(tyvar)

        lowered: list[_ir.Node] = []
        seen: set[str] = set()
        for part in parts:
            if isinstance(part, _ir.Not):
                continue
            node = self._node(part, constraints)
            if (key := type_text(node)) not in seen:
                seen.add(key)
                lowered.append(node)
        return self._combine(lowered)

    def _proto_app(
        self,
        candidate: str,
        *,
        bases: Sequence[_ir.Node] = (),
        members: Sequence[Member] = (),
    ) -> _ir.App:
        """Register a helper `Protocol` (canonicalized for reuse) and apply it."""
        fv = free_tyvars([*bases, *member_nodes(members)], self._tyvars)
        # a member binds its name in the class body, where a type parameter is looked up
        taken = {mem.name for mem in members}
        fresh = (n for n in map(_ir.tyvar_name, count()) if n not in taken)
        canon_names = list(islice(fresh, len(fv)))
        m = {name: _ir.Name(c) for name, c in zip(fv, canon_names, strict=True)}
        canon_bases = tuple(_ir.subst(b, m) for b in bases)
        canon_members = tuple(subst_member(mem, m) for mem in members)
        tp_bounds = _inherited_bounds(canon_bases, frozenset(canon_names))
        typars = tuple(_ir.TypeParam(c, tp_bounds.get(c)) for c in canon_names)
        key = (
            tuple(type_text(b) for b in canon_bases),
            tuple(map(member_key, canon_members)),
        )
        name = self._registry.register(
            candidate,
            key,
            lambda nm: ProtocolDef(nm, typars, canon_bases, canon_members),
        )
        return _ir.App(name, tuple(_ir.Name(f) for f in fv))

    def _combine(self, parts: Sequence[_ir.Node]) -> _ir.Node:
        # an `(A | B) & C` distributes to `(A & C) | (B & C)`: a union cannot be a base
        if not parts:
            return _ir.OBJECT
        if len(parts) == 1:
            return parts[0]

        unions = [p for p in parts if isinstance(p, _ir.Union)]
        if not unions:
            # a callable is not a valid base; it lifts into a `__call__` method instead
            bases = [p for p in parts if not isinstance(p, _ir.Fn)]
            members = tuple(
                Method("__call__", p.params, p.ret)
                for p in parts
                if isinstance(p, _ir.Fn)
            )
            candidate = (
                combine_name([p.origin for p in bases if isinstance(p, _ir.App)]) or "P"
            )
            return self._proto_app(candidate, bases=bases, members=members)

        rest = [p for p in parts if not isinstance(p, _ir.Union)]
        variants = [
            self._combine([*rest, *picks])
            for picks in product(*(u.parts for u in unions))
        ]
        return _ir.union(variants) or _ir.OBJECT

    def _has(
        self,
        attr: str,
        signed: tuple[_ir.Node, ...],
        constraints: _Constraints,
    ) -> _ir.Node:
        if not attr.isidentifier() or keyword.iskeyword(attr):
            # a protocol member must be named for the real attribute, so a name that is
            # not a valid identifier (e.g. from `getattr(x, "a-b")`) is inexpressible
            msg = f"cannot render attribute {attr!r} as a protocol member"
            raise InferError(msg)

        member = self._has_member(attr, signed, constraints)
        candidate = "Has" + attr[:1].upper() + attr[1:]
        return self._proto_app(candidate, members=(member,))

    def _has_member(  # ruff: ignore[too-many-return-statements]
        self,
        attr: str,
        signed: tuple[_ir.Node, ...],
        constraints: _Constraints,
        *,
        classvar: bool = False,
    ) -> Member:
        if (
            len(signed) == 1
            and isinstance(signed[0], _ir.App)
            and signed[0].origin == "ClassVar"
        ):
            inner = tuple(_ir.term_node(a) for a in signed[0].args)
            return self._has_member(attr, inner, constraints, classvar=True)
        if not signed:
            return Attr(attr, _ir.OBJECT, classvar=classvar, readonly=not classvar)
        if len(signed) == 1 and isinstance(signed[0], _ir.Fn):
            fn = signed[0]
            ret = self._node(strip_variance(fn.ret), constraints)
            params = tuple(self._arg(p, constraints) for p in fn.params)
            return Method(attr, params, ret)
        signs = [s for s in signed if isinstance(s, _ir.Variance)]
        if len(signs) != len(signed):
            node = self._node(strip_variance(signed[0]), constraints)
            cv = classvar and not is_generic(node, self._tyvars)
            return Attr(attr, node, classvar=cv)
        reads = [s.part for s in signs if s.sign == _ir.COVARIANT]
        writes = [s.part for s in signs if s.sign == _ir.CONTRAVARIANT]
        read = self._node(_ir.intersection(reads) or _ir.OBJECT, constraints)
        write = self._node(_ir.union(writes) or _ir.OBJECT, constraints)
        # a class attribute is a plain `ClassVar`, which cannot hold a typevar; a
        # generic one demotes to the instance form
        nodes = (read, write) if writes else (read,)
        if classvar and not any(is_generic(n, self._tyvars) for n in nodes):
            return Attr(attr, read if reads else write, classvar=True)
        # a dunder has a declared type, which a settable property would override
        # incompatibly
        if writes and attr.startswith("__") and attr.endswith("__"):
            return Attr(attr, read if reads else write)
        # a property with a setter is what a plain attribute and a settable property
        # both satisfy; an annotation is only matched by a plain attribute
        return Attr(attr, read, readonly=not writes, setter=write if writes else None)

    def _fn(
        self,
        params: _ir.Terms,
        ret: _ir.Node,
        constraints: _Constraints,
    ) -> _ir.Node:
        lowered_ret = self._node(ret, constraints)
        lowered = tuple(self._arg(p, constraints) for p in params)
        # `Callable` covers positional params; a keyword or default needs `__call__`
        if not any(isinstance(p, _ir.Arg) and (p.key or p.default) for p in lowered):
            return _ir.Fn(lowered, lowered_ret)
        member = Method("__call__", lowered, lowered_ret)
        return self._proto_app("CanCallP", members=(member,))

    def _merge_bound(
        self,
        bound: _ir.Node | None,
        extra: list[_ir.Node] | None,
    ) -> _ir.Node | None:
        parts: list[_ir.Node] = []
        if bound is not None:
            parts += bound.parts if isinstance(bound, _ir.Intersection) else [bound]
        if extra:
            parts += extra
        if not parts:
            return None
        node = parts[0] if len(parts) == 1 else _ir.Intersection(tuple(parts))
        return self._node(node, {})

    def _hoist_group(
        self,
        group: frozenset[str],
        bound: Mapping[str, _ir.Node],
    ) -> dict[str, _ir.Node]:
        """Turn a cyclic bound group into mutually-referential helper definitions."""
        members = sorted(group)
        free = list(
            dict.fromkeys(
                name
                for tyvar in members
                for name in _ir.names(bound[tyvar])
                if name in self._tyvars and name not in group
            ),
        )
        canon = [_ir.tyvar_name(i) for i in range(len(free))]
        rename: dict[str, _ir.Node] = {
            f: _ir.Name(canon[i]) for i, f in enumerate(free)
        }
        roles: dict[str, _ir.Node] = {
            tyvar: _ir.Name(_ir.placeholder_name(i)) for i, tyvar in enumerate(members)
        }
        key = tuple(
            (
                is_protocol_node(bound[tyvar]),
                type_text(_ir.subst(bound[tyvar], rename | roles)),
            )
            for tyvar in members
        )

        if key not in self._registry.groups:
            names = [
                self._registry.claim(bound_name(bound[tyvar], tyvar))
                for tyvar in members
            ]
            self._registry.groups[key] = names
            refs: dict[str, _ir.Node] = {
                tyvar: _ir.App(names[i], tuple(map(_ir.Name, canon)))
                for i, tyvar in enumerate(members)
            }
            typars = tuple(_ir.TypeParam(c) for c in canon)
            for i, tyvar in enumerate(members):
                body = _ir.subst(bound[tyvar], rename | refs)
                self._registry.defs[names[i]] = (
                    ProtocolDef(names[i], typars, (body,), ())
                    if is_protocol_node(bound[tyvar])
                    else Alias(names[i], typars, body)
                )

        names = self._registry.groups[key]
        site = tuple(map(_ir.Name, free))
        return {tyvar: _ir.App(names[i], site) for i, tyvar in enumerate(members)}
