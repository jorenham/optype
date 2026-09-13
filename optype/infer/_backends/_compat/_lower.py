"""`Lowerer`: rewrite the `Signature` IR into a printable `Module`.

It synthesizes helper `Protocol`s (and recursive aliases) for the constructs the typing
spec cannot express: intersections, the inline `Has[...]` form, typevar-referencing
bounds, and keyword/defaulted callables. `docs/infer.md` is the source of truth.
"""

import builtins
import functools
import keyword
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    resolution_order,
    strip_variance,
    subst_member,
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


# the terse form counts arguments where `optype` ships one protocol per arity
_ARITY_FORMS = {
    ("CanRound", 1): "CanRound1",
    ("CanRound", 2): "CanRound2",
    ("CanPow", 2): "CanPow2",
    ("CanPow", 3): "CanPow3",
}


@functools.cache
def _protocol_params(origin: str) -> tuple[object, ...] | None:
    """The type parameters of an `optype` protocol in declared order, if known.

    `__parameters__` orders them by first appearance across the bases, which is not
    the `Protocol[...]` order for every shipped protocol.
    """
    if origin not in OPTYPE:
        return None

    import optype  # ruff: ignore[import-outside-top-level]

    cls = getattr(optype, origin, None)
    for base in getattr(cls, "__orig_bases__", ()):
        if typing.get_origin(base) is typing.Protocol:
            return typing.get_args(base)
    return getattr(cls, "__parameters__", None)


def _protocol_bounds(origin: str) -> tuple[_ir.Node | None, ...] | None:
    """The per-argument typevar bounds of an `optype` protocol, or `None` if unknown.

    Reconciles the inferred `App` with the shipped generic: a non-generic protocol
    reports `()` (so excess arguments drop), and a bounded argument lets the matching
    inferred typevar pick up that bound.
    """
    if (params := _protocol_params(origin)) is None:
        return None
    return tuple(_bound_node(getattr(p, "__bound__", None)) for p in params)


def _protocol_variances(origin: str) -> tuple[_ir.Sign | None, ...] | None:
    """The declared variance per argument of an `optype` protocol, if known."""
    if (params := _protocol_params(origin)) is None:
        return None
    return tuple(
        _ir.COVARIANT
        if getattr(p, "__covariant__", False)
        else _ir.CONTRAVARIANT
        if getattr(p, "__contravariant__", False)
        else None
        for p in params
    )


type _Bounds = Mapping[str, Sequence[_ir.Node]]


def _reads_writes(
    signed: tuple[_ir.Node, ...],
) -> tuple[list[_ir.Node], list[_ir.Node]]:
    signs = [s for s in signed if isinstance(s, _ir.Variance)]
    reads = [s.part for s in signs if s.sign == _ir.COVARIANT]
    writes = [s.part for s in signs if s.sign == _ir.CONTRAVARIANT]
    return reads, writes


def _variances_at(origin: str, arity: int) -> tuple[_ir.Sign | None, ...] | None:
    """The declared variances of the shipped protocol behind `origin` at `arity`."""
    signs = _protocol_variances(_ARITY_FORMS.get((origin, arity), origin))
    return signs if signs is not None and len(signs) == arity else None


def _implies_args(sub: _ir.App, sup: _ir.App, bounds: _Bounds) -> bool:
    """Argument by argument in the declared variance; identical if invariant."""
    signs = _variances_at(sub.origin, len(sub.args))
    if signs is None or len(sup.args) != len(signs):
        return _ir.subtype(sub, sup)
    return all(
        _implies(_ir.term_node(a), _ir.term_node(b), bounds)
        if sign == _ir.COVARIANT
        else _implies(_ir.term_node(b), _ir.term_node(a), bounds)
        if sign == _ir.CONTRAVARIANT
        else a == b
        for a, b, sign in zip(sub.args, sup.args, signs, strict=True)
    )


def _implies_attr(sub: _ir.Has, sup: _ir.Has, bounds: _Bounds) -> bool:
    """Every read of `sup` is implied by a read of `sub`, and every write likewise."""
    own, other = _signed(sub.args), _signed(sup.args)
    if own is None or other is None:
        return sub == sup
    reads, writes = _reads_writes(own)
    wider_reads, wider_writes = _reads_writes(other)
    return all(any(_implies(r, w, bounds) for r in reads) for w in wider_reads) and all(
        any(_implies(w, r, bounds) for r in writes) for w in wider_writes
    )


def _same_layout(params: _ir.Terms, wider: _ir.Terms) -> bool:
    """Positional parameters in equal number, unpacked at the same positions."""
    return (
        len(params) == len(wider)
        and not any(isinstance(p, _ir.Arg) for p in (*params, *wider))
        and all(
            isinstance(p, _ir.Unpack) == isinstance(w, _ir.Unpack)
            for p, w in zip(params, wider, strict=True)
        )
    )


def _implies(  # ruff: ignore[too-many-return-statements]
    sub: _ir.Node,
    sup: _ir.Node,
    bounds: _Bounds,
) -> bool:
    """Whether requiring `sub` requires `sup`: `_ir.subtype`, plus what the IR does
    not know: a typevar's bounds, a shipped protocol's declared variance, and the
    reads and writes of an attribute."""
    if sub == sup:
        return True
    match sub, sup:
        case _ir.Variance(_, part), _:
            return _implies(part, sup, bounds)
        case _, _ir.Variance(_, part):
            return _implies(sub, part, bounds)
        case _, _ir.Intersection(parts):
            return all(_implies(sub, p, bounds) for p in parts)
        case _ir.Intersection(parts), _:
            return any(_implies(p, sup, bounds) for p in parts)
        case _ir.Name(name), _ if any(
            _implies(b, sup, bounds) for b in bounds.get(name, ())
        ):
            return True
        case _ir.App(origin), _ir.App(wider) if origin == wider:
            return _implies_args(sub, sup, bounds)
        case _ir.Has(attr), _ir.Has(wider_attr) if attr == wider_attr:
            return _implies_attr(sub, sup, bounds)
        case _ir.Fn(params, ret), _ir.Fn(wider_params, wider_ret):
            return (
                _same_layout(params, wider_params)
                and _implies(ret, wider_ret, bounds)
                and all(
                    _implies(_ir.term_node(w), _ir.term_node(p), bounds)
                    for p, w in zip(params, wider_params, strict=True)
                )
            )
        case _:
            return _ir.subtype(sub, sup)


def _merge_apps(first: _ir.App, second: _ir.App, bounds: _Bounds) -> _ir.App | None:
    """One application of a protocol that `first` and `second` both require, if at
    most one argument differs and the two are ordered by implication: the narrower
    one wins in a covariant position, the wider one in a contravariant position.

    Both must apply every parameter: an omitted one defaults to another, which a join
    would then change as well. Two differing arguments may be correlated, as in
    `CanSetitem[int, int] & CanSetitem[str, str]`, so they stay apart.
    """
    signs = _variances_at(first.origin, len(first.args))
    if signs is None or len(second.args) != len(signs):
        return None
    pairs = zip(first.args, second.args, strict=True)
    differing = [i for i, (x, y) in enumerate(pairs) if x != y]
    if not differing:
        return first
    if len(differing) != 1:
        return None
    (i,) = differing
    x, y = _ir.term_node(first.args[i]), _ir.term_node(second.args[i])
    if _implies(x, y, bounds):
        narrower, wider = x, y
    elif _implies(y, x, bounds):
        narrower, wider = y, x
    else:
        return None
    if signs[i] == _ir.COVARIANT:
        joined = narrower
    elif signs[i] == _ir.CONTRAVARIANT:
        joined = wider
    else:
        return None
    return _ir.App(first.origin, (*first.args[:i], joined, *first.args[i + 1 :]))


def _merge_parts(parts: Sequence[_ir.Node], bounds: _Bounds) -> list[_ir.Node]:
    """Join the applications of one protocol, which a class cannot inherit twice."""
    out: list[_ir.Node] = []
    for part in parts:
        for i, prev in enumerate(out):
            if (
                isinstance(part, _ir.App)
                and isinstance(prev, _ir.App)
                and prev.origin == part.origin
                and (joined := _merge_apps(prev, part, bounds)) is not None
            ):
                out[i] = joined
                break
        else:
            out.append(part)
    return out


def _fold_arities(parts: list[_ir.Node]) -> list[_ir.Node]:
    """Join the two arity forms of an operation into the protocol that has both."""
    for origin, (one, two) in (("CanRound", (1, 2)), ("CanPow", (2, 3))):
        apps = {
            len(p.args): p
            for p in parts
            if isinstance(p, _ir.App) and p.origin == origin
        }
        if not {one, two} <= apps.keys():
            continue
        first, second = apps[one], apps[two]
        if origin == "CanRound":
            (r1,), (n, r2) = first.args, second.args
            joined = _ir.App(origin, (n, r1, r2))
        else:
            (t, r2), (t2, v, r3) = first.args, second.args
            if not (_ir.subtype(t, t2) and _ir.subtype(t2, t)):
                continue
            joined = _ir.App(origin, (t, v, r2, r3))
        parts = [joined if p is first else p for p in parts if p is not second]
    return parts


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
class Lowerer:
    """Rewrite a sequence of `Signature`s into a printable `Module`.

    The helper definitions are shared across every signature of one render.
    """

    defs: dict[str, Helper]
    groups: dict[_GroupKey, list[str]]
    _keys: dict[_ProtoKey, str]
    _names: set[str]  # every claimed helper name

    def __init__(self) -> None:
        self.defs = {}
        self.groups = {}
        self._keys = {}
        self._names = set()

    def module(self, sigs: Sequence[_ir.Signature]) -> Module:
        funcs = [_SigLowerer(self, sig).func() for sig in sigs]
        return Module(tuple(_reachable(self.defs, funcs)), tuple(funcs))

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


def _mentions(node: _ir.Term) -> Iterable[str]:
    # the applied and bare names in `node`, helpers included
    match node:
        case _ir.Name(name):
            yield name
        case _ir.App(origin, parts) | _ir.Has(origin, parts):
            yield origin
            for part in parts:
                yield from _mentions(part)
        case _ir.Union(parts) | _ir.Intersection(parts):
            for part in parts:
                yield from _mentions(part)
        case _ir.Fn(params, ret):
            for part in (*params, ret):
                yield from _mentions(part)
        case (
            _ir.Arg(value=part)
            | _ir.Not(part)
            | _ir.Variance(part=part)
            | _ir.Unpack(part)
        ):
            yield from _mentions(part)
        case _:
            return


def _typar_nodes(typars: Iterable[_ir.TypeParam]) -> Iterable[_ir.Node]:
    for typar in typars:
        yield from (n for n in (typar.bound, typar.default) if n is not None)


def _reachable(
    defs: Mapping[str, Helper],
    funcs: Iterable[_ir.Signature],
) -> list[Helper]:
    """The helpers the functions mention, directly or through other helpers, in the
    order they were registered; a bound lowered more than once leaves unused ones."""
    todo = [
        name
        for func in funcs
        for node in (
            *(p.node for p in func.params),
            func.ret,
            *_typar_nodes(func.type_params),
        )
        for name in _mentions(node)
    ]
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name in seen or (helper := defs.get(name)) is None:
            continue
        seen.add(name)
        nodes: list[_ir.Term] = list(_typar_nodes(helper.type_params))
        if isinstance(helper, ProtocolDef):
            nodes += [*helper.bases, *member_nodes(helper.members)]
        else:
            nodes.append(helper.value)
        todo.extend(name for node in nodes for name in _mentions(node))
    return [helper for name, helper in defs.items() if name in seen]


def _signed(args: tuple[_ir.Node, ...]) -> tuple[_ir.Node, ...] | None:
    """`args` as reads and writes; a lone method is a read of its callable type."""
    if all(isinstance(arg, _ir.Variance) for arg in args):
        return args
    if len(args) == 1 and isinstance(args[0], _ir.Fn):
        return (_ir.Variance(_ir.COVARIANT, args[0]),)
    return None


def _order_typars(typars: list[_ir.TypeParam]) -> list[_ir.TypeParam]:
    """PEP 696 order: no default right after a typevar tuple, and no parameter without
    one after a default, so the tuple goes last with an empty default."""
    if not any(typar.default is not None for typar in typars):
        return typars
    empty = _ir.Unpack(_ir.App("tuple", ()))
    tuples = [replace(t, default=empty) for t in typars if t.unpack]
    return [t for t in typars if not t.unpack] + tuples


def _distinct_text(nodes: Iterable[_ir.Node]) -> list[_ir.Node]:
    out: list[_ir.Node] = []
    seen: set[str] = set()
    for node in nodes:
        if (key := type_text(node)) not in seen:
            seen.add(key)
            out.append(node)
    return out


def _constrain(
    constraints: _Constraints,
    tyvar: str,
    parts: Iterable[_ir.Node],
) -> None:
    """Add the `parts` that `tyvar` is not constrained by yet; the bounds are lowered
    more than once, which must not grow them."""
    known = constraints.setdefault(tyvar, [])
    known.extend(p for p in parts if p not in known)


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

    _registry: Lowerer
    _sig: _ir.Signature
    _tyvars: frozenset[str]  # the signature's type parameter names

    def __init__(self, registry: Lowerer, sig: _ir.Signature) -> None:
        self._registry = registry
        self._sig = sig
        self._tyvars = frozenset(typar.name for typar in sig.type_params)

    def func(self) -> _ir.Signature:
        sig = self._sig
        constraints: _Constraints = {}
        params = [replace(p, node=self._node(p.node, constraints)) for p in sig.params]
        ret = self._node(sig.ret, constraints)
        kept, subst = self._resolve_typars(sig.type_params, constraints)
        params = [replace(p, node=_ir.subst(p.node, subst)) for p in params]
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
        # lowering a bound may lift a requirement into another typevar's constraints,
        # so the bounds are lowered again until no constraint is new
        counts: dict[str, int] = {}
        while True:
            for typar in typars:
                extra = constraints.get(typar.name)
                merged = self._merge_bound(typar.bound, extra, constraints)
                bound[typar.name] = None if merged == _ir.OBJECT else merged
            if (found := {n: len(c) for n, c in constraints.items()}) == counts:
                break
            counts = found
        default = {
            typar.name: None if typar.default is None else self._node(typar.default, {})
            for typar in typars
        }

        elim = {
            tyvar: b for tyvar, b in bound.items() if b and is_generic(b, self._tyvars)
        }
        deps = {tyvar: frozenset(_ir.names(b)) & set(elim) for tyvar, b in elim.items()}
        cyclic = cyclic_names(deps)

        # each unit sees the substitutions of the units it depends on, so no
        # eliminated name survives in a hoisted body or a substituted bound
        subst: dict[str, _ir.Node] = {}
        for unit in resolution_order(
            components(cyclic, deps),
            set(elim) - cyclic,
            deps,
        ):
            resolved = {tyvar: _ir.subst(elim[tyvar], subst) for tyvar in unit}
            if unit <= cyclic:
                subst |= self._hoist_group(unit, resolved)
            else:
                subst |= resolved

        kept = [
            replace(
                typar,
                bound=bound[typar.name],
                default=None
                if (d := default[typar.name]) is None
                else _ir.subst(d, subst),
            )
            for typar in typars
            if typar.name not in elim
        ]
        return _order_typars(kept), subst

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
        origin = _ARITY_FORMS.get((origin, len(args)), origin)
        lowered = [self._arg(a, constraints) for a in args]
        if (bounds := _protocol_bounds(origin)) is not None:
            lowered = lowered[: len(bounds)]
            for arg, bound in zip(lowered, bounds, strict=False):
                if bound and isinstance(arg, _ir.Name) and arg.name in self._tyvars:
                    _constrain(constraints, arg.name, [bound])
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
            _constrain(constraints, tyvar, extra)
            return _ir.Name(tyvar)

        bounds = {
            typar.name: [
                *([typar.bound] if typar.bound is not None else []),
                *constraints.get(typar.name, ()),
            ]
            for typar in self._sig.type_params
        }
        # `(A | B) & C` distributes to `(A & C) | (B & C)`: a union cannot be a base
        parts = _ir.distinct(parts)
        unions = [p for p in parts if isinstance(p, _ir.Union)]
        rest = [p for p in parts if not isinstance(p, (_ir.Union, _ir.Not))]
        lowered: list[tuple[_ir.Node, _ir.Node]] = []

        def lower(part: _ir.Node) -> _ir.Node:
            for raw, node in lowered:
                if raw == part:
                    return node
            node = self._node(part, constraints)
            lowered.append((part, node))
            return node

        variants: list[_ir.Node] = []
        for picks in product(*(u.parts for u in unions)):
            joined = _fold_arities(_merge_parts([*rest, *picks], bounds))
            variants.append(self._combine(_distinct_text(map(lower, joined))))
        return _ir.union(variants) or _ir.OBJECT

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
        if not parts:
            return _ir.OBJECT
        if len(parts) == 1:
            return parts[0]
        apps = [p for p in parts if isinstance(p, _ir.App)]
        candidate = combine_name([p.origin for p in apps]) or "P"
        # a callable is not a valid base; it lifts into a `__call__` method instead,
        # and the two `pow` forms at different exponents, which share no shipped
        # protocol, into `__pow__` overloads
        pows = [p for p in apps if p.origin in {"CanPow2", "CanPow3"}]
        overloaded: list[_ir.App] = pows if len(pows) > 1 else []
        bases = [p for p in parts if not isinstance(p, _ir.Fn) and p not in overloaded]
        members = (
            *(
                Method("__call__", p.params, p.ret)
                for p in parts
                if isinstance(p, _ir.Fn)
            ),
            *(
                Method("__pow__", p.args[:-1], _ir.term_node(p.args[-1]))
                for p in overloaded
            ),
        )
        return self._proto_app(candidate, bases=bases, members=members)

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
        if not all(isinstance(s, _ir.Variance) for s in signed):
            node = self._node(strip_variance(signed[0]), constraints)
            cv = classvar and not is_generic(node, self._tyvars)
            return Attr(attr, node, classvar=cv)
        reads, writes = _reads_writes(signed)
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
        constraints: _Constraints,
    ) -> _ir.Node | None:
        parts: list[_ir.Node] = []
        if bound is not None:
            parts += bound.parts if isinstance(bound, _ir.Intersection) else [bound]
        if extra:
            parts += extra
        if not parts:
            return None
        node = parts[0] if len(parts) == 1 else _ir.Intersection(tuple(parts))
        return self._node(node, constraints)

    def _hoist_group(
        self,
        group: frozenset[str],
        bound: Mapping[str, _ir.Node],
    ) -> dict[str, _ir.Node]:
        """Turn a cyclic bound group into mutually-referential helper definitions."""
        members = sorted(group)
        free = free_tyvars([bound[tyvar] for tyvar in members], self._tyvars - group)
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
