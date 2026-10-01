"""`Lowerer`: rewrite the `Signature` IR into a printable `Module`.

It synthesizes helper `Protocol`s for intersections, the inline `Has[...]` form,
and keyword/defaulted callables. Dependent bounds and intersections with type
variables are unsupported. `docs/infer.md` is the source of truth.
"""

import builtins
import functools
import keyword
import typing
from collections.abc import Iterable, Sequence
from dataclasses import replace
from itertools import count, islice, product
from typing import final

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]
from ._model import (
    Attr,
    Member,
    Method,
    Module,
    ProtocolDef,
    combine_name,
    free_tyvars,
    is_generic,
    member_nodes,
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
def _protocol_typars(origin: str) -> tuple[object, ...] | None:
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
    if (typars := _protocol_typars(origin)) is None:
        return None
    return tuple(_bound_node(getattr(typar, "__bound__", None)) for typar in typars)


def _protocol_variances(origin: str) -> tuple[_ir.Variance | None, ...] | None:
    """The declared variance per argument of an `optype` protocol, if known."""
    if (typars := _protocol_typars(origin)) is None:
        return None
    return tuple(
        _ir.COVARIANT
        if getattr(typar, "__covariant__", False)
        else _ir.CONTRAVARIANT
        if getattr(typar, "__contravariant__", False)
        else None
        for typar in typars
    )


def _by_variance(args: tuple[_ir.Node, ...]) -> tuple[list[_ir.Node], list[_ir.Node]]:
    """The covariant and the contravariant `Has` arguments, unwrapped."""
    co = [a.part for a in args if isinstance(a, _ir.Covariant)]
    contra = [a.part for a in args if isinstance(a, _ir.Contravariant)]
    return co, contra


def _variances_at(origin: str, arity: int) -> tuple[_ir.Variance | None, ...] | None:
    """The declared variances of the shipped protocol behind `origin` at `arity`."""
    variances = _protocol_variances(_ARITY_FORMS.get((origin, arity), origin))
    return variances if variances is not None and len(variances) == arity else None


def _implies_args(sub: _ir.App, sup: _ir.App) -> bool:
    """Argument by argument in the declared variance; identical if invariant."""
    variances = _variances_at(sub.origin, len(sub.args))
    if variances is None or len(sup.args) != len(variances):
        return _ir.subtype(sub, sup)
    return all(
        _implies(_ir.term_node(a), _ir.term_node(b))
        if variance == _ir.COVARIANT
        else _implies(_ir.term_node(b), _ir.term_node(a))
        if variance == _ir.CONTRAVARIANT
        else a == b
        for a, b, variance in zip(sub.args, sup.args, variances, strict=True)
    )


def _implies_attr(sub: _ir.Has, sup: _ir.Has) -> bool:
    """Each covariant argument of `sup` is implied by one of `sub`, and each
    contravariant one implies one of `sub`."""
    own, other = _with_variance(sub.args), _with_variance(sup.args)
    if own is None or other is None:
        return sub == sup
    co, contra = _by_variance(own)
    wider_co, wider_contra = _by_variance(other)
    return all(any(_implies(c, w) for c in co) for w in wider_co) and all(
        any(_implies(w, c) for c in contra) for w in wider_contra
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


def _implies(sub: _ir.Node, sup: _ir.Node) -> bool:  # ruff: ignore[too-many-return-statements]
    """Whether requiring `sub` requires `sup`: `_ir.subtype`, plus what the IR does
    not know: a shipped protocol's declared variance and each `Has` argument's
    variance."""
    if sub == sup:
        return True
    match sub, sup:
        case _ir.Covariant(part) | _ir.Contravariant(part), _:
            return _implies(part, sup)
        case _, _ir.Covariant(part) | _ir.Contravariant(part):
            return _implies(sub, part)
        case _, _ir.Intersection(parts):
            return all(_implies(sub, p) for p in parts)
        case _ir.Intersection(parts), _:
            return any(_implies(p, sup) for p in parts)
        case _ir.App(origin), _ir.App(wider) if origin == wider:
            return _implies_args(sub, sup)
        case _ir.Has(attr), _ir.Has(wider_attr) if attr == wider_attr:
            return _implies_attr(sub, sup)
        case _ir.Fn(params, ret), _ir.Fn(wider_params, wider_ret):
            return (
                _same_layout(params, wider_params)
                and _implies(ret, wider_ret)
                and all(
                    _implies(_ir.term_node(w), _ir.term_node(p))
                    for p, w in zip(params, wider_params, strict=True)
                )
            )
        case _:
            return _ir.subtype(sub, sup)


def _merge_apps(first: _ir.App, second: _ir.App) -> _ir.App | None:
    """One application of a protocol that `first` and `second` both require, if at
    most one argument differs and the two are ordered by implication: the narrower
    one wins in a covariant position, the wider one in a contravariant position.

    Both must apply every parameter: an omitted one defaults to another, which a join
    would then change as well. Two differing arguments may be correlated, as in
    `CanSetitem[int, int] & CanSetitem[str, str]`, so they stay apart.
    """
    variances = _variances_at(first.origin, len(first.args))
    if variances is None or len(second.args) != len(variances):
        return None
    pairs = zip(first.args, second.args, strict=True)
    differing = [i for i, (x, y) in enumerate(pairs) if x != y]
    if not differing:
        return first
    if len(differing) != 1:
        return None
    (i,) = differing
    x, y = _ir.term_node(first.args[i]), _ir.term_node(second.args[i])
    if _implies(x, y):
        narrower, wider = x, y
    elif _implies(y, x):
        narrower, wider = y, x
    else:
        return None
    if variances[i] == _ir.COVARIANT:
        joined = narrower
    elif variances[i] == _ir.CONTRAVARIANT:
        joined = wider
    else:
        return None
    return _ir.App(first.origin, (*first.args[:i], joined, *first.args[i + 1 :]))


def _merge_parts(parts: Sequence[_ir.Node]) -> list[_ir.Node]:
    """Join the applications of one protocol, which a class cannot inherit twice."""
    out: list[_ir.Node] = []
    for part in parts:
        for i, prev in enumerate(out):
            if (
                isinstance(part, _ir.App)
                and isinstance(prev, _ir.App)
                and prev.origin == part.origin
                and (joined := _merge_apps(prev, part)) is not None
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
    tyvars: frozenset[str],
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
            if bound is not None and isinstance(arg, _ir.Name) and arg.name in tyvars:
                result.setdefault(arg.name, bound)
    return result


# the canonical dedup keys: equal definitions serialize to equal keys
type _ProtoKey = tuple[tuple[str, ...], tuple[str, ...]]  # bases, then members


@final
class Lowerer:
    """Rewrite a sequence of `Signature`s into a printable `Module`.

    The helper definitions are shared across every signature of one render.
    """

    defs: dict[str, ProtocolDef]
    keys: dict[_ProtoKey, str]
    _names: set[str]  # every claimed helper name

    def __init__(self) -> None:
        self.defs = {}
        self.keys = {}
        self._names = set()

    def module(self, sigs: Sequence[_ir.Signature]) -> Module:
        # Reject complements before intersections or bounds can erase them.
        for sig in sigs:
            nodes = [
                *(p.node for p in sig.params),
                sig.ret,
                *_typar_nodes(sig.type_params),
            ]
            if any(
                isinstance(part, _ir.Not) for node in nodes for part in _ir.walk(node)
            ):
                msg = "compat cannot preserve type complements; use backend='terse'"
                raise InferError(msg)
            tyvars = frozenset(p.name for p in sig.type_params)
            if any(
                p.bound is not None and is_generic(p.bound, tyvars)
                for p in sig.type_params
            ):
                msg = (
                    "compat cannot preserve typevar-referencing bounds; "
                    "use backend='terse'"
                )
                raise InferError(msg)
        funcs = tuple(_SigLowerer(self, sig).func() for sig in sigs)
        return Module(_used_helpers(self.defs, funcs), funcs)

    def claim(self, candidate: str) -> str:
        """A fresh helper name that collides with no real import or other helper."""
        # a qualified origin (`collections.OrderedDict`) is named after its class,
        # which must not shadow the module it is imported by
        module, _, candidate = candidate.rpartition(".")
        name = candidate
        i = 2
        while (
            name in self._names
            or name == module.partition(".")[0]
            or import_of(name) is not None
            or hasattr(builtins, name)
        ):
            name = f"{candidate}{i}"
            i += 1
        self._names.add(name)
        return name


def _typar_nodes(typars: Iterable[_ir.TypeParam]) -> Iterable[_ir.Node]:
    for typar in typars:
        yield from (n for n in (typar.bound, typar.default) if n is not None)


def _used_helpers(
    defs: dict[str, ProtocolDef],
    funcs: Sequence[_ir.Signature],
) -> tuple[ProtocolDef, ...]:
    def references(nodes: Iterable[_ir.Term]) -> set[str]:
        return {
            part.name if isinstance(part, _ir.Name) else part.origin
            for node in nodes
            for part in _ir.walk(node)
            if isinstance(part, (_ir.Name, _ir.App))
        }

    needed = references(
        node
        for f in funcs
        for node in (*(p.node for p in f.params), f.ret, *_typar_nodes(f.type_params))
    )
    used: list[ProtocolDef] = []
    # Helpers are registered after their dependencies; none are recursive.
    for helper in reversed(defs.values()):
        if helper.name in needed:
            used.append(helper)
            needed.update(
                references([
                    *helper.bases,
                    *member_nodes(helper.members),
                    *_typar_nodes(helper.type_params),
                ]),
            )
    return tuple(reversed(used))


def _with_variance(args: tuple[_ir.Node, ...]) -> tuple[_ir.Node, ...] | None:
    """The `Has` arguments, each in its variance position, or `None` if one has no
    variance (a `ClassVar` or a bare type). A lone method is only read, so its
    callable type is covariant."""
    if all(isinstance(arg, (_ir.Covariant, _ir.Contravariant)) for arg in args):
        return args
    if len(args) == 1 and isinstance(args[0], _ir.Fn):
        return (_ir.Covariant(args[0]),)
    return None


def _order_typars(typars: list[_ir.TypeParam]) -> list[_ir.TypeParam]:
    """PEP 696 order: no default right after a typevar tuple, and no parameter without
    one after a default, so the tuple goes last with an empty default."""
    if not any(typar.default is not None for typar in typars):
        return typars
    empty = _ir.Unpack(_ir.App("tuple", ()))
    tuples = [replace(typar, default=empty) for typar in typars if typar.unpack]
    return [typar for typar in typars if not typar.unpack] + tuples


def _distinct_text(nodes: Iterable[_ir.Node]) -> list[_ir.Node]:
    out: list[_ir.Node] = []
    seen: set[str] = set()
    for node in nodes:
        if (key := type_text(node)) not in seen:
            seen.add(key)
            out.append(node)
    return out


def _merge_has(parts: Sequence[_ir.Node]) -> list[_ir.Node]:
    """Join the `Has` members of one attribute into one with all of their arguments."""
    out: list[_ir.Node] = []
    index: dict[str, int] = {}
    for part in parts:
        if (
            isinstance(part, _ir.Has)
            and (args := _with_variance(part.args)) is not None
        ):
            if (i := index.get(part.attr)) is not None:
                prev = out[i]
                assert isinstance(prev, _ir.Has)
                prev_args = _with_variance(prev.args)
                assert prev_args is not None
                out[i] = _ir.Has(part.attr, prev_args + args)
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
    _bounds: dict[str, list[_ir.Node]]

    def __init__(self, registry: Lowerer, sig: _ir.Signature) -> None:
        self._registry = registry
        self._sig = sig
        self._tyvars = frozenset(typar.name for typar in sig.type_params)
        self._bounds = {}

    def func(self) -> _ir.Signature:
        sig = self._sig
        params = tuple(replace(p, node=self._node(p.node)) for p in sig.params)
        ret = self._node(sig.ret)
        defaults = {
            p.name: None if p.default is None else self._node(p.default)
            for p in sig.type_params
        }
        typars: list[_ir.TypeParam] = []
        for p in sig.type_params:
            parts = [p.bound] if p.bound is not None else []
            parts += self._bounds.get(p.name, [])
            bound = _ir.intersection(parts)
            bound = None if bound is None else self._node(bound)
            typars.append(replace(p, bound=bound, default=defaults[p.name]))
        return replace(
            sig,
            type_params=tuple(_order_typars(typars)),
            params=params,
            ret=ret,
        )

    def _node(self, node: _ir.Node) -> _ir.Node:  # ruff: ignore[too-many-return-statements]
        match node:
            case _ir.Has(attr, args):
                return self._has(attr, args)
            case _ir.App(origin, args):
                return self._app(origin, args)
            case _ir.Fn(params, ret):
                return self._fn(params, ret)
            case _ir.Union(parts):
                lowered = [self._node(p) for p in parts]
                return _ir.union(lowered) or _ir.OBJECT
            case _ir.Intersection(parts):
                return self._inter(parts)
            case _ir.Unpack(part):
                return _ir.Unpack(self._node(part))
            case _ir.Covariant(part) | _ir.Contravariant(part):
                return self._node(part)
            case _:
                return node

    def _arg(self, arg: _ir.Term) -> _ir.Term:
        if isinstance(arg, _ir.Arg):
            return replace(arg, value=self._node(arg.value))

        return self._node(arg)

    def _app(self, origin: str, args: _ir.Terms) -> _ir.App:
        origin = _ARITY_FORMS.get((origin, len(args)), origin)
        bounds = _protocol_bounds(origin)
        if bounds is not None:
            args = args[: len(bounds)]
        lowered = tuple(self._arg(a) for a in args)
        for arg, bound in zip(lowered, bounds or (), strict=False):
            if bound and isinstance(arg, _ir.Name) and arg.name in self._tyvars:
                self._bounds.setdefault(arg.name, []).append(bound)
        return _ir.App(origin, lowered)

    def _inter(self, parts: Sequence[_ir.Node]) -> _ir.Node:
        parts = _merge_has(parts)
        # `(A | B) & C` distributes to `(A & C) | (B & C)`: a union cannot be a base
        parts = _ir.distinct(parts)
        unions = [p for p in parts if isinstance(p, _ir.Union)]
        rest = [p for p in parts if not isinstance(p, _ir.Union)]
        lowered: list[tuple[_ir.Node, _ir.Node]] = []

        def lower(part: _ir.Node) -> _ir.Node:
            for raw, node in lowered:
                if raw == part:
                    return node
            node = self._node(part)
            lowered.append((part, node))
            return node

        variants: list[_ir.Node] = []
        for picks in product(*(u.parts for u in unions)):
            joined = _fold_arities(_merge_parts([*rest, *picks]))
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
        inherited = _inherited_bounds(canon_bases, frozenset(canon_names))
        typars = tuple(_ir.TypeParam(c, inherited.get(c)) for c in canon_names)
        key = (
            tuple(type_text(b) for b in canon_bases),
            tuple(map(member_key, canon_members)),
        )
        registry = self._registry
        if (name := registry.keys.get(key)) is None:
            name = registry.keys[key] = registry.claim(candidate)
            registry.defs[name] = ProtocolDef(name, typars, canon_bases, canon_members)
        return _ir.App(name, tuple(_ir.Name(f) for f in fv))

    def _combine(self, parts: Sequence[_ir.Node]) -> _ir.Node:
        if not parts:
            return _ir.OBJECT
        if len(parts) == 1:
            return parts[0]
        if any(isinstance(p, _ir.Name) and p.name in self._tyvars for p in parts):
            msg = (
                "compat cannot preserve intersections with type variables; "
                "use backend='terse'"
            )
            raise InferError(msg)

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
        args: tuple[_ir.Node, ...],
    ) -> _ir.Node:
        if not attr.isidentifier() or keyword.iskeyword(attr):
            # a protocol member must be named for the real attribute, so a name that is
            # not a valid identifier (e.g. from `getattr(x, "a-b")`) is inexpressible
            msg = f"cannot render attribute {attr!r} as a protocol member"
            raise InferError(msg)

        member = self._has_member(attr, args)
        candidate = "Has" + attr[:1].upper() + attr[1:]
        return self._proto_app(candidate, members=(member,))

    def _has_member(
        self,
        attr: str,
        args: tuple[_ir.Node, ...],
        *,
        classvar: bool = False,
    ) -> Member:
        if (
            len(args) == 1
            and isinstance(args[0], _ir.App)
            and args[0].origin == "ClassVar"
        ):
            inner = tuple(_ir.term_node(a) for a in args[0].args)
            return self._has_member(attr, inner, classvar=True)
        if not args:
            return Attr(attr, _ir.OBJECT, classvar=classvar, readonly=not classvar)
        if len(args) == 1 and isinstance(args[0], _ir.Fn):
            if classvar:
                msg = (
                    "compat cannot preserve class-level callable attributes; "
                    "use backend='terse'"
                )
                raise InferError(msg)
            fn = args[0]
            ret = self._node(fn.ret)
            params = tuple(self._arg(p) for p in fn.params)
            return Method(attr, params, ret)
        co, contra = _by_variance(args)
        getter = _ir.intersection(co) or _ir.OBJECT
        setter = _ir.union(contra) or _ir.OBJECT
        # An instance member does not satisfy a requirement on the class itself.
        nodes = (getter, setter) if contra else (getter,)
        if classvar:
            if any(is_generic(n, self._tyvars) for n in nodes):
                msg = (
                    "compat cannot preserve generic class attributes; "
                    "use backend='terse'"
                )
                raise InferError(msg)
            return Attr(attr, self._node(getter if co else setter), classvar=True)
        # a dunder has a declared type, which a settable property would override
        # incompatibly
        if contra and attr.startswith("__") and attr.endswith("__"):
            return Attr(attr, self._node(getter if co else setter))
        # a property with a setter is what a plain attribute and a settable property
        # both satisfy; an annotation is only matched by a plain attribute
        return Attr(
            attr,
            self._node(getter),
            readonly=not contra,
            setter=self._node(setter) if contra else None,
        )

    def _fn(
        self,
        params: _ir.Terms,
        ret: _ir.Node,
    ) -> _ir.Node:
        lowered_ret = self._node(ret)
        lowered = tuple(self._arg(p) for p in params)
        # `Callable` covers positional params; a keyword or default needs `__call__`
        if not any(isinstance(p, _ir.Arg) and (p.key or p.default) for p in lowered):
            return _ir.Fn(lowered, lowered_ret)
        member = Method("__call__", lowered, lowered_ret)
        return self._proto_app("CanCallP", members=(member,))
