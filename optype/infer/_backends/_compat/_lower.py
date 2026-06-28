"""`_Lowerer`: rewrite the `Signature` IR into a printable `_Module`.

It synthesizes helper `Protocol`s (and recursive aliases) for the constructs the typing
spec cannot express: intersections, the inline `Has[...]` form, typevar-referencing
bounds, and keyword/defaulted callables. `docs/infer.md` is the source of truth.
"""

import ast
import builtins
import keyword
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from itertools import product
from typing import final

# `from optype.infer import _ir` would re-enter the package, which imports this module
import optype.infer._ir as _ir  # noqa: PLR0402
from ._model import (
    _OBJECT,
    _Alias,
    _Attr,
    _bound_name,
    _combine_name,
    _components,
    _cyclic,
    _free_vars,
    _Func,
    _is_generic,
    _is_protocol_node,
    _Member,
    _member_nodes,
    _Method,
    _Module,
    _Protocol,
    _strip_variance,
    _subst,
    _subst_member,
    _toposort,
    _value,
)
from ._print import _OPTYPE, _import_of, _member_key, _type
from optype.infer._errors import InferError

__all__ = ("_Lowerer",)


def _bound_node(bound: object) -> _ir.Node | None:
    """A typevar bound as an IR node: a plain type, or a union of plain types."""
    if isinstance(bound, type):
        return _ir.Type(bound)
    args = typing.get_args(bound)
    if args and all(isinstance(a, type) for a in args):
        return _ir.union([_ir.Type(a) for a in args])
    return None


_param_bounds: dict[str, tuple[_ir.Node | None, ...] | None] = {}


def _protocol_bounds(base: str) -> tuple[_ir.Node | None, ...] | None:
    """The per-argument typevar bounds of an `optype` protocol, or `None` if unknown.

    Reconciles the inferred `App` with the shipped generic: a non-generic protocol
    reports `()` (so excess arguments drop), and a bounded argument lets the matching
    inferred typevar pick up that bound.
    """
    if base not in _OPTYPE:
        return None
    if base not in _param_bounds:
        import optype  # noqa: PLC0415

        cls = getattr(optype, base, None)
        params = getattr(cls, "__parameters__", None)
        _param_bounds[base] = (
            None
            if params is None
            else tuple(_bound_node(getattr(p, "__bound__", None)) for p in params)
        )
    return _param_bounds[base]


def _inherited_bounds(
    bases: Sequence[_ir.Node],
    params: frozenset[str],
) -> dict[str, _ir.Node]:
    """Each helper type parameter's bound, taken from the protocol base it fills."""
    result: dict[str, _ir.Node] = {}
    for base in bases:
        if (
            not isinstance(base, _ir.App)
            or (bounds := _protocol_bounds(base.base)) is None
        ):
            continue
        for arg, bound in zip(base.args, bounds, strict=False):
            if bound is not None and isinstance(arg, _ir.Name) and arg.name in params:
                result.setdefault(arg.name, bound)
    return result


type _Constraints = dict[str, list[_ir.Node]]


@final
class _Lowerer:
    """Rewrite a sequence of `Signature`s into a printable `_Module`."""

    def __init__(self) -> None:
        self._defs: dict[str, _Protocol | _Alias] = {}
        self._keys: dict[object, str] = {}
        self._groups: dict[object, list[str]] = {}
        self._names: set[str] = set()

    def module(self, sigs: Sequence[_ir.Signature]) -> _Module:
        funcs = [self._func(sig) for sig in sigs]
        return _Module(tuple(self._defs.values()), tuple(funcs))

    def _func(self, sig: _ir.Signature) -> _Func:
        typevars = frozenset(tp.name for tp in sig.type_params)
        constraints: _Constraints = {}
        params = [self._param(p, typevars, constraints) for p in sig.params]
        ret = self._node(sig.ret, typevars, constraints)
        kept, subst = self._resolve_type_params(sig.type_params, typevars, constraints)
        params = [self._subst_param(p, subst) for p in params]
        return _Func(tuple(kept), tuple(params), _subst(ret, subst), sig.deprecated)

    def _resolve_type_params(
        self,
        type_params: Sequence[_ir.TypeParam],
        typevars: frozenset[str],
        constraints: _Constraints,
    ) -> tuple[list[_ir.TypeParam], dict[str, _ir.Node]]:
        """Eliminate typevar-referencing bounds, which PEP 695 forbids.

        An acyclic bound substitutes in place; a cyclic (self- or mutually-referential)
        one becomes a helper `Protocol`.
        """
        bound: dict[str, _ir.Node | None] = {}
        default: dict[str, _ir.Node | None] = {}
        for tp in type_params:
            merged = self._merge_bound(tp.bound, constraints.get(tp.name), typevars)
            bound[tp.name] = None if merged == _OBJECT else merged
            default[tp.name] = (
                None if tp.default is None else self._node(tp.default, typevars, {})
            )

        elim = {tv: b for tv, b in bound.items() if b and _is_generic(b, typevars)}
        deps = {tv: frozenset(_ir.names(b)) & set(elim) for tv, b in elim.items()}
        cyclic = _cyclic(deps)

        subst: dict[str, _ir.Node] = {}
        for group in _components(cyclic, deps):
            subst |= self._hoist_group(group, elim, typevars)
        for tv in _toposort(set(elim) - cyclic, deps):
            subst[tv] = _subst(elim[tv], subst)

        kept = [
            replace(tp, bound=bound[tp.name], default=default[tp.name])
            for tp in type_params
            if tp.name not in elim
        ]
        return kept, subst

    def _param(
        self,
        param: _ir.Param,
        typevars: frozenset[str],
        constraints: _Constraints,
    ) -> _ir.Param:
        return replace(param, node=self._node(param.node, typevars, constraints))

    @staticmethod
    def _subst_param(param: _ir.Param, subst: Mapping[str, _ir.Node]) -> _ir.Param:
        return replace(param, node=_subst(param.node, subst))

    def _node(  # noqa: PLR0911
        self,
        node: _ir.Node,
        typevars: frozenset[str],
        constraints: _Constraints,
    ) -> _ir.Node:
        match node:
            case _ir.App("Has", args):
                first, *signed = (_value(a) for a in args)
                return self._has(first, tuple(signed), typevars, constraints)
            case _ir.App(base, args):
                return self._app(base, args, typevars, constraints)
            case _ir.Fn(params, ret):
                return self._fn(params, ret, typevars, constraints)
            case _ir.Union(parts):
                lowered = [self._node(p, typevars, constraints) for p in parts]
                return _ir.union(lowered) or _OBJECT
            case _ir.Inter(parts):
                return self._inter(parts, typevars, constraints)
            case _ir.Not(_):
                return _OBJECT
            case _ir.Unpack(part):
                return _ir.Unpack(self._node(part, typevars, constraints))
            case _ir.Variance(_, part):
                return self._node(part, typevars, constraints)
            case _:
                return node

    def _arg(
        self,
        arg: _ir.Node | _ir.Arg,
        typevars: frozenset[str],
        constraints: _Constraints,
    ) -> _ir.Node | _ir.Arg:
        if isinstance(arg, _ir.Arg):
            return replace(arg, value=self._node(arg.value, typevars, constraints))
        return self._node(arg, typevars, constraints)

    def _app(
        self,
        base: str,
        args: tuple[_ir.Node | _ir.Arg, ...],
        typevars: frozenset[str],
        constraints: _Constraints,
    ) -> _ir.App:
        lowered = [self._arg(a, typevars, constraints) for a in args]
        if (bounds := _protocol_bounds(base)) is not None:
            lowered = lowered[: len(bounds)]
            for arg, bound in zip(lowered, bounds, strict=False):
                if bound and isinstance(arg, _ir.Name) and arg.name in typevars:
                    constraints.setdefault(arg.name, []).append(bound)
        return _ir.App(base, tuple(lowered))

    def _inter(
        self,
        parts: Sequence[_ir.Node],
        typevars: frozenset[str],
        constraints: _Constraints,
    ) -> _ir.Node:
        # a typevar member lifts the others into that typevar's bound, which is sound
        tv = next(
            (p.name for p in parts if isinstance(p, _ir.Name) and p.name in typevars),
            None,
        )
        if tv is not None:
            extra = [
                p
                for p in parts
                if not isinstance(p, _ir.Not)
                and not (isinstance(p, _ir.Name) and p.name == tv)
            ]
            constraints.setdefault(tv, []).extend(extra)
            return _ir.Name(tv)

        lowered: list[_ir.Node] = []
        seen: set[str] = set()
        for part in parts:
            if isinstance(part, _ir.Not):
                continue
            node = self._node(part, typevars, constraints)
            if (key := _type(node)) not in seen:
                seen.add(key)
                lowered.append(node)
        return self._combine(lowered, typevars)

    def _proto_app(
        self,
        candidate: str,
        typevars: frozenset[str],
        *,
        bases: Sequence[_ir.Node] = (),
        members: Sequence[_Member] = (),
    ) -> _ir.App:
        """Register a helper `Protocol` (canonicalized for reuse) and apply it."""
        fv = _free_vars([*bases, *_member_nodes(members)], typevars)
        m = {name: _ir.Name(_ir.typevar_name(i)) for i, name in enumerate(fv)}
        canon_bases = tuple(_subst(b, m) for b in bases)
        canon_members = tuple(_subst_member(mem, m) for mem in members)
        canon_names = [_ir.typevar_name(i) for i in range(len(fv))]
        tp_bounds = _inherited_bounds(canon_bases, frozenset(canon_names))
        tps = tuple(_ir.TypeParam(c, tp_bounds.get(c)) for c in canon_names)
        key = (
            tuple(_type(b) for b in canon_bases),
            tuple(map(_member_key, canon_members)),
        )
        name = self._register(
            candidate,
            key,
            lambda nm: _Protocol(nm, tps, canon_bases, canon_members),
        )
        return _ir.App(name, tuple(_ir.Name(f) for f in fv))

    def _combine(self, parts: Sequence[_ir.Node], typevars: frozenset[str]) -> _ir.Node:
        # an `(A | B) & C` distributes to `(A & C) | (B & C)`: a union cannot be a base
        if not parts:
            return _OBJECT
        if len(parts) == 1:
            return parts[0]
        unions = [p for p in parts if isinstance(p, _ir.Union)]
        if not unions:
            # a callable is not a valid base; it lifts into a `__call__` method instead
            bases = [p for p in parts if not isinstance(p, _ir.Fn)]
            members = tuple(
                _Method("__call__", p.params, p.ret)
                for p in parts
                if isinstance(p, _ir.Fn)
            )
            candidate = (
                _combine_name([p.base for p in bases if isinstance(p, _ir.App)]) or "P"
            )
            return self._proto_app(candidate, typevars, bases=bases, members=members)
        rest = [p for p in parts if not isinstance(p, _ir.Union)]
        variants = [
            self._combine([*rest, *picks], typevars)
            for picks in product(*(u.parts for u in unions))
        ]
        return _ir.union(variants) or _OBJECT

    def _has(
        self,
        first: _ir.Node,
        signed: tuple[_ir.Node, ...],
        typevars: frozenset[str],
        constraints: _Constraints,
    ) -> _ir.Node:
        attr = (
            ast.literal_eval(first.name) if isinstance(first, _ir.Name) else str(first)
        )
        if not attr.isidentifier() or keyword.iskeyword(attr):
            # a protocol member must be named for the real attribute, so a name that is
            # not a valid identifier (e.g. from `getattr(x, "a-b")`) is inexpressible
            msg = f"cannot render attribute {attr!r} as a protocol member"
            raise InferError(msg)
        member = self._has_member(attr, signed, typevars, constraints)
        candidate = "Has" + attr[:1].upper() + attr[1:]
        return self._proto_app(candidate, typevars, members=(member,))

    def _has_member(
        self,
        attr: str,
        signed: tuple[_ir.Node, ...],
        typevars: frozenset[str],
        constraints: _Constraints,
        *,
        classvar: bool = False,
    ) -> _Member:
        if (
            len(signed) == 1
            and isinstance(signed[0], _ir.App)
            and signed[0].base == "ClassVar"
        ):
            inner = tuple(_value(a) for a in signed[0].args)
            return self._has_member(attr, inner, typevars, constraints, classvar=True)
        if not signed:
            return _Attr(attr, _OBJECT, classvar=classvar)
        if len(signed) == 1 and isinstance(signed[0], _ir.Fn):
            fn = signed[0]
            ret = self._node(_strip_variance(fn.ret), typevars, constraints)
            params = tuple(self._arg(p, typevars, constraints) for p in fn.params)
            return _Method(attr, params, ret)
        if len(signed) == 1 and isinstance(signed[0], _ir.Variance):
            sign, part = signed[0].sign, signed[0].part
            node = self._node(part, typevars, constraints)
            # `ClassVar` can't hold a typevar; a generic one demotes to instance read
            cv = classvar and not _is_generic(node, typevars)
            return _Attr(
                attr,
                node,
                classvar=cv,
                readonly=not cv and sign == _ir.COVARIANT,
            )
        # a read-and-write attribute is invariant: a plain annotation accepts both
        reads = [
            s.part
            for s in signed
            if isinstance(s, _ir.Variance) and s.sign == _ir.COVARIANT
        ]
        chosen = reads[0] if reads else _strip_variance(signed[0])
        node = self._node(chosen, typevars, constraints)
        cv = classvar and not _is_generic(node, typevars)
        return _Attr(attr, node, classvar=cv)

    def _fn(
        self,
        params: tuple[_ir.Node | _ir.Arg, ...],
        ret: _ir.Node,
        typevars: frozenset[str],
        constraints: _Constraints,
    ) -> _ir.Node:
        lowered_ret = self._node(ret, typevars, constraints)
        lowered = tuple(self._arg(p, typevars, constraints) for p in params)
        # `Callable` covers positional params; a keyword or default needs `__call__`
        if not any(isinstance(p, _ir.Arg) and (p.key or p.default) for p in lowered):
            return _ir.Fn(lowered, lowered_ret)
        member = _Method("__call__", lowered, lowered_ret)
        return self._proto_app("CanCallP", typevars, members=(member,))

    def _merge_bound(
        self,
        bound: _ir.Node | None,
        extra: list[_ir.Node] | None,
        typevars: frozenset[str],
    ) -> _ir.Node | None:
        parts: list[_ir.Node] = []
        if bound is not None:
            parts += bound.parts if isinstance(bound, _ir.Inter) else [bound]
        if extra:
            parts += extra
        if not parts:
            return None
        node = parts[0] if len(parts) == 1 else _ir.Inter(tuple(parts))
        return self._node(node, typevars, {})

    def _hoist_group(
        self,
        group: frozenset[str],
        bound: Mapping[str, _ir.Node],
        typevars: frozenset[str],
    ) -> dict[str, _ir.Node]:
        """Turn a cyclic bound group into mutually-referential helper definitions."""
        members = sorted(group)
        free = list(
            dict.fromkeys(
                name
                for tv in members
                for name in _ir.names(bound[tv])
                if name in typevars and name not in group
            ),
        )
        canon = [_ir.typevar_name(i) for i in range(len(free))]
        rename: dict[str, _ir.Node] = {
            f: _ir.Name(canon[i]) for i, f in enumerate(free)
        }
        roles: dict[str, _ir.Node] = {
            tv: _ir.Name(f"\0{i}") for i, tv in enumerate(members)
        }
        key = tuple(
            (_is_protocol_node(bound[tv]), _type(_subst(bound[tv], rename | roles)))
            for tv in members
        )

        if key not in self._groups:
            names = [self._claim(_bound_name(bound[tv], tv)) for tv in members]
            self._groups[key] = names
            refs: dict[str, _ir.Node] = {
                tv: _ir.App(names[i], tuple(map(_ir.Name, canon)))
                for i, tv in enumerate(members)
            }
            tps = tuple(_ir.TypeParam(c) for c in canon)
            for i, tv in enumerate(members):
                body = _subst(bound[tv], rename | refs)
                self._defs[names[i]] = (
                    _Protocol(names[i], tps, (body,), ())
                    if _is_protocol_node(bound[tv])
                    else _Alias(names[i], tps, body)
                )

        names = self._groups[key]
        site = tuple(map(_ir.Name, free))
        return {tv: _ir.App(names[i], site) for i, tv in enumerate(members)}

    def _register(
        self,
        candidate: str,
        key: object,
        build: Callable[[str], _Protocol | _Alias],
    ) -> str:
        if key in self._keys:
            return self._keys[key]
        name = self._claim(candidate)
        self._keys[key] = name
        self._defs[name] = build(name)
        return name

    def _claim(self, candidate: str) -> str:
        """A fresh helper name that collides with no real import or other helper."""
        name = candidate
        i = 2
        while (
            name in self._names
            or _import_of(name) is not None
            or hasattr(builtins, name)
        ):
            name = f"{candidate}{i}"
            i += 1
        self._names.add(name)
        return name
