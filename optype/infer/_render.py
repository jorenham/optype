"""Render the recorded spy traces as a PEP 695 signature."""

import sys
from collections import defaultdict
from collections.abc import Collection, Generator, Iterable, Mapping, Sequence
from inspect import Parameter, _ParameterKind
from itertools import groupby
from typing import Any, NamedTuple, cast, final

# `from . import` would import the package itself, which imports this module
import optype.infer._ir as _ir
import optype.infer._numpy as _numpy
from ._errors import InferError
from ._explore import _Recon, _Traces
from ._spy import (
    _AnyFunc,
    _Args,
    _class_spy,
    _Kwargs,
    _Marker,
    _own_spy,
    _Spy,
    _SpyBytes,
    _SpyObject,
    _SpyStr,
    _TraceItem,
    as_spy,
)
from ._values import _children, _Fn, _fn_spies, _Gen, _walk
from optype._core import _can, _has
from optype.inspect import get_protocol_members

__all__ = ("_Defaults", "_Names", "_signatures")

_PARAM_PREFIX: dict[_ParameterKind, str] = {
    Parameter.VAR_POSITIONAL: "*",
    Parameter.VAR_KEYWORD: "**",
}

_TYPEVARS = "TUVWXYZ"
_TYPEVAR_TUPLE = "*Ts"
_NEVER = "Never"
_OBJECT = "object"

_DUNDER_ATTR = frozenset({
    "__delattr__",
    "__getattr__",
    "__getattribute__",
    "__setattr__",
})


def _get_dunder_can_map() -> dict[str, str]:
    return {
        dunder: name
        for name in _can.__all__
        if not name.endswith(("Self", "Same"))
        if len(members := get_protocol_members(getattr(_can, name))) == 1
        if (dunder := next(iter(members))) not in _DUNDER_ATTR
        # CanPow2, CanRound1, ... share their dunder; keep the canonical protocol
        if dunder.replace("_", "") == name.removeprefix("Can").lower()
    } | _numpy.DUNDER_CAN_MAP


_DUNDER_CAN_MAP = _get_dunder_can_map()
_DUNDER_CAN_R = frozenset(
    dunder
    for dunder, proto in _DUNDER_CAN_MAP.items()
    if "CanR" + proto.removeprefix("Can") in _DUNDER_CAN_MAP.values()
)


def _get_dunder_has_map() -> dict[str, str]:
    return {
        next(iter(members)): name
        for name in _has.__all__
        if len(members := get_protocol_members(getattr(_has, name))) == 1
    }


_DUNDER_HAS_MAP = _get_dunder_has_map()

_COERCION_FALLBACK = {
    "__float__": ("__index__",),
    "__int__": ("__index__",),
    "__complex__": ("__float__", "__index__"),
}
_COERCION_PROTOS = {
    dunder: tuple(map(_DUNDER_CAN_MAP.__getitem__, (dunder, *fallback)))
    for dunder, fallback in _COERCION_FALLBACK.items()
}

type _Vars = dict[int, str]
type _RetKey = tuple[int, str, int, tuple[str, ...]]
type _Proto = str | tuple[str, ...]  # a tuple is rendered as a union of protocols

type _Names = Sequence[str]
type _Defaults = Mapping[str, object]


class _Op(NamedTuple):
    proto: _Proto
    args: _Args
    kwargs: _Kwargs
    ret: object


def _resolve(trace: _TraceItem) -> _Op:
    if trace.attr in _DUNDER_ATTR:
        name = trace.args[0]
        if not isinstance(name, str) or name not in _DUNDER_HAS_MAP:
            msg = f"no protocol for attribute {name!r}"
            raise InferError(msg)
        return _Op(_DUNDER_HAS_MAP[name], (), {}, trace.return_)
    # checked before _DUNDER_CAN_MAP, which also contains the coercion dunders
    if trace.attr in _COERCION_PROTOS:
        return _Op(_COERCION_PROTOS[trace.attr], (), {}, trace.return_)
    if trace.attr in _DUNDER_CAN_MAP:
        proto = _DUNDER_CAN_MAP[trace.attr]
        return _Op(proto, trace.args, trace.kwargs, trace.return_)
    msg = f"no protocol for {trace.attr!r}"
    raise InferError(msg)


def _or_object(node: _ir.Node | None) -> _ir.Node:
    """The node itself, or `object` for an unconstrained (`None`) one."""
    return _ir.Name(_OBJECT) if node is None else node


def _analyze(
    params: Sequence[_SpyObject],
    results: Iterable[object],
    traces: _Traces,
) -> tuple[list[_SpyObject], dict[int, int]]:
    appear: defaultdict[int, int] = defaultdict(int)
    for spy in params:
        appear[id(spy)] += 1
    for result in results:
        for spy in _return_spies(result):
            appear[id(spy)] += 1

    order: list[_SpyObject] = []
    seen: set[int] = set()
    stack = list(reversed(params))
    while stack:
        spy = stack.pop()
        if id(spy) in seen:
            continue
        seen.add(id(spy))
        order.append(spy)
        for op in traces[id(spy)]:
            for value in (*op.args, *op.kwargs.values()):
                if (arg := as_spy(value)) is not None:
                    appear[id(arg)] += 1
            if isinstance(op.return_, _SpyObject):
                appear[id(op.return_)] += 1
                stack.append(op.return_)
    return order, appear


def _ret_keys(
    order: Iterable[_SpyObject],
    traces: _Traces,
) -> tuple[dict[int, _RetKey], set[int]]:
    """The shape of the op that returned each spy, and the spies used as args."""
    keys: dict[int, _RetKey] = {}
    args: set[int] = set()
    for owner in order:
        for item in traces[id(owner)]:
            for value in (*item.args, *item.kwargs.values()):
                if (arg := as_spy(value)) is not None:
                    args.add(id(arg))
            if isinstance(item.return_, _SpyObject):
                key = id(owner), item.attr, len(item.args), tuple(sorted(item.kwargs))
                keys[id(item.return_)] = key
    return keys, args


def _spy_runs(items: Iterable[object], spy: _SpyObject) -> list[int]:
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
        case tuple() if not isinstance(value, _Gen | _Fn):
            tup = cast("tuple[object, ...]", value)
            if runs := _spy_runs(tup, spy):
                yield runs == [count]
            items: Iterable[object] = (item for item in tup if item is not spy)
        case _:
            items = _children(value)
    for item in items:
        yield from _packed_uses(item, spy, count)


def _all_packed(
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


def _return_spies(value: object) -> Generator[_SpyObject]:
    # an `_Fn`'s parameter spies are not returns; they are named like parameters
    for node in _walk(value):
        if (spy := as_spy(node)) is not None:
            yield spy


def _suffix(defaults: _Defaults, name: str) -> str:
    """The ` = <default>` suffix of a defaulted parameter, in stub style."""
    if name not in defaults:
        return ""
    value = defaults[name]
    simple = value is None or isinstance(value, int | float | complex | str | bytes)
    return f" = {value!r}" if simple else " = ..."


@final
class _Renderer:
    """Render an inferred `def` signature from the recorded spy traces."""

    def __init__(
        self,
        recon: _Recon,
        params: Mapping[str, Parameter],
        traces: _Traces,
    ) -> None:
        spies, _, results, count, self._fixed = recon
        self._spies = spies
        self._results = results
        self._traces = traces

        self._prefix = {n: _PARAM_PREFIX.get(p.kind, "") for n, p in params.items()}
        self._optional = {
            name
            for name, p in params.items()
            if p.default is not Parameter.empty or p.kind in _PARAM_PREFIX
        }
        self._varpos = next(
            (
                spies[name]
                for name, p in params.items()
                if p.kind is Parameter.VAR_POSITIONAL
            ),
            None,
        )

        # a returned function's parameter spies are named like regular parameters
        param_spies = [*spies.values(), *_fn_spies(results)]
        order, appear = _analyze(param_spies, results, traces)
        param_ids = {id(spy) for spy in param_spies}

        self._vars: _Vars = {}
        varpos = self._varpos
        if varpos is not None and _all_packed(varpos, results, traces, count):
            self._vars[id(varpos)] = _TYPEVAR_TUPLE
        self._result_spies: list[_SpyObject] = []
        self._name_results(order, param_ids)
        self._pool = [
            spy for spy in param_spies if appear[id(spy)] >= 2 or id(spy) in self._vars
        ]
        self._pool += [
            spy
            for spy in order
            if appear[id(spy)] >= 2
            and id(spy) not in param_ids
            and id(spy) not in self._vars
        ]
        unnamed = [spy for spy in self._pool if id(spy) not in self._vars]
        for i, spy in enumerate(unnamed):
            self._vars[id(spy)] = _TYPEVARS[i] if i < len(_TYPEVARS) else f"T{i}"

    def _name_results(self, order: Iterable[_SpyObject], param_ids: set[int]) -> None:
        ret_keys, arg_ids = _ret_keys(order, self._traces)
        shared: dict[_RetKey, str] = {}
        for result in self._results:
            for spy in _return_spies(result):
                if id(spy) in param_ids or id(spy) in self._vars:
                    continue
                # untraced results of same-shaped ops that are never passed as an
                # argument are interchangeable, so they share a typevar
                key = (
                    ret_keys.get(id(spy))
                    if not self._traces[id(spy)] and id(spy) not in arg_ids
                    else None
                )
                if key is not None and key in shared:
                    self._vars[id(spy)] = shared[key]
                    continue
                n = len(self._result_spies)
                var = "R" if not n else f"R{n + 1}"
                self._vars[id(spy)] = var
                self._result_spies.append(spy)
                if key is not None:
                    shared[key] = var

    def union(self, values: Iterable[object]) -> _ir.Node | None:
        literals: list[object] = []
        parts: list[_ir.Node] = []
        for value in values:
            if isinstance(value, int | str | bytes) and not isinstance(value, _Spy):
                literals.append(value)
            else:
                parts.append(self.return_type(value))

        nodes: list[_ir.Node] = []
        if literals:
            nodes.append(_ir.Lit(tuple(literals)))
        nodes.extend(dict.fromkeys(parts))
        return _ir.union(nodes)

    def returns(self, members: Iterable[_Op]) -> _ir.Node | None:
        named: list[str] = []
        items: list[_TraceItem] = []
        returned = False
        for m in members:
            if not isinstance(m.ret, _SpyObject):
                continue
            returned = True
            if (var := self._vars.get(id(m.ret))) is not None:
                named.append(var)
            else:
                items.extend(self._traces[id(m.ret)])
        parts: list[_ir.Node] = [_ir.Name(var) for var in dict.fromkeys(named)]
        if items and (node := self.traces(items)) is not None:
            parts.append(node)
        if (out := _ir.inter(parts)) is None and returned:
            # an unused result is unconstrained, but omitting it would leave the
            # last argument in the return slot, e.g. the `R` of `(T) -> R`
            out = _ir.Name(_OBJECT)
        return out

    def group(self, proto: _Proto, members: Sequence[_Op]) -> _ir.Node:
        if isinstance(proto, tuple):  # coercion protocols, which record no args
            return _ir.Union(tuple(map(_ir.Name, proto)))
        if proto == "CanArrayFunction":
            ret_str = _ir.render(_or_object(self.returns(members)))
            func = cast("_AnyFunc", members[0].args[0])
            return _ir.Name(_numpy.array_function_type(func, ret_str))
        args: list[_ir.Node | _ir.Arg] = [
            arg
            for i in range(len(members[0].args))
            if (arg := self.union(m.args[i] for m in members)) is not None
        ]
        args += [
            _ir.Arg(key, kw)
            for key in members[0].kwargs
            if (kw := self.union(m.kwargs[key] for m in members)) is not None
        ]
        ret = self.returns(members)
        if proto == "CanCall":
            return _ir.Fn(tuple(args), _or_object(ret))
        if ret is not None:
            args.append(ret)
        return _ir.App(proto, tuple(args))

    def traces(self, items: Iterable[_TraceItem]) -> _ir.Node | None:
        items = list(items)
        # a surviving absence marker proves the dunder was only probed optionally
        optional = {item.args[0] for item in items if item.attr == _Marker.ABSENT}
        groups: dict[tuple[_Proto, int, tuple[str, ...]], list[_Op]] = {}
        for item in items:
            if item.attr == _Marker.ABSENT or item.attr in optional:
                continue
            op = _resolve(item)
            key = op.proto, len(op.args), tuple(sorted(op.kwargs))
            groups.setdefault(key, []).append(op)
        return _ir.inter([self.group(key[0], group) for key, group in groups.items()])

    def spy(self, spy: _SpyObject) -> _ir.Node | None:
        return self.traces(self._traces[id(spy)])

    def _slot(self, spy: _SpyObject) -> _ir.Node:
        if (var := self._vars.get(id(spy))) is not None:
            return _ir.Name(var)
        return _or_object(self.spy(spy))

    def slot(self, spy: _SpyObject) -> str:
        return _ir.render(self._slot(spy))

    def typevar(
        self,
        spy: _SpyObject,
        defaulted: Mapping[int, _ir.Node],
        *,
        negate: bool = False,
    ) -> str:
        var = self._vars[id(spy)]
        node = self.spy(spy)
        default = defaulted.get(id(spy))
        if negate and default is not None:
            node = _ir.exclude(node, default)
        decl = f"{var}: {_ir.render(node)}" if node is not None else var
        if not negate and default is not None:
            decl += f" = {_ir.render(default)}"
        return decl

    def return_type(self, result: object) -> _ir.Node:
        match result:
            case _SpyObject():
                node: _ir.Node = _ir.Name(self._vars.get(id(_own_spy(result)), _OBJECT))
            case _SpyStr():
                node = _ir.Type(str)
            case _SpyBytes():
                node = _ir.Type(bytes)
            case _Gen():
                node = _ir.App(result.kind, (self._union_type(result.yielded),))
            case _Fn():
                node = self._function(result)
            case None:
                node = _ir.Name("None")
            case _:
                node = self._container(result)
        return node

    def _union_type(self, values: Iterable[object]) -> _ir.Node:
        """The deduplicated union of the types of `values`, or `Never` if empty."""
        parts = dict.fromkeys(map(self.return_type, values))
        return _ir.union(parts) or _ir.Name(_NEVER)

    def _function(self, fn: _Fn) -> _ir.Node:
        """The signature-syntax type of an explored function result."""
        params = tuple(
            _ir.Arg(name, self._fn_param(fn, name), _suffix(fn.defaults, name))
            for name in fn.names
        )
        return _ir.Fn(params, self._union_type(fn.results))

    def _fn_param(self, fn: _Fn, name: str) -> _ir.Node:
        if (spy := fn.spies.get(name)) is not None:
            return self._slot(spy)
        value = fn.fixed[name]
        if name in fn.defaults:
            # a pinned default renders as its value, like the outer parameters do
            return self.union((value,)) or _ir.Name(_NEVER)
        return _ir.Type(type(value))  # a method descriptor's pinned `self`

    def _class_of(self, cls: type[Any]) -> _ir.Node | None:
        """The type of `cls`'s instances, if it is expressible."""
        if (spy := _class_spy(cls)) is not None:
            return self.return_type(spy)
        if issubclass(cls, _SpyStr):
            return _ir.Type(str)
        if issubclass(cls, _SpyBytes):
            return _ir.Type(bytes)
        if cls is type(None):
            return _ir.Name("None")
        if getattr(sys.modules.get(cls.__module__), cls.__name__, None) is cls:
            return _ir.Type(cls)
        return None  # a local class has no nameable type, so it stays a bare `type`

    def _container(self, result: object) -> _ir.Node:
        cls = type(result)
        match result:
            case type() if (inner := self._class_of(result)) is not None:
                return _ir.App("type", (inner,))
            case dict():
                mapping = cast("Mapping[object, object]", result)
                key = self.union(mapping) or _ir.Name(_NEVER)
                val = self.union(mapping.values()) or _ir.Name(_NEVER)
                return _ir.App("dict", (key, val))
            case list() | set() | frozenset():
                inner = self.union(cast("Collection[object]", result))
                return _ir.App(cls.__name__, (inner or _ir.Name(_NEVER),))
            case tuple() if cls is tuple:
                if not result:
                    return _ir.Name("tuple[()]")
                return self._tuple(cast("tuple[object, ...]", result))
            case _:
                return _ir.Type(cls)

    def _tuple(self, items: tuple[object, ...]) -> _ir.Node:
        spy = self._varpos
        if spy is not None and any(item is spy for item in items):
            if self._vars.get(id(spy)) == _TYPEVAR_TUPLE:
                # every use is packed, so the placeholders splat into a single `*Ts`
                start = next(i for i, item in enumerate(items) if item is spy)
                parts = [
                    self.union((item,)) or _ir.Name(_NEVER)
                    for item in items
                    if item is not spy
                ]
                parts.insert(start, _ir.Name(_TYPEVAR_TUPLE))
                return _ir.App("tuple", tuple(parts))

            if all(item is spy for item in items):
                return _ir.App("tuple", (self.return_type(spy), _ir.Name("...")))

        elems = (self.union((item,)) or _ir.Name(_NEVER) for item in items)
        return _ir.App("tuple", tuple(elems))

    def return_types(self) -> str:
        return _ir.render(self._union_type(self._results))

    def render(
        self,
        selected: _Names,
        defaults: _Defaults | None = None,
        *,
        negate: bool = False,
    ) -> str:
        defaults = defaults or {}
        defaulted = {
            spy_id: node
            for name, value in defaults.items()
            if (spy := self._spies.get(name)) is not None
            if (spy_id := id(spy)) in self._vars
            if (node := self.union((value,))) is not None
        }
        ordered = self._pool + self._result_spies
        if not negate:
            # PEP 696 requires defaulted type parameters to come last
            ordered.sort(key=lambda spy: id(spy) in defaulted)
        typevars = [self.typevar(spy, defaulted, negate=negate) for spy in ordered]
        generics = f"[{', '.join(typevars)}]" if typevars else ""
        params = ", ".join(
            decl
            for name in selected
            if (decl := self._param(name, defaults, negate=negate)) is not None
        )
        return f"{generics}({params}) -> {self.return_types()}"

    def _param(self, name: str, defaults: _Defaults, *, negate: bool) -> str | None:
        if name in self._fixed and name not in self._optional:
            # a fixed parameter without a default is a method descriptor's `self`
            return f"{name}: {_ir.render(_ir.Type(type(self._fixed[name])))}"
        if (spy := self._spies.get(name)) is None:
            # an omitted parameter binds its default, so passing it behaves the same
            node = self.union((defaults[name],)) or _ir.Name(_NEVER)
            return f"{name}: {_ir.render(node)}{_suffix(defaults, name)}"
        slot = self.slot(spy)
        if negate and name in defaults:
            if id(spy) not in self._vars and (mark := self.union((defaults[name],))):
                slot = _ir.render(_ir.exclude(self.spy(spy), mark))
            return f"{self._prefix[name]}{name}: {slot}"
        if slot == _OBJECT and name in self._optional:
            return None
        return f"{self._prefix[name]}{name}: {slot}{_suffix(defaults, name)}"


def _reflect(
    params: Sequence[_SpyObject],
    results: Iterable[object],
    traces: _Traces,
) -> _Traces:
    """A copy of `traces` with each spy-spy binary op reflected onto its RHS."""
    order, _ = _analyze(params, results, traces)
    kept: _Traces = dict(traces)
    added: defaultdict[int, list[_TraceItem]] = defaultdict(list)
    for spy in order:
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


def _signatures(
    recon: _Recon,
    params: Mapping[str, Parameter],
    selected: _Names,
    defaults: _Defaults | None = None,
    *,
    negate: bool = False,
) -> list[str]:
    spies, traces, results, _, _ = recon
    roots = [*spies.values(), *_fn_spies(results)]
    reflected = _reflect(roots, results, traces)
    return [
        _Renderer(recon, params, t).render(selected, defaults, negate=negate)
        for t in (traces, reflected)
    ]
