"""Render the recorded spy traces as a PEP 695 signature."""

from collections import defaultdict
from collections.abc import Collection, Generator, Iterable, Mapping, Sequence
from inspect import Parameter, _ParameterKind
from itertools import groupby
from typing import NamedTuple, cast, final

# `from . import` would import the package itself, which imports this module
import optype.infer._ir as _ir
import optype.infer._numpy as _numpy
from ._errors import InferError
from ._explore import _Gen, _Recon, _Traces
from ._spy import (
    _ABSENT,
    _AnyFunc,
    _Args,
    _Kwargs,
    _Spy,
    _SpyBytes,
    _SpyObject,
    _SpyStr,
    _TraceItem,
)
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
                if isinstance(value, _SpyObject):
                    appear[id(value)] += 1
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
                if isinstance(value, _SpyObject):
                    args.add(id(value))
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
        case _Gen():
            items = value.yielded
        case tuple():
            tup = cast("tuple[object, ...]", value)
            if runs := _spy_runs(tup, spy):
                yield runs == [count]
            items = (item for item in tup if item is not spy)
        case list() | set() | frozenset():
            items = cast("Collection[object]", value)
        case dict():
            mapping = cast("Mapping[object, object]", value)
            items = (*mapping, *mapping.values())
        case _:
            return
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
    match value:
        case _SpyObject():
            yield value
        case _Gen():
            for item in value.yielded:
                yield from _return_spies(item)
        case list() | set() | frozenset() | tuple():
            for item in cast("Collection[object]", value):
                yield from _return_spies(item)
        case dict():
            mapping = cast("Mapping[object, object]", value)
            for key, val in mapping.items():
                yield from _return_spies(key)
                yield from _return_spies(val)
        case _:
            pass


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

        param_spies = list(spies.values())
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
        for m in members:
            if not isinstance(m.ret, _SpyObject):
                continue
            if (var := self._vars.get(id(m.ret))) is not None:
                named.append(var)
            else:
                items.extend(self._traces[id(m.ret)])
        parts: list[_ir.Node] = [_ir.Name(var) for var in dict.fromkeys(named)]
        if items and (node := self.traces(items)) is not None:
            parts.append(node)
        return _ir.inter(parts)

    def group(self, proto: _Proto, members: Sequence[_Op]) -> _ir.Node:
        if isinstance(proto, tuple):  # coercion protocols, which record no args
            return _ir.Union(tuple(map(_ir.Name, proto)))
        if proto == "CanArrayFunction":
            ret = self.returns(members)
            ret_str = _ir.render(ret) if ret is not None else _OBJECT
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
        if (ret := self.returns(members)) is not None:
            args.append(ret)
        return _ir.App(proto, tuple(args))

    def traces(self, items: Iterable[_TraceItem]) -> _ir.Node | None:
        items = list(items)
        # a surviving absence marker proves the dunder was only probed optionally
        optional = {item.args[0] for item in items if item.attr == _ABSENT}
        groups: dict[tuple[_Proto, int, tuple[str, ...]], list[_Op]] = {}
        for item in items:
            if item.attr == _ABSENT or item.attr in optional:
                continue
            op = _resolve(item)
            key = op.proto, len(op.args), tuple(sorted(op.kwargs))
            groups.setdefault(key, []).append(op)
        return _ir.inter([self.group(key[0], group) for key, group in groups.items()])

    def spy(self, spy: _SpyObject) -> _ir.Node | None:
        return self.traces(self._traces[id(spy)])

    def slot(self, spy: _SpyObject) -> str:
        if (var := self._vars.get(id(spy))) is not None:
            return var
        node = self.spy(spy)
        return _ir.render(node) if node is not None else _OBJECT

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
                return _ir.Name(self._vars.get(id(result), _OBJECT))
            case _SpyStr():
                return _ir.Type(str)
            case _SpyBytes():
                return _ir.Type(bytes)
            case _Gen():
                yields = list(dict.fromkeys(map(self.return_type, result.yielded)))
                inner = _ir.union(yields) or _ir.Name(_NEVER)
                return _ir.App(result.kind, (inner,))
            case None:
                return _ir.Name("None")
            case _:
                return self._container(result)

    def _container(self, result: object) -> _ir.Node:
        cls = type(result)
        match result:
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
        node = _ir.union(dict.fromkeys(map(self.return_type, self._results)))
        return _ir.render(node) if node is not None else _NEVER

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
                added[id(rhs)].append(reflected)
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
    reflected = _reflect(list(spies.values()), results, traces)
    return [
        _Renderer(recon, params, t).render(selected, defaults, negate=negate)
        for t in (traces, reflected)
    ]
