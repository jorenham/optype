"""Render the recorded spy traces as a PEP 695 signature."""

import builtins
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
from ._values import _children, _Fn, _Gen, _Rec, _RecRef, _RecVar, _walk, fn_spies
from optype._core import _can, _has
from optype.inspect import get_protocol_members

_PARAM_PREFIX: dict[_ParameterKind, str] = {
    Parameter.VAR_POSITIONAL: "*",
    Parameter.VAR_KEYWORD: "**",
}

_TYPEVARS = "TUVWXYZ"
_TYPEVAR_TUPLE = "*Ts"
_NEVER = "Never"
_OBJECT = "object"
_TUPLE_LIMIT = 16

# the attribute polarity sigils of the fictional inline `Has['name', T]` form
_READ = "+"  # covariant: a read-only property suffices
_WRITE = "-"  # contravariant: the attribute only has to accept the value

_DUNDER_ATTR_READ = frozenset({"__getattr__", "__getattribute__"})
_DUNDER_ATTR_WRITE = frozenset({"__delattr__", "__setattr__"})
_DUNDER_ATTR = _DUNDER_ATTR_READ | _DUNDER_ATTR_WRITE
_DUNDER_CLASS_ATTR = frozenset({
    _Marker.CLASS_DELATTR,
    _Marker.CLASS_GETATTR,
    _Marker.CLASS_SETATTR,
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
type _Proto = str | tuple[str, ...]  # a tuple is rendered as a union of protocols

type _Names = Sequence[str]
type _Defaults = Mapping[str, object]


class _Op(NamedTuple):
    proto: _Proto
    args: _Args
    kwargs: _Kwargs
    ret: object
    attr: str | None = None  # the subject of a synthesized `Has[...]` form
    classvar: bool = False  # a class-level attribute, i.e. a `ClassVar` member


def _resolve(trace: _TraceItem) -> _Op:
    if trace.attr in _DUNDER_ATTR or trace.attr in _DUNDER_CLASS_ATTR:
        name = trace.args[0]
        if not isinstance(name, str) or isinstance(name, _Spy):
            msg = "no protocol for a dynamic attribute name"
            raise InferError(msg)

        # a class-level attribute mirrors a `ClassVar` protocol member, which no
        # shipped instance-member `Has*` protocol declares
        if trace.attr in _DUNDER_CLASS_ATTR:
            return _Op("Has", trace.args[1:], {}, trace.return_, name, classvar=True)

        # a read of an attribute with a shipped single-member `Has*` protocol
        if name in _DUNDER_HAS_MAP and trace.attr not in _DUNDER_ATTR_WRITE:
            return _Op(_DUNDER_HAS_MAP[name], (), {}, trace.return_)

        # everything else synthesizes the inline `Has['name', T]` form; a write
        # binds the assigned value's type, which a bounded `Has*` could reject
        return _Op("Has", trace.args[1:], {}, trace.return_, name)

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


def _sign_read(ret: _ir.Node) -> _ir.Node:
    """Mark a read covariant; a callable signs its return type instead: a method."""
    if isinstance(ret, _ir.Fn):
        return _ir.Fn(ret.params, _sign_read(ret.ret))
    return ret if ret == _ir.Name(_OBJECT) else _ir.Polarity(_READ, ret)


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


type _Producer = Mapping[int, tuple[_SpyObject, _TraceItem]]


def _shape(spy: _SpyObject, made_by: _Producer, keys: dict[int, object]) -> object:
    """A structural key over op-shape, not operands: `x[y]` and `x[z]` share one."""
    sid = id(spy)
    if sid in keys:
        return keys[sid]
    keys[sid] = sid  # a parameter or leaf is its own key
    if (made := made_by.get(sid)) is not None:
        owner, item = made
        # an attribute name replaces the fixed arity: `x.spam` is not `x.ham`
        attr = item.attr
        arity: object = (
            item.args[0]
            if attr in _DUNDER_ATTR or attr in _DUNDER_CLASS_ATTR
            else len(item.args)
        )
        owner_key = _shape(owner, made_by, keys)
        keys[sid] = "op", owner_key, attr, arity, tuple(sorted(item.kwargs))
    return keys[sid]


def _representatives(order: Sequence[_SpyObject], traces: _Traces) -> dict[int, int]:
    """Map each spy to a representative: the same operation on an owner of the same
    shape shares one, so the fresh placeholder each forked run allocates for a
    repeated subexpression collapses onto it instead of spawning a type parameter.
    """
    # a result is always a fresh spy, so a parameter never appears here and stays a leaf
    made_by: dict[int, tuple[_SpyObject, _TraceItem]] = {}
    for owner in order:
        for item in traces[id(owner)]:
            if isinstance(item.return_, _SpyObject):
                made_by.setdefault(id(item.return_), (owner, item))

    keys: dict[int, object] = {}
    rep: dict[object, int] = {}  # structural key -> the first spy that had it
    reps: dict[int, int] = {}
    for spy in order:
        reps[id(spy)] = rep.setdefault(_shape(spy, made_by, keys), id(spy))
    return reps


def _group_traces(
    named: Iterable[int],
    reps: dict[int, int],
    traces: _Traces,
) -> _Traces:
    """Each representative's bound: the traces of every named spy that shares it.

    Only named spies merge; an inline one still renders its constraints where used.
    """
    merged: _Traces = {}
    for spy_id in named:
        merged.setdefault(reps.get(spy_id, spy_id), []).extend(traces.get(spy_id, ()))
    return merged


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
        case tuple() if not isinstance(value, _Gen | _Fn | _Rec | _RecRef):
            tup = cast("tuple[object, ...]", value)
            if runs := _spy_runs(tup, spy):
                yield runs == [count]
            items = (item for item in tup if item is not spy)
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


def _is_sentinel(x: object, /) -> bool:
    # the getattr is a workaround for a pyrefly (1.0.0) bug
    return sys.version_info >= (3, 15) and isinstance(x, getattr(builtins, "sentinel"))  # noqa: B009


def _suffix(defaults: _Defaults, name: str) -> str:
    """The ` = <default>` suffix of a defaulted parameter, in stub style."""
    if name not in defaults:
        return ""
    value = defaults[name]
    simple = (
        value is None
        or _is_sentinel(value)  # a sentinel's repr is its declared name
        or isinstance(value, int | float | complex | str | bytes)
    )
    return f" = {value!r}" if simple else " = ..."


_CLEAN_EXIT = _ir.App("CanExit", (_ir.Name("None"),) * 3)
_CLEAN_AEXIT = _ir.App(
    "CanAExit",
    (*(_ir.Name("None"),) * 3, _ir.App("CanAwait", (_ir.Name(_OBJECT),))),
)
_LEN = _ir.App("CanLen", ())


def _merge_combined(parts: list[_ir.Node]) -> list[_ir.Node]:
    """Merge an intersection pair into the combined protocol that optype ships.

    A `with` statement requires `__enter__` and a clean-exit `__exit__` together,
    which combine as `CanWith`; the unused `__exit__` result is unconstrained.
    `CanAsyncWith` is alike, but its declared parameters are the awaited results, so
    the `CanAwait` wrappers unwrap. `CanGetitem & CanLen` combines as `CanSequence`.
    """
    for app in parts:
        match app:
            case _ir.App("CanEnter", (entered,)):
                partner = _CLEAN_EXIT
                merged = _ir.App("CanWith", (entered, _ir.Name(_OBJECT)))
            case _ir.App("CanAEnter", (_ir.App("CanAwait", (entered,)),)):
                partner = _CLEAN_AEXIT
                merged = _ir.App("CanAsyncWith", (entered, _ir.Name(_OBJECT)))
            case _ir.App("CanGetitem", (key, value)):
                partner = _LEN
                merged = _ir.App("CanSequence", (key, value))
            case _:
                continue
        if partner in parts:
            parts = [merged if p is app else p for p in parts if p != partner]
    return parts


def _result_var(index: int) -> str:
    """The `index`-th return typevar name: `R`, `R2`, `R3`, ..."""
    return "R" if not index else f"R{index + 1}"


@final
class _Renderer:
    """Render an inferred `def` signature from the recorded spy traces."""

    _spies: Mapping[str, _SpyObject]
    _fixed: Mapping[str, object]
    _results: list[object]
    _traces: _Traces

    _prefix: dict[str, str]
    _nameless: set[str]
    _optional: set[str]
    _varpos: _SpyObject | None

    _reps: dict[int, int]
    _vars: _Vars
    _named: dict[int, str]

    _result_spies: list[_SpyObject]

    _rec_vars: dict[_RecVar, str]
    _rec_body: dict[_RecVar, object]

    _pool: list[_SpyObject]
    _group_traces: _Traces

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
        self._nameless = {
            n for n, p in params.items() if p.kind is Parameter.POSITIONAL_ONLY
        }
        self._optional = {
            name
            for name, p in params.items()
            if p.default is not Parameter.empty or p.kind in _PARAM_PREFIX
        }
        self._varpos = next(  # ty:ignore[invalid-assignment]
            (
                spies[name]
                for name, p in params.items()
                if p.kind is Parameter.VAR_POSITIONAL
            ),
            None,
        )

        # a returned function's parameter spies are named like regular parameters
        param_spies = [*spies.values(), *fn_spies(results)]
        order, appear = _analyze(param_spies, results, traces)
        param_ids = {id(spy) for spy in param_spies}
        self._reps = reps = _representatives(order, traces)

        self._vars = {}
        self._named = {}  # representative id -> name
        varpos = self._varpos
        if varpos is not None and _all_packed(varpos, results, traces, count):
            self._vars[id(varpos)] = _TYPEVAR_TUPLE

        self._result_spies = []
        self._name_results(param_ids, reps)

        self._rec_body = {
            node.var: node.body
            for result in results
            for node in _walk(result)
            if isinstance(node, _Rec)
        }
        base = len(self._result_spies)
        self._rec_vars = {
            var: _result_var(base + i) for i, var in enumerate(self._rec_body)
        }

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

        # one type parameter per distinct expression: duplicates reuse its name
        pool: list[_SpyObject] = []
        n = 0
        for spy in self._pool:
            rep = reps.get(id(spy), id(spy))
            if (var := self._named.get(rep)) is None:
                var = self._vars.get(id(spy))  # a `*Ts` variadic keeps its name
                if var is None:
                    var = _TYPEVARS[n] if n < len(_TYPEVARS) else f"T{n}"
                    n += 1
                self._named[rep] = var
                pool.append(spy)
            self._vars[id(spy)] = var
        self._pool = pool
        self._group_traces = _group_traces(self._vars, reps, traces)

    def _name_results(self, param_ids: set[int], reps: dict[int, int]) -> None:
        for result in self._results:
            for spy in _return_spies(result):
                sid = id(spy)
                if sid in param_ids or sid in self._vars:
                    continue
                # results of one op-shape share a type parameter, even traced or reused
                rep = reps.get(sid, sid)
                if (var := self._named.get(rep)) is not None:
                    self._vars[sid] = var
                    continue
                var = _result_var(len(self._result_spies))
                self._vars[sid] = var
                self._result_spies.append(spy)
                self._named[rep] = var

    def union(
        self,
        values: Iterable[object],
        *,
        tuples: bool = False,
    ) -> _ir.Node | None:
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
        return _ir.union(nodes, tuples=tuples)

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

        pos = [
            arg
            for i in range(len(members[0].args))
            if (arg := self.union(m.args[i] for m in members)) is not None
        ]
        args = (
            *pos,
            *(
                _ir.Arg(key, kw)
                for key in members[0].kwargs
                if (kw := self.union(m.kwargs[key] for m in members)) is not None
            ),
        )
        ret = self.returns(members)

        if proto == "CanCall":
            return _ir.Fn(args, _or_object(ret))

        if (attr := members[0].attr) is not None:
            # a read is covariant (a property suffices) and a write contravariant
            # (the attribute only has to accept the value); existence renders bare
            signed: tuple[_ir.Node, ...] = tuple(_ir.Polarity(_WRITE, a) for a in pos)
            if ret is not None and ret != _ir.Name(_OBJECT):
                signed = *signed, _sign_read(ret)
            if members[0].classvar:
                signed = (_ir.App("ClassVar", signed),)
            return _ir.App(proto, (_ir.Name(repr(attr)), *signed))

        if ret is not None:
            args = *args, ret

        return _ir.App(proto, args)

    def traces(self, items: Iterable[_TraceItem]) -> _ir.Node | None:
        items = list(items)
        # a surviving absence marker proves the dunder was only probed optionally
        optional = {item.args[0] for item in items if item.attr == _Marker.ABSENT}
        groups: dict[
            tuple[_Proto, str | None, bool, int, tuple[str, ...]],
            list[_Op],
        ] = {}
        for item in items:
            if item.attr == _Marker.ABSENT or item.attr in optional:
                continue
            op = _resolve(item)
            key = op.proto, op.attr, op.classvar, len(op.args), tuple(sorted(op.kwargs))
            groups.setdefault(key, []).append(op)
        parts = [self.group(key[0], group) for key, group in groups.items()]
        return _ir.inter(_merge_combined(parts))

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
        rep = self._reps.get(id(spy), id(spy))
        node = self.traces(self._group_traces.get(rep, ()))
        default = defaulted.get(id(spy))
        if negate and default is not None:
            node = _ir.exclude(node, default)
        decl = f"{var}: {_ir.render(node)}" if node is not None else var
        if not negate and default is not None:
            decl += f" = {_ir.render(default)}"
        return decl

    def return_type(self, result: object) -> _ir.Node:
        node: _ir.Node
        match result:
            case _RecRef() | _Rec():
                node = _ir.Name(self._rec_vars[result.var])
            case _SpyObject() if (spy := as_spy(result)) is not None:
                node = _ir.Name(self._vars.get(id(spy), _OBJECT))
            case _SpyStr():
                node = _ir.Type(str)
            case _SpyBytes():
                node = _ir.Type(bytes)
            case _Gen():
                node = _ir.App(result.kind, (self._union_type(result.yielded),))
            case _Fn():
                node = self._function(result)
            case _ if result is None or _is_sentinel(result):
                node = _ir.Name(repr(result))
            case _:
                node = self._container(result)
        return node

    def _union_type(self, values: Iterable[object]) -> _ir.Node:
        """The deduplicated union of the types of `values`, or `Never` if empty."""
        parts = dict.fromkeys(map(self.return_type, values))
        return _ir.union(parts, tuples=True) or _ir.Name(_NEVER)

    def _function(self, fn: _Fn) -> _ir.Node:
        """The signature-syntax type of an explored function result."""
        params = tuple(
            _ir.Arg(
                None if p.kind is Parameter.POSITIONAL_ONLY else name,
                self._fn_param(fn, name),
                _suffix(fn.defaults, name),
            )
            for name, p in fn.params.items()
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
            case Mapping():
                mapping = cast("Mapping[object, object]", result)
                key = self.union(mapping, tuples=True) or _ir.Name(_NEVER)
                val = self.union(mapping.values(), tuples=True) or _ir.Name(_NEVER)
                return _ir.App(cls.__name__, (key, val))
            case list() | set() | frozenset():
                inner = self.union(cast("Collection[object]", result), tuples=True)
                return _ir.App(cls.__name__, (inner or _ir.Name(_NEVER),))
            case tuple() if cls is tuple:
                return (
                    self._tuple(cast("tuple[object, ...]", result))
                    if result
                    else _ir.Name("tuple[()]")
                )
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

        if len(items) > _TUPLE_LIMIT:  # e.g. `random.getstate`
            return _ir.App("tuple", (self._union_type(items), _ir.Name("...")))

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

        # a recursive typevar carries no default, so it precedes the defaulted tail
        deferred = 0 if negate else sum(id(spy) in defaulted for spy in ordered)
        cut = len(ordered) - deferred
        typevars[cut:cut] = [
            f"{name}: {_ir.render(self.return_type(self._rec_body[var]))}"
            for var, name in self._rec_vars.items()
        ]

        generics = f"[{', '.join(typevars)}]" if typevars else ""
        params = ", ".join(
            decl
            for name in selected
            if (decl := self._param(name, defaults, negate=negate)) is not None
        )
        return f"{generics}({params}) -> {self.return_types()}"

    def _param(self, name: str, defaults: _Defaults, *, negate: bool) -> str | None:
        # a positional-only parameter cannot be passed by keyword, so no name shows
        label = "" if name in self._nameless else f"{self._prefix[name]}{name}: "
        if name in self._fixed and name not in self._optional:
            # a fixed parameter without a default is a method descriptor's `self`
            return f"{label}{_ir.render(_ir.Type(type(self._fixed[name])))}"
        if (spy := self._spies.get(name)) is None:
            # an omitted parameter binds its default, so passing it behaves the same
            node = self.union((defaults[name],)) or _ir.Name(_NEVER)
            return f"{label}{_ir.render(node)}{_suffix(defaults, name)}"
        slot = self.slot(spy)
        if negate and name in defaults:
            if id(spy) not in self._vars and (mark := self.union((defaults[name],))):
                slot = _ir.render(_ir.exclude(self.spy(spy), mark))
            return f"{label}{slot}"
        if slot == _OBJECT and name in self._optional:
            return None
        return f"{label}{slot}{_suffix(defaults, name)}"


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


def signatures(
    recon: _Recon,
    params: Mapping[str, Parameter],
    selected: _Names,
    defaults: _Defaults | None = None,
    *,
    negate: bool = False,
) -> list[str]:
    spies, traces, results, _, _ = recon
    roots = [*spies.values(), *fn_spies(results)]
    reflected = _reflect(roots, results, traces)
    return [
        _Renderer(recon, params, t).render(selected, defaults, negate=negate)
        for t in (traces, reflected)
    ]
