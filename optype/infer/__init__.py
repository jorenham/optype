"""Structurally infer the `optype` protocols required by a function."""

import re
import warnings
from collections import defaultdict
from collections.abc import (
    AsyncGenerator,
    Callable,
    Collection,
    Coroutine,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)
from inspect import (
    Parameter,
    _ParameterKind,
    isasyncgen,
    iscoroutine,
    isgenerator,
    signature,
)
from itertools import groupby, islice
from typing import Any, NamedTuple, cast, final, overload

from . import _ir, _numpy
from ._spy import (
    _AnyFunc,
    _Args,
    _Fork,
    _fork,
    _Kwargs,
    _Spy,
    _SpyBytes,
    _SpyObject,
    _SpyStr,
    _TraceItem,
)
from optype._core import _can, _has
from optype.inspect import get_protocol_members

__all__ = ("InferError", "InferWarning", "infer")


class InferError(NotImplementedError):
    """Raised when `infer` does not support the given function."""


class InferWarning(RuntimeWarning):
    """Emitted when `infer` could not explore the function exhaustively."""


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

_RE_DOC_SIGNATURE = re.compile(r"\b(\w+)\(([^)]*)\)")
_RE_DOC_PARAM = re.compile(r"(?:^|,)\s*\**([a-zA-Z_]\w*)")

_FORK_LIMIT = 64
_RUN_LIMIT = 256
_YIELD_LIMIT = 64
# the `*args` placeholder counts to try: exact arities first, then doubling so that
# large indices stay within reach
_VARIADIC_COUNTS = (2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024)
_KWARGS_LIMIT = 8  # max injected `**kwargs` keys


type _Vars = dict[int, str]
type _Traces = dict[int, list[_TraceItem]]
type _RetKey = tuple[int, str, int, tuple[str, ...]]
type _Proto = str | tuple[str, ...]  # a tuple is rendered as a union of protocols

type _Names = Sequence[str]
type _Defaults = Mapping[str, object]

# the spies, their traces, the results, and the `*args` placeholder count
type _Recon = tuple[Mapping[str, _SpyObject], _Traces, list[object], int]


class _Op(NamedTuple):
    proto: _Proto
    args: _Args
    kwargs: _Kwargs
    ret: object


class _Gen(NamedTuple):
    yielded: list[object]
    is_async: bool

    @property
    def kind(self) -> str:
        return "AsyncGenerator" if self.is_async else "Generator"


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


def _snapshot(params: Iterable[_SpyObject]) -> _Traces:
    """Capture the traces of every spy reachable from `params`."""
    traces: _Traces = {}
    stack = list(params)
    while stack:
        spy = stack.pop()
        if id(spy) in traces:
            continue
        items = traces[id(spy)] = list(spy.__optype_trace__)
        stack.extend(
            ret for item in items if isinstance(ret := item.return_, _SpyObject)
        )
    return traces


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


def _select(params: Iterable[str | int], names: _Names) -> _Names:
    selected: list[str] = []
    for p in params:
        if isinstance(p, int):
            if not -len(names) <= p < len(names):
                msg = f"no parameter at position {p}"
                raise ValueError(msg)
            selected.append(names[p])
        elif p in names:
            selected.append(p)
        else:
            msg = f"unknown parameter {p!r}"
            raise ValueError(msg)
    return selected or names


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
        spies: Mapping[str, _SpyObject],
        results: Sequence[object],
        params: Mapping[str, Parameter],
        count: int,
        traces: _Traces,
    ) -> None:
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
        ret_keys, arg_ids = _ret_keys(order, traces)
        shared: dict[_RetKey, str] = {}
        for result in results:
            for spy in _return_spies(result):
                if id(spy) in param_ids or id(spy) in self._vars:
                    continue
                # untraced results of same-shaped ops that are never passed as an
                # argument are interchangeable, so they share a typevar
                key = (
                    ret_keys.get(id(spy))
                    if not traces[id(spy)] and id(spy) not in arg_ids
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
        groups: dict[tuple[_Proto, int, tuple[str, ...]], list[_Op]] = {}
        for item in items:
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


def _doc_params(func: _AnyFunc) -> list[str] | None:
    name = getattr(func, "__name__", "")
    if not name:
        return None
    for match in _RE_DOC_SIGNATURE.finditer(func.__doc__ or ""):
        if match[1] == name:
            params = match[2].replace("[", "").replace("]", "")
            return _RE_DOC_PARAM.findall(params) or None
    return None


def _parameters(func: _AnyFunc) -> Mapping[str, Parameter]:
    try:
        return signature(func).parameters
    except TypeError as exc:  # not callable
        raise InferError(str(exc)) from exc
    except ValueError as exc:  # no signature
        if (names := _doc_params(func)) is None:
            raise InferError(str(exc)) from exc
        return {n: Parameter(n, Parameter.POSITIONAL_OR_KEYWORD) for n in names}


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


def _await[R](coro: Coroutine[Any, Any, R]) -> R:
    # a spy's awaitables resolve synchronously, so the coroutine runs straight through
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    coro.close()
    raise InferError("await on a non-spy awaitable")


def _yield_key(value: object) -> tuple[str, *tuple[str, ...]]:
    # a value's "shape": two yields with the same key are treated as the same type
    if isinstance(value, _SpyObject):
        return ("spy", *(op.attr for op in value.__optype_trace__))
    return ("val", type(value).__name__)


def _yields[T](values: Iterable[T]) -> list[T]:
    seen: set[tuple[str, ...]] = set()
    out: list[T] = []
    for value in islice(values, _YIELD_LIMIT):
        if (key := _yield_key(value)) in seen:
            break
        seen.add(key)
        out.append(value)
    return out


def _sync[T](agen: AsyncGenerator[T, Any]) -> Generator[T]:
    # iterate an async generator synchronously by resolving each step's awaitable
    for _ in range(_YIELD_LIMIT):
        try:
            yield _await(anext(agen))
        except StopAsyncIteration:
            return


@overload
def _next(result: Generator[object] | AsyncGenerator[object]) -> _Gen: ...
@overload
def _next[T](result: T) -> T: ...
def _next(result: object) -> object:
    if isgenerator(result):
        return _Gen(_yields(result), is_async=False)
    if isasyncgen(result):
        return _Gen(_yields(_sync(result)), is_async=True)
    return result


def _explore[T](
    func: Callable[..., T] | Callable[..., Coroutine[Any, None, T]],
    args: Sequence[_SpyObject],
    kwds: Mapping[str, _SpyObject],
) -> list[T]:
    results: list[T] = []
    stack: list[list[bool]] = [[]]
    dropped = False
    for _ in range(_RUN_LIMIT):  # caps the exponential blowup of independent forks
        if not stack:
            break
        plan = stack.pop()
        token = _fork.set(iter(plan))
        try:
            result = func(*args, **kwds)
            results.append(_await(result) if iscoroutine(result) else cast("T", result))
        except _Fork:
            if len(plan) < _FORK_LIMIT:
                stack.extend(([*plan, False], [*plan, True]))
            else:
                dropped = True
        finally:
            _fork.reset(token)
    if not results:
        raise InferError("the function never ran to completion")
    if dropped or stack:
        warnings.warn("not every branch was explored", InferWarning, stacklevel=3)
    return results


def _placeholders(
    params: Mapping[str, Parameter],
    count: int,
    keys: Sequence[str],
    omit: Collection[str] = (),
) -> tuple[dict[str, _SpyObject], list[_SpyObject], dict[str, _SpyObject]]:
    # one spy per non-omitted parameter, distributed over the call's args and kwds
    spies = {name: _SpyObject() for name in params if name not in omit}
    args: list[_SpyObject] = []
    kwds: dict[str, _SpyObject] = {}
    gap = False  # a positional parameter after an omitted one must pass by keyword
    for name, param in params.items():
        if name in omit:
            gap = gap or param.kind is not Parameter.KEYWORD_ONLY
            continue
        match param.kind:
            case Parameter.VAR_POSITIONAL:
                args += [spies[name]] * count
            case Parameter.VAR_KEYWORD:
                kwds |= dict.fromkeys(map(_SpyStr, keys or ("",)), spies[name])
            case Parameter.KEYWORD_ONLY:
                kwds[name] = spies[name]
            case Parameter.POSITIONAL_ONLY if gap:
                msg = f"cannot pass {name!r} by keyword"
                raise InferError(msg)
            case _ if gap:
                kwds[name] = spies[name]
            case _:
                args.append(spies[name])
    return spies, args, kwds


def _explore_spies(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    omit: Collection[str] = (),
) -> _Recon:
    # rerun with fresh spies whenever the variadic placeholders come up short
    kinds = {p.kind for p in params.values()}
    counts = iter(_VARIADIC_COUNTS)
    count = next(counts)
    keys: list[str] = []
    while True:
        spies, args, kwds = _placeholders(params, count, keys, omit)
        try:
            results: list[object] = [_next(r) for r in _explore(func, args, kwds)]
        except KeyError as exc:
            key = exc.args[0] if exc.args else None
            if (
                Parameter.VAR_KEYWORD not in kinds
                or not isinstance(key, str)
                or key in keys
                or key in params
            ):
                raise
            if len(keys) >= _KWARGS_LIMIT:
                msg = f"ran out of `**kwargs` placeholder keys ({exc})"
                raise InferError(msg) from exc
            keys.append(key)
        except (IndexError, TypeError, ValueError) as exc:
            if Parameter.VAR_POSITIONAL not in kinds:
                raise
            if (count := next(counts, 0)) == 0:
                msg = f"ran out of `*args` placeholders ({exc})"
                raise InferError(msg) from exc
        else:
            return spies, _snapshot(spies.values()), results, count


def _bind(value: object, binding: Mapping[int, object]) -> object:
    """A deep copy of `value` with every bound spy replaced by its binding."""
    cls = type(value)
    match value:
        case _Gen():
            yielded = [_bind(item, binding) for item in value.yielded]
            return _Gen(yielded, value.is_async)
        case tuple() if cls is tuple:
            tup = cast("tuple[object, ...]", value)
            return tuple(_bind(item, binding) for item in tup)
        case list():
            return [_bind(item, binding) for item in cast("list[object]", value)]
        case set() | frozenset():
            items = {_bind(item, binding) for item in cast("Collection[object]", value)}
            return frozenset(items) if isinstance(value, frozenset) else items
        case dict():
            mapping = cast("Mapping[object, object]", value)
            return {_bind(k, binding): _bind(v, binding) for k, v in mapping.items()}
        case _:
            return binding.get(id(value), value)


def _bind_recon(recon: _Recon, defaults: _Defaults) -> _Recon:
    """The recon as it would look with every defaulted parameter omitted."""
    spies, traces, results, count = recon
    binding = {id(spies[name]): value for name, value in defaults.items()}
    bound = {
        spy_id: [
            _TraceItem(
                item.attr,
                tuple(_bind(arg, binding) for arg in item.args),
                {key: _bind(val, binding) for key, val in item.kwargs.items()},
                item.return_,
            )
            for item in items
        ]
        for spy_id, items in traces.items()
    }
    kept = {name: spy for name, spy in spies.items() if name not in defaults}
    return kept, bound, [_bind(result, binding) for result in results], count


def _signatures(
    recon: _Recon,
    params: Mapping[str, Parameter],
    selected: _Names,
    defaults: _Defaults | None = None,
    *,
    negate: bool = False,
) -> list[str]:
    spies, traces, results, count = recon
    reflected = _reflect(list(spies.values()), results, traces)
    return [
        _Renderer(spies, results, params, count, t).render(
            selected,
            defaults,
            negate=negate,
        )
        for t in (traces, reflected)
    ]


def _defaults(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    selected: _Names,
    recon: _Recon,
) -> tuple[_Defaults, bool, list[str]]:
    """The parameter defaults if expressible as typevar defaults, else overloads.

    Omitting the defaulted parameters must behave like substituting their values
    into the generic signature; the function is rerun without them to check. On a
    mismatch the omitted calls are reported as separate overload lines, and a
    single defaulted parameter's type is excluded from the generic signature.
    """
    defaults = {
        name: p.default
        for name, p in params.items()
        if p.default is not Parameter.empty
    }
    kinds = {p.kind for p in params.values()}
    if not defaults or (
        # `*args` placeholders would positionally fill an omitted default
        Parameter.VAR_POSITIONAL in kinds
        and any(params[n].kind is not Parameter.KEYWORD_ONLY for n in defaults)
    ):
        return {}, False, []

    required = {name: p for name, p in params.items() if name not in defaults}
    names = list(required)
    try:
        omitted = _explore_spies(func, params, omit=defaults)
        # the comparison must see every required parameter, regardless of selection
        observed = _signatures(omitted, required, names)
    except Exception:  # noqa: BLE001  # the omitted call may legitimately fail
        return {}, False, []
    if _signatures(_bind_recon(recon, defaults), required, names) == observed:
        return defaults, False, []

    overloads = _signatures(omitted, required, selected, defaults)
    if len(defaults) == 1:
        return defaults, True, overloads
    for name, value in defaults.items():
        rest = {n: p for n, p in params.items() if n != name}
        try:
            variant = _explore_spies(func, params, omit={name})
        except Exception:  # noqa: BLE001, S112
            continue
        overloads += _signatures(variant, rest, selected, {name: value})
    return {}, False, overloads


def infer(func: _AnyFunc, /, *params: str | int) -> str:
    """Infer the `optype` protocol(s) required of `func`'s parameters.

    Pass parameter names or positions to report only those parameters.

    >>> print(infer(lambda x: x + 1))
    [R](x: CanAdd[Literal[1], R]) -> R

    Raises:
        InferError: If `func` is not supported, such as a non-callable, or an
            operation without a matching protocol.
    """  # noqa: DOC502
    if nin := _numpy.ufunc_nin(func):
        names = _numpy.ufunc_params(nin)
        return _numpy.infer_ufunc(func, names, _select(params, names))

    parameters = _parameters(func)
    selected = _select(params, list(parameters))
    recon = _explore_spies(func, parameters)
    defaults, negate, overloads = _defaults(func, parameters, selected, recon)
    lines = _signatures(recon, parameters, selected, defaults, negate=negate)
    return "\n".join(dict.fromkeys((*overloads, *lines)))
