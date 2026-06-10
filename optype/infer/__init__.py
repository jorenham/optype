"""Structurally infer the `optype` protocols required by a function."""

import re
import warnings
from collections import defaultdict
from collections.abc import (
    AsyncGenerator,
    Callable,
    Collection,
    Container,
    Coroutine,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)
from inspect import Parameter, isasyncgen, iscoroutine, isgenerator, signature
from itertools import islice
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


_PARAM_VAR = frozenset({Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD})

_TYPEVARS = "TUVWXYZ"
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


type _Vars = dict[int, str]
type _Traces = dict[int, list[_TraceItem]]
type _RetKey = tuple[int, str, int, tuple[str, ...]]
type _Proto = str | tuple[str, ...]  # a tuple is rendered as a union of protocols


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


def _select(params: Iterable[str | int], names: Sequence[str]) -> Sequence[str]:
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


@final
class _Renderer:
    """Render an inferred `def` signature from the recorded spy traces."""

    def __init__(
        self,
        selected: Sequence[str],
        spies: Mapping[str, _SpyObject],
        results: Sequence[object],
        optional: Container[str],
        traces: _Traces,
    ) -> None:
        self._selected = selected
        self._spies = spies
        self._results = results
        self._optional = optional
        self._traces = traces

        param_spies = list(spies.values())
        order, appear = _analyze(param_spies, results, traces)
        param_ids = {id(spy) for spy in param_spies}

        self._vars: _Vars = {}
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
        self._pool = [spy for spy in param_spies if appear[id(spy)] >= 2]
        self._pool += [
            spy
            for spy in order
            if appear[id(spy)] >= 2
            and id(spy) not in param_ids
            and id(spy) not in self._vars
        ]
        for i, spy in enumerate(self._pool):
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

    def typevar(self, spy: _SpyObject) -> str:
        var = self._vars[id(spy)]
        node = self.spy(spy)
        return f"{var}: {_ir.render(node)}" if node is not None else var

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
                items = cast("tuple[object, ...]", result)
                elems = (self.union((item,)) or _ir.Name(_NEVER) for item in items)
                return _ir.App("tuple", tuple(elems))
            case _:
                return _ir.Type(cls)

    def return_types(self) -> str:
        node = _ir.union(dict.fromkeys(map(self.return_type, self._results)))
        return _ir.render(node) if node is not None else _NEVER

    def render(self) -> str:
        typevars = [self.typevar(spy) for spy in self._pool + self._result_spies]
        generics = f"[{', '.join(typevars)}]" if typevars else ""
        params = ", ".join(
            f"{name}: {slot}"
            for name in self._selected
            if (slot := self.slot(self._spies[name])) != _OBJECT
            or name not in self._optional
        )
        return f"{generics}({params}) -> {self.return_types()}"


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
    kept: _Traces = {}
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
    func: Callable[..., T | Coroutine[Any, None, T]],
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


def infer(func: _AnyFunc, /, *params: str | int) -> str:
    """Infer the `optype` protocol(s) required of `func`'s parameters.

    Pass parameter names or positions to report only those parameters.

    >>> print(infer(lambda x: x + 1))
    [R](x: CanAdd[Literal[1], R]) -> R

    Raises:
        InferError: If `func` is not supported, such as a non-callable, variadic
            parameters, or an operation without a matching protocol.
    """
    if nin := _numpy.ufunc_nin(func):
        names = _numpy.ufunc_params(nin)
        return _numpy.infer_ufunc(func, names, _select(params, names))

    parameters = _parameters(func)
    if any(p.kind in _PARAM_VAR for p in parameters.values()):
        raise InferError("variadic parameters")

    names = list(parameters)
    selected = _select(params, names)
    optional = frozenset(
        name for name, p in parameters.items() if p.default is not Parameter.empty
    )

    spies = {name: _SpyObject() for name in names}
    args: list[_SpyObject] = []
    kwds: dict[str, _SpyObject] = {}
    for name, param in parameters.items():
        if param.kind is Parameter.KEYWORD_ONLY:
            kwds[name] = spies[name]
        else:
            args.append(spies[name])
    results: list[object] = [_next(r) for r in _explore(func, args, kwds)]
    param_spies = list(spies.values())
    traces = _snapshot(param_spies)
    reflected = _reflect(param_spies, results, traces)

    sig1 = _Renderer(selected, spies, results, optional, traces).render()
    sig2 = _Renderer(selected, spies, results, optional, reflected).render()
    return "\n".join(dict.fromkeys((sig1, sig2)))
