"""Structurally infer the ``optype`` protocols required by a function."""

import re
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
from inspect import Parameter, isasyncgen, iscoroutine, isgenerator, signature
from itertools import islice
from typing import Any, NamedTuple, cast, final, overload

from ._spy import _Fork, _fork, _Spy, _SpyBytes, _SpyObject, _SpyStr, _TraceItem
from optype._core import _can, _has
from optype.inspect import get_protocol_members

__all__ = ("infer",)

_PARAM_VAR = frozenset({Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD})

_TYPEVARS = "TUVWXYZ"

_DUNDER_ATTR = frozenset({
    "__delattr__",
    "__getattr__",
    "__getattribute__",
    "__setattr__",
})
_DUNDER_CAN_MAP = {
    dunder: name
    for name in _can.__all__
    if not name.endswith(("Self", "Same"))
    if (dunder := "__" + name.removeprefix("Can").lower() + "__") not in _DUNDER_ATTR
} | {
    "__array_ufunc__": "CanArrayUFunc",
    "__array_function__": "CanArrayFunction",
}
_DUNDER_CAN_R = frozenset(
    dunder
    for dunder, proto in _DUNDER_CAN_MAP.items()
    if "CanR" + proto.removeprefix("Can") in _DUNDER_CAN_MAP.values()
)
_DUNDER_HAS_MAP = {
    next(iter(members)): name
    for name in _has.__all__
    if len(members := get_protocol_members(getattr(_has, name))) == 1
}

_COERCION_FALLBACK = {
    "__float__": ("__index__",),
    "__int__": ("__index__",),
    "__complex__": ("__float__", "__index__"),
}
_COERCION_UNION = {
    dunder: " | ".join(map(_DUNDER_CAN_MAP.__getitem__, (dunder, *fallback)))
    for dunder, fallback in _COERCION_FALLBACK.items()
}

_RE_DOC_SIGNATURE = re.compile(r"\b(\w+)\(([^)]*)\)")
_RE_DOC_PARAM = re.compile(r"(?:^|,)\s*\**([a-zA-Z_]\w*)")

_FORK_LIMIT = 64
_YIELD_LIMIT = 64


type _AnyFunc = Callable[..., Any]
type _Vars = dict[int, str]


class _Op(NamedTuple):
    proto: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    ret: Any


class _Gen(NamedTuple):
    yielded: list[object]
    is_async: bool

    @property
    def kind(self) -> str:
        return "AsyncGenerator" if self.is_async else "Generator"


def _resolve(trace: _TraceItem) -> _Op:
    if trace.attr in _DUNDER_ATTR:
        name = trace.args[0]
        if name not in _DUNDER_HAS_MAP:
            raise NotImplementedError(name)
        return _Op(_DUNDER_HAS_MAP[name], (), {}, trace.return_)
    if trace.attr in _COERCION_UNION:
        return _Op(_COERCION_UNION[trace.attr], (), {}, trace.return_)
    if trace.attr in _DUNDER_CAN_MAP:
        proto = _DUNDER_CAN_MAP[trace.attr]
        return _Op(proto, trace.args, trace.kwargs, trace.return_)
    raise NotImplementedError(trace.attr)


def _analyze(
    params: Sequence[_SpyObject],
    results: Iterable[object],
) -> tuple[list[_SpyObject], dict[int, int]]:
    appear: defaultdict[int, int] = defaultdict(int)
    for spy in params:
        appear[id(spy)] += 1
    for result in results:
        if isinstance(result, _SpyObject):
            appear[id(result)] += 1

    order: list[_SpyObject] = []
    seen: set[int] = set()
    stack = list(reversed(params))
    while stack:
        spy = stack.pop()
        if id(spy) in seen:
            continue
        seen.add(id(spy))
        order.append(spy)
        for op in spy.__optype_trace__:
            for value in (*op.args, *op.kwargs.values()):
                if isinstance(value, _SpyObject):
                    appear[id(value)] += 1
            if isinstance(op.return_, _SpyObject):
                appear[id(op.return_)] += 1
                stack.append(op.return_)
    return order, appear


def _select(params: tuple[str | int, ...], names: list[str]) -> list[str]:
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
        case list() | set() | frozenset():
            for item in cast("Collection[object]", value):
                yield from _return_spies(item)
        case dict():
            mapping = cast("Mapping[object, object]", value)
            for key, val in mapping.items():
                yield from _return_spies(key)
                yield from _return_spies(val)
        case _:
            pass


def _required_args(func: _AnyFunc) -> int | None:
    try:
        params = signature(func).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind in _PARAM_VAR for p in params):
        return None
    return sum(
        p.kind is not Parameter.KEYWORD_ONLY and p.default is Parameter.empty
        for p in params
    )


@final
class _Renderer:
    """Render an inferred ``def`` signature from the recorded spy traces."""

    def __init__(
        self,
        names: list[str],
        selected: list[str],
        spies: dict[str, _SpyObject],
        results: list[object],
        optional: frozenset[str],
    ) -> None:
        self._selected = selected
        self._spies = spies
        self._results = results
        self._optional = optional

        param_spies = [spies[name] for name in names]
        order, appear = _analyze(param_spies, results)
        param_ids = {id(spy) for spy in param_spies}

        self._vars: _Vars = {}
        self._result_spies: list[_SpyObject] = []
        for result in results:
            for spy in _return_spies(result):
                if id(spy) in param_ids or id(spy) in self._vars:
                    continue
                n = len(self._result_spies)
                self._vars[id(spy)] = "R" if not n else f"R{n + 1}"
                self._result_spies.append(spy)
        self._pool = [spies[name] for name in names if appear[id(spies[name])] >= 2]
        self._pool += [
            spy
            for spy in order
            if appear[id(spy)] >= 2
            and id(spy) not in param_ids
            and id(spy) not in self._vars
        ]
        for i, spy in enumerate(self._pool):
            self._vars[id(spy)] = _TYPEVARS[i] if i < len(_TYPEVARS) else f"T{i}"

    def union(self, values: Iterable[Any]) -> str | None:
        literals: list[Any] = []
        names: list[str] = []
        for value in values:
            if isinstance(value, int | str | bytes) and not isinstance(value, _Spy):
                literals.append(value)
            else:
                names.append(self.return_type(value))
        if "float" in names or "complex" in names:
            literals = [value for value in literals if not isinstance(value, int)]
        if "complex" in names:
            names = [name for name in names if name != "float"]

        parts: list[str] = []
        if literals:
            parts.append(f"Literal[{', '.join(map(repr, dict.fromkeys(literals)))}]")
        parts.extend(dict.fromkeys(names))
        return " | ".join(parts) or None

    def returns(self, members: Iterable[_Op]) -> str | None:
        named: list[str] = []
        traces: list[_TraceItem] = []
        for m in members:
            if not isinstance(m.ret, _SpyObject):
                continue
            if (var := self._vars.get(id(m.ret))) is not None:
                named.append(var)
            else:
                traces.extend(m.ret.__optype_trace__)
        parts = list(dict.fromkeys(named))
        if traces and (formatted := self.traces(traces)):
            parts.append(formatted)
        return " & ".join(parts) or None

    def group(self, proto: str, members: Sequence[_Op]) -> str:
        if proto == "CanArrayFunction":
            ret = self.returns(members) or "object"
            n = _required_args(members[0].args[0])
            args = ["Any"] * n if n is not None else ["..."]
            return f"CanArrayFunction[CanCall[{', '.join([*args, ret])}], {ret}]"
        parts = [
            arg
            for i in range(len(members[0].args))
            if (arg := self.union(m.args[i] for m in members)) is not None
        ]
        parts += [
            f"{key}={kw}"
            for key in members[0].kwargs
            if (kw := self.union(m.kwargs[key] for m in members)) is not None
        ]
        if (ret := self.returns(members)) is not None:
            parts.append(ret)

        return f"{proto}[{', '.join(parts)}]" if parts else proto

    def traces(self, traces: Iterable[_TraceItem]) -> str:
        groups: dict[tuple[str, int, tuple[str, ...]], list[_Op]] = {}
        for trace in traces:
            op = _resolve(trace)
            key = op.proto, len(op.args), tuple(sorted(op.kwargs))
            groups.setdefault(key, []).append(op)
        parts = [(key[0], self.group(key[0], group)) for key, group in groups.items()]
        wrap = len(parts) > 1
        return " & ".join(
            f"({s})" if wrap and " | " in proto else s for proto, s in parts
        )

    def spy(self, spy: _Spy) -> str:
        return self.traces(spy.__optype_trace__)

    def slot(self, spy: _SpyObject) -> str:
        return self._vars.get(id(spy)) or self.spy(spy) or "object"

    def typevar(self, spy: _SpyObject) -> str:
        var = self._vars[id(spy)]
        return f"{var}: {bound}" if (bound := self.spy(spy)) else var

    def return_type(self, result: object) -> str:
        match result:
            case _SpyObject():
                return self._vars.get(id(result), "object")
            case _SpyStr():
                return "str"
            case _SpyBytes():
                return "bytes"
            case _Gen():
                yields = dict.fromkeys(map(self.return_type, result.yielded))
                return f"{result.kind}[{' | '.join(yields) or 'Never'}]"
            case None:
                return "None"
            case _:
                return self._container(result)

    def _container(self, result: object) -> str:
        name = type(result).__name__
        match result:
            case dict():
                mapping = cast("Mapping[object, object]", result)
                key = self.union(mapping) or "object"
                val = self.union(mapping.values()) or "object"
                return f"dict[{key}, {val}]" if mapping else name
            case list() | set() | frozenset():
                inner = self.union(cast("Collection[object]", result))
                return f"{name}[{inner}]" if inner else name
            case _:
                module = type(result).__module__.partition(".")[0]
                return f"np.{name}" if module == "numpy" else name

    def return_types(self) -> str:
        return " | ".join(dict.fromkeys(map(self.return_type, self._results)))

    def render(self) -> str:
        typevars = [self.typevar(spy) for spy in self._pool + self._result_spies]
        generics = f"[{', '.join(typevars)}]" if typevars else ""
        params = ", ".join(
            f"{name}: {slot}"
            for name in self._selected
            if (slot := self.slot(self._spies[name])) != "object"
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


def _ufunc_nin(func: _AnyFunc) -> int | None:
    nin, nout = getattr(func, "nin", None), getattr(func, "nout", None)
    if isinstance(nin, int) and isinstance(nout, int):
        return nin
    return None


_DTYPE_KINDS = (
    ("?", "ToBoolND"),
    ("bhilqpBHILQP", "ToIntND"),
    ("efdg", "ToFloatND"),
    ("FDG", "ToComplexND"),
)
_DTYPE_RANK: dict[str, int] = {
    char: rank for rank, (chars, _) in enumerate(_DTYPE_KINDS) for char in chars
}
_DTYPE_ALIASES = [alias for _, alias in _DTYPE_KINDS]


def _ufunc_dtype(func: _AnyFunc, i: int) -> str | None:
    types = cast("list[str]", getattr(func, "types", ()))
    ranks = [
        _DTYPE_RANK[ins[i]]
        for sig in types
        if i < len(ins := sig.partition("->")[0]) and ins[i] in _DTYPE_RANK
    ]
    return _DTYPE_ALIASES[max(ranks)] if ranks else None


def _infer_ufunc(func: _AnyFunc, nin: int, params: tuple[str | int, ...]) -> str:
    names = ["x"] if nin == 1 else [f"x{i + 1}" for i in range(nin)]
    parts: list[str] = []
    for name in _select(params, names):
        i = names.index(name)
        arms: list[str] = ["CanArrayUFunc[np.ufunc, R]"] if i == 0 else []
        if (dtype := _ufunc_dtype(func, i)) is not None:
            arms.append(dtype)
        parts.append(f"{name}: {' | '.join(arms) or 'object'}")
    return f"[R]({', '.join(parts)}) -> R"


def _parameters(func: _AnyFunc) -> Mapping[str, Parameter]:
    try:
        return signature(func).parameters
    except ValueError as exc:
        if (names := _doc_params(func)) is None:
            raise NotImplementedError(str(exc)) from exc
        return {n: Parameter(n, Parameter.POSITIONAL_OR_KEYWORD) for n in names}


def _reflect(param_spies: Sequence[_SpyObject], results: Iterable[object]) -> None:
    order, _ = _analyze(param_spies, results)
    added: defaultdict[int, list[_TraceItem]] = defaultdict(list)
    for spy in order:
        keep: list[_TraceItem] = []
        for item in spy.__optype_trace__:
            rhs = item.args[0] if item.args else None
            if item.attr in _DUNDER_CAN_R and isinstance(rhs, _SpyObject):
                reflected = _TraceItem("__r" + item.attr[2:], (spy,), {}, item.return_)
                added[id(rhs)].append(reflected)
            else:
                keep.append(item)
        spy.__optype_trace__ = keep

    for spy in order:
        spy.__optype_trace__ += added[id(spy)]


def _await[R](coro: Coroutine[Any, Any, R]) -> R:
    # a spy's awaitables resolve synchronously, so the coroutine runs straight through
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    coro.close()
    raise NotImplementedError("await on a non-spy awaitable")


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
def _next(result: Generator[object]) -> _Gen: ...
@overload
def _next(result: AsyncGenerator[object]) -> _Gen: ...
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
    args: list[_SpyObject],
    kwds: dict[str, _SpyObject],
) -> list[T]:
    results: list[T] = []
    stack: list[list[bool]] = [[]]
    while stack:
        plan = stack.pop()
        token = _fork.set(iter(plan))
        try:
            result = func(*args, **kwds)
            results.append(_await(result) if iscoroutine(result) else cast("T", result))
        except _Fork:
            if len(plan) < _FORK_LIMIT:
                stack.extend(([*plan, False], [*plan, True]))
        finally:
            _fork.reset(token)
    return results


def infer(func: _AnyFunc, /, *params: str | int) -> str:
    """Infer the ``optype`` protocol(s) required of ``func``'s parameters.

    Pass parameter names or positions to report only those parameters.

    >>> print(infer(lambda x: x + 1))
    [R](x: CanAdd[Literal[1], R]) -> R
    """
    if (nin := _ufunc_nin(func)) is not None:
        return _infer_ufunc(func, nin, params)

    parameters = _parameters(func)
    if any(p.kind in _PARAM_VAR for p in parameters.values()):
        raise NotImplementedError("variadic parameters")

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

    sig1 = _Renderer(names, selected, spies, results, optional).render()
    _reflect(list(spies.values()), results)
    sig2 = _Renderer(names, selected, spies, results, optional).render()
    return "\n".join(dict.fromkeys((sig1, sig2)))
