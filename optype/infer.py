# ruff: noqa: TD002, TD003, PYI034, T201
"""Structurally infer the ``optype`` protocols required by a function."""

import ast
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping
from inspect import Parameter, signature
from typing import Any, NamedTuple, final, override

from optype._core import _can, _has
from optype.inspect import get_protocol_members

__all__ = ("infer",)

type _AnyFunc = Callable[..., Any]

_VARIADIC = frozenset({Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD})

_TYPEVARS = "TUVWXYZ"

_ATTRIBUTE_DUNDERS = frozenset({
    "__delattr__",
    "__getattr__",
    "__getattribute__",
    "__setattr__",
})

_DUNDER_PROTOCOL_MAP = {
    dunder: name
    for name in _can.__all__
    if not name.endswith(("Self", "Same"))
    if (dunder := "__" + name.removeprefix("Can").lower() + "__")
    not in _ATTRIBUTE_DUNDERS
}

_ATTR_PROTOCOL_MAP = {
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
    dunder: " | ".join(map(_DUNDER_PROTOCOL_MAP.__getitem__, (dunder, *fallback)))
    for dunder, fallback in _COERCION_FALLBACK.items()
}

_FORWARD_ARITH = frozenset(
    dunder
    for dunder, proto in _DUNDER_PROTOCOL_MAP.items()
    if "CanR" + proto.removeprefix("Can") in _DUNDER_PROTOCOL_MAP.values()
)


class _TraceItem(NamedTuple):
    attr: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    return_: Any


class _Spy:
    __optype_trace__: list[_TraceItem]

    def __optype_trace_add__[OutT](  # noqa: PLW3201
        self,
        attr: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        out: OutT,
    ) -> OutT:
        self.__optype_trace__.append(_TraceItem(attr, args, kwargs, out))
        return out

    def __init__(self, /, *_args: object, **_kwargs: object) -> None:
        self.__optype_trace__ = []


@final
class _SpyStr(str, _Spy):
    __slots__ = ()  # pyrefly:ignore[implicit-any-attribute]

    # TODO: override __eq__ to figure out what the expected value is


@final
class _SpyBytes(bytes, _Spy):
    __slots__ = ()  # pyrefly:ignore[implicit-any-attribute]

    # TODO: override __eq__ to figure out what the expected value is


class _SpyObject(_Spy):  # noqa: PLR0904
    ###

    def __getattr__(self, attr: str, /) -> "_SpyObject | None":
        # TODO: specialize for known special dunder attrs, e.g. `__name__: str`
        if attr.startswith("__optype"):
            return None

        return self.__optype_trace_add__("__getattr__", (attr,), {}, _SpyObject())

    @override
    def __setattr__(self, attr: str, value: object, /) -> None:
        if attr.startswith("__optype"):
            return super().__setattr__(attr, value)

        return self.__optype_trace_add__("__setattr__", (attr, value), {}, None)

    @override
    def __delattr__(self, attr: str, /) -> None:
        self.__optype_trace_add__("__delattr__", (attr,), {}, None)

    @override
    def __dir__(self, /) -> "_SpyObject":
        # TODO: maybe return specialized `_SpyObject & Iterable[str]`
        return self.__optype_trace_add__("__dir__", (), {}, _SpyObject())

    # TODO: __class__ -> _SpyType

    ###

    @override
    def __repr__(self, /) -> _SpyStr:
        return self.__optype_trace_add__("__repr__", (), {}, _SpyStr())

    @override
    def __str__(self, /) -> _SpyStr:
        return self.__optype_trace_add__("__str__", (), {}, _SpyStr())

    @override
    def __format__(self, format_spec: str, /) -> _SpyStr:
        return self.__optype_trace_add__("__format__", (format_spec,), {}, _SpyStr())

    def __bytes__(self, /) -> _SpyBytes:
        return self.__optype_trace_add__("__bytes__", (), {}, _SpyBytes())

    ###

    @override
    def __eq__(self, other: object, /) -> "_SpyObject":  # type:ignore[override] # pyright:ignore[reportIncompatibleMethodOverride] # ty:ignore[invalid-method-override]
        return self.__optype_trace_add__("__eq__", (other,), {}, _SpyObject())

    @override
    def __ne__(self, other: object, /) -> "_SpyObject":  # type:ignore[override] # pyright:ignore[reportIncompatibleMethodOverride] # ty:ignore[invalid-method-override]
        return self.__optype_trace_add__("__ne__", (other,), {}, _SpyObject())

    def __lt__(self, other: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__lt__", (other,), {}, _SpyObject())

    def __le__(self, other: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__le__", (other,), {}, _SpyObject())

    def __gt__(self, other: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__gt__", (other,), {}, _SpyObject())

    def __ge__(self, other: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__ge__", (other,), {}, _SpyObject())

    ###

    @override
    def __hash__(self, /) -> int:
        return self.__optype_trace_add__("__hash__", (), {}, super().__hash__())

    def __bool__(self, /) -> bool:
        # TODO: fork (by raising); and try both `True` and `False` paths
        return self.__optype_trace_add__("__bool__", (), {}, True)

    ###

    def __get__(self, instance: object, owner: type | None = None) -> "_SpyObject":
        return self.__optype_trace_add__("__get__", (instance, owner), {}, _SpyObject())

    def __set__(self, instance: object, value: object, /) -> None:
        self.__optype_trace_add__("__set__", (instance, value), {}, None)

    def __delete__(self, instance: object, /) -> None:
        self.__optype_trace_add__("__delete__", (instance,), {}, None)

    # TODO: __objclass__ -> _SpyType

    def __set_name__(self, owner: type, name: str, /) -> None:
        self.__optype_trace_add__("__set_name__", (owner, name), {}, None)

    ###

    # TODO: __mro_entries__
    # TODO: __instancecheck__
    # TODO: __subclasscheck__

    ###

    def __call__(self, /, *args: object, **kwargs: object) -> "_SpyObject":
        return self.__optype_trace_add__("__call__", args, kwargs, _SpyObject())

    ###

    def __len__(self, /) -> int:
        # TODO: fork (by raising); and try 0, 1, ...
        return self.__optype_trace_add__("__len__", (), {}, 1)

    # no need for `__length_hint__`

    def __getitem__(self, key: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__getitem__", (key,), {}, _SpyObject())

    def __setitem__(self, key: object, value: object, /) -> None:
        return self.__optype_trace_add__("__setitem__", (key, value), {}, None)

    def __delitem__(self, key: object, /) -> None:
        return self.__optype_trace_add__("__delitem__", (key,), {}, None)

    # no need for `__missing__`

    def __iter__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__iter__", (), {}, _SpyObject())

    def __reversed__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__reversed__", (), {}, _SpyObject())

    def __contains__(self, item: object, /) -> bool:
        # TODO: fork (by raising); and try both `True` and `False` paths
        return self.__optype_trace_add__("__contains__", (item,), {}, True)

    # return `Any` instead of `_SpyObject` to avoid an LSP error for `__dir__`
    def __next__(self, /) -> Any:
        return self.__optype_trace_add__("__next__", (), {}, _SpyObject())

    ###

    def __add__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__add__", (rhs,), {}, _SpyObject())

    def __sub__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__sub__", (rhs,), {}, _SpyObject())

    def __mul__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__mul__", (rhs,), {}, _SpyObject())

    def __matmul__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__matmul__", (rhs,), {}, _SpyObject())

    def __truediv__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__truediv__", (rhs,), {}, _SpyObject())

    def __floordiv__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__floordiv__", (rhs,), {}, _SpyObject())

    def __mod__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__mod__", (rhs,), {}, _SpyObject())

    def __divmod__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__divmod__", (rhs,), {}, _SpyObject())

    def __pow__(self, rhs: object, /, *args: object) -> "_SpyObject":
        return self.__optype_trace_add__("__pow__", (rhs, *args), {}, _SpyObject())

    def __lshift__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__lshift__", (rhs,), {}, _SpyObject())

    def __rshift__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rshift__", (rhs,), {}, _SpyObject())

    def __and__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__and__", (rhs,), {}, _SpyObject())

    def __xor__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__xor__", (rhs,), {}, _SpyObject())

    def __or__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__or__", (rhs,), {}, _SpyObject())

    ###

    def __radd__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__radd__", (lhs,), {}, _SpyObject())

    def __rsub__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rsub__", (lhs,), {}, _SpyObject())

    def __rmul__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rmul__", (lhs,), {}, _SpyObject())

    def __rmatmul__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rmatmul__", (lhs,), {}, _SpyObject())

    def __rtruediv__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rtruediv__", (lhs,), {}, _SpyObject())

    def __rfloordiv__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rfloordiv__", (lhs,), {}, _SpyObject())

    def __rmod__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rmod__", (lhs,), {}, _SpyObject())

    def __rdivmod__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rdivmod__", (lhs,), {}, _SpyObject())

    def __rpow__(self, lhs: object, /, *args: object) -> "_SpyObject":
        return self.__optype_trace_add__("__rpow__", (lhs, *args), {}, _SpyObject())

    def __rlshift__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rlshift__", (lhs,), {}, _SpyObject())

    def __rrshift__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rrshift__", (lhs,), {}, _SpyObject())

    def __rand__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rand__", (lhs,), {}, _SpyObject())

    def __rxor__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__rxor__", (lhs,), {}, _SpyObject())

    def __ror__(self, lhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__ror__", (lhs,), {}, _SpyObject())

    ###

    def __iadd__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__iadd__", (rhs,), {}, _SpyObject())

    def __isub__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__isub__", (rhs,), {}, _SpyObject())

    def __imul__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__imul__", (rhs,), {}, _SpyObject())

    def __imatmul__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__imatmul__", (rhs,), {}, _SpyObject())

    def __itruediv__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__itruediv__", (rhs,), {}, _SpyObject())

    def __ifloordiv__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__ifloordiv__", (rhs,), {}, _SpyObject())

    def __imod__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__imod__", (rhs,), {}, _SpyObject())

    def __ipow__(self, rhs: object, /, *args: object) -> "_SpyObject":
        return self.__optype_trace_add__("__ipow__", (rhs, *args), {}, _SpyObject())

    def __ilshift__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__ilshift__", (rhs,), {}, _SpyObject())

    def __irshift__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__irshift__", (rhs,), {}, _SpyObject())

    def __iand__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__iand__", (rhs,), {}, _SpyObject())

    def __ixor__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__ixor__", (rhs,), {}, _SpyObject())

    def __ior__(self, rhs: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__ior__", (rhs,), {}, _SpyObject())

    ###

    def __neg__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__neg__", (), {}, _SpyObject())

    def __pos__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__pos__", (), {}, _SpyObject())

    def __abs__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__abs__", (), {}, _SpyObject())

    def __invert__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__invert__", (), {}, _SpyObject())

    ###

    def __complex__(self, /) -> complex:
        return self.__optype_trace_add__("__complex__", (), {}, 0j)

    def __float__(self, /) -> float:
        return self.__optype_trace_add__("__float__", (), {}, 0.0)

    def __int__(self, /) -> int:
        # TODO: fork (by raising); and try 0, 1, ...
        return self.__optype_trace_add__("__int__", (), {}, 0)

    def __index__(self, /) -> int:
        # TODO: fork (by raising); and try 0, 1, ...
        return self.__optype_trace_add__("__index__", (), {}, 0)

    ###

    def __round__(self, /, *args: object) -> "_SpyObject":
        return self.__optype_trace_add__("__round__", args, {}, _SpyObject())

    def __trunc__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__trunc__", (), {}, _SpyObject())

    def __floor__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__floor__", (), {}, _SpyObject())

    def __ceil__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__ceil__", (), {}, _SpyObject())

    ###

    def __enter__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__enter__", (), {}, _SpyObject())

    def __exit__(self, /, *args: object) -> None:
        # TODO: maybe fork and return falsy/truthy in case of exception??
        return self.__optype_trace_add__("__exit__", args, {}, None)

    ###

    def __buffer__(self, flags: int, /) -> memoryview:
        return self.__optype_trace_add__("__buffer__", (flags,), {}, memoryview(b""))

    def __release_buffer__(self, buffer: memoryview, /) -> None:
        return self.__optype_trace_add__("__releasebuffer__", (buffer,), {}, None)

    ###

    def __await__(self, /) -> Generator[Any, None, "_SpyObject"]:
        out = _SpyObject()

        def spy_generator() -> Generator[Any, None, "_SpyObject"]:
            yield from ()
            return out  # noqa: B901

        return self.__optype_trace_add__("__await__", (), {}, spy_generator())

    def __aiter__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__aiter__", (), {}, _SpyObject())

    def __anext__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__anext__", (), {}, _SpyObject())

    def __aenter__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__aenter__", (), {}, _SpyObject())

    def __aexit__(self, /, *args: object) -> None:
        return self.__optype_trace_add__("__aexit__", args, {}, None)


class _Op(NamedTuple):
    proto: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    ret: Any


type _Vars = dict[int, str]


def _resolve(trace: _TraceItem) -> _Op:
    if trace.attr in _ATTRIBUTE_DUNDERS:
        name = trace.args[0]
        if name not in _ATTR_PROTOCOL_MAP:
            raise NotImplementedError(name)
        return _Op(_ATTR_PROTOCOL_MAP[name], (), {}, trace.return_)
    if trace.attr in _COERCION_UNION:
        return _Op(_COERCION_UNION[trace.attr], (), {}, trace.return_)
    if trace.attr in _DUNDER_PROTOCOL_MAP:
        proto = _DUNDER_PROTOCOL_MAP[trace.attr]
        return _Op(proto, trace.args, trace.kwargs, trace.return_)
    raise NotImplementedError(trace.attr)


def _analyze(
    params: list[_SpyObject],
    result: object,
) -> tuple[list[_SpyObject], dict[int, int]]:
    appear: defaultdict[int, int] = defaultdict(int)
    for spy in params:
        appear[id(spy)] += 1
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


@final
class _Renderer:
    """Render an inferred ``def`` signature from the recorded spy traces."""

    def __init__(
        self,
        names: list[str],
        selected: list[str],
        spies: dict[str, _SpyObject],
        result: object,
        optional: frozenset[str],
    ) -> None:
        self._selected = selected
        self._spies = spies
        self._result = result
        self._optional = optional

        param_spies = [spies[name] for name in names]
        order, appear = _analyze(param_spies, result)
        param_ids = {id(spy) for spy in param_spies}

        self._has_result = (
            isinstance(result, _SpyObject) and id(result) not in param_ids
        )
        self._vars: _Vars = {id(result): "R"} if self._has_result else {}
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

    def union(self, values: list[Any]) -> str | None:
        concrete = [value for value in values if not isinstance(value, _Spy)]
        literals = [value for value in concrete if isinstance(value, int | str | bytes)]
        names = [
            "None" if value is None else type(value).__name__
            for value in concrete
            if not isinstance(value, int | str | bytes)
        ]

        if "float" in names or "complex" in names:
            literals = [value for value in literals if not isinstance(value, int)]
        if "complex" in names:
            names = [name for name in names if name != "float"]

        names += [
            self._vars[id(value)]
            for value in values
            if isinstance(value, _Spy) and id(value) in self._vars
        ]

        parts: list[str] = []
        if literals:
            parts.append(f"Literal[{', '.join(map(repr, dict.fromkeys(literals)))}]")
        parts.extend(dict.fromkeys(names))
        return " | ".join(parts) or None

    def returns(self, members: list[_Op]) -> str | None:
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

    def group(self, proto: str, members: list[_Op]) -> str:
        parts = [
            arg
            for i in range(len(members[0].args))
            if (arg := self.union([m.args[i] for m in members])) is not None
        ]
        parts += [
            f"{key}={kw}"
            for key in members[0].kwargs
            if (kw := self.union([m.kwargs[key] for m in members])) is not None
        ]
        if (ret := self.returns(members)) is not None:
            parts.append(ret)

        return f"{proto}[{', '.join(parts)}]" if parts else proto

    def traces(self, traces: list[_TraceItem]) -> str:
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
        return self._vars.get(id(spy)) or self.spy(spy) or "Unused"

    def typevar(self, spy: _SpyObject) -> str:
        var = self._vars[id(spy)]
        return f"{var}: {bound}" if (bound := self.spy(spy)) else var

    def return_type(self) -> str:
        match self._result:
            case _SpyObject() as result:
                return self._vars.get(id(result), "object")
            case _SpyStr():
                return "str"
            case _SpyBytes():
                return "bytes"
            case None:
                return "None"
            case result:
                return type(result).__name__

    def render(self) -> str:
        typevars = [self.typevar(spy) for spy in self._pool]
        if self._has_result:
            typevars.append("R")
        generics = f"[{', '.join(typevars)}]" if typevars else ""
        params = ", ".join(
            f"{name}: {slot}"
            for name in self._selected
            # an unused optional param is dropped; an unused required one stays `Unused`
            if (slot := self.slot(self._spies[name])) != "Unused"
            or name not in self._optional
        )
        return f"{generics}({params}) -> {self.return_type()}"


_DOC_SIGNATURE = re.compile(r"\b(\w+)\(([^)]*)\)")
_DOC_PARAM = re.compile(r"(?:^|,)\s*\**([a-zA-Z_]\w*)")


def _doc_params(func: _AnyFunc) -> list[str] | None:
    name = getattr(func, "__name__", "")
    if not name:
        return None
    for match in _DOC_SIGNATURE.finditer(func.__doc__ or ""):
        if match[1] == name:
            params = match[2].replace("[", "").replace("]", "")
            return _DOC_PARAM.findall(params) or None
    return None


def _parameters(func: _AnyFunc) -> Mapping[str, Parameter]:
    try:
        return signature(func).parameters
    except ValueError as exc:
        if (names := _doc_params(func)) is None:
            raise NotImplementedError(str(exc)) from exc
        return {n: Parameter(n, Parameter.POSITIONAL_OR_KEYWORD) for n in names}


def _reflect(param_spies: list[_SpyObject], result: object) -> None:
    order, _ = _analyze(param_spies, result)
    added: defaultdict[int, list[_TraceItem]] = defaultdict(list)
    for spy in order:
        keep: list[_TraceItem] = []
        for item in spy.__optype_trace__:
            rhs = item.args[0] if item.args else None
            if item.attr in _FORWARD_ARITH and isinstance(rhs, _SpyObject):
                reflected = _TraceItem("__r" + item.attr[2:], (spy,), {}, item.return_)
                added[id(rhs)].append(reflected)
            else:
                keep.append(item)
        spy.__optype_trace__ = keep

    for spy in order:
        spy.__optype_trace__ += added[id(spy)]


def _infer(func: _AnyFunc, /, *params: str | int) -> str:
    parameters = _parameters(func)
    if any(p.kind in _VARIADIC for p in parameters.values()):
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
    result = func(*args, **kwds)

    sig1 = _Renderer(names, selected, spies, result, optional).render()
    _reflect(list(spies.values()), result)
    sig2 = _Renderer(names, selected, spies, result, optional).render()
    return sig1 if sig1 == sig2 else f"{sig1}\n{sig2}"


def infer(func: _AnyFunc, /, *params: str | int) -> None:
    """Print the ``optype`` protocol(s) required of ``func``'s parameters.

    Pass parameter names or positions to report only those parameters.

    >>> infer(lambda x: x + 1)
    [R](x: CanAdd[Literal[1], R]) -> R
    """
    print(_infer(func, *params))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m optype.infer EXPR [PARAM ...]")

    source, *selectors = sys.argv[1:]
    params = [int(s) if s.lstrip("-").isdigit() else s for s in selectors]

    body = ast.parse(source).body
    last = body[-1] if body else None
    if not isinstance(last, ast.Expr):
        sys.exit("the final statement must be an expression")

    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body[:-1], []), "<expr>", "exec"), namespace)  # noqa: S102
    code = compile(ast.Expression(last.value), "<expr>", "eval")
    try:
        infer(eval(code, namespace), *params)  # noqa: S307
    except (NotImplementedError, ValueError, TypeError) as exc:
        sys.exit(f"{type(exc).__name__}: {exc}")
