"""Recording proxy objects that trace the operations performed on them."""

import dis
import sys
from collections.abc import Callable, Generator, Iterator
from contextvars import ContextVar
from enum import StrEnum
from functools import lru_cache
from types import CodeType
from typing import Any, ClassVar, NamedTuple, Self, cast, final, override

type _AnyFunc = Callable[..., object]
type _Args = tuple[object, ...]
type _Kwargs = dict[str, object]
type _Memo = tuple[object, int | None]  # (fork plan, value); None value = absent
type _KeyedMemo = tuple[object, dict[int, tuple[object, int]]]


def _slot(attr: str) -> str:
    return f"__optype_{attr.strip('_')}__"


def _internal(attr: str) -> bool:
    """Whether `attr` is a spy's own `__optype*` bookkeeping attribute."""
    return attr.startswith("__optype")


class _Fork(BaseException): ...


class _AbsentError(TypeError):
    """A simulated missing dunder; subclasses `TypeError` so probes suppress it."""


class _Marker(StrEnum):
    """A pseudo-operation trace marker, not a real dunder."""

    ABSENT = "__absent__"  # a simulated-missing dunder
    SIBLING = "__sibling__"  # a `type(spy)(...)` instantiation

    CLASS_DELATTR = "__class_delattr__"
    CLASS_GETATTR = "__class_getattr__"
    CLASS_SETATTR = "__class_setattr__"


_fork: ContextVar[Iterator[bool] | None] = ContextVar("_fork", default=None)

# the yield count for a splatted-call iterator whose arity no bytecode pins: the exact
# arity the splat demands, which `_explore_spies` grows into until the call succeeds
_yield_budget: ContextVar[int] = ContextVar("_yield_budget", default=1)
# set when a growable splatted-call iterator hits the budget, so `_explore_spies` only
# grows the budget when a fixed-arity splat actually came up short
_starved: ContextVar[bool] = ContextVar("_starved", default=False)

# one element exercises no pairwise op, so `sorted`/`min` never reach their elements'
# `__lt__` (#686); a pair suffices, and `_render` inlines the extra typevar away
_DEFAULT_YIELD = 2

# the caller's bytecode is a CPython detail; elsewhere fall back to two elements
_CPYTHON = sys.implementation.name == "cpython"


@lru_cache(maxsize=256)
def _co_code(code: CodeType) -> bytes:
    # `co_code` rebuilds a deoptimized copy on each access, so cache it per code object
    return code.co_code


def _instruction_arg(code: bytes, i: int) -> int:
    arg = code[i + 1]
    shift = 8
    j = i - 2
    while j >= 0 and dis.opname[code[j]] == "EXTENDED_ARG":
        arg |= code[j + 1] << shift
        shift += 8
        j -= 2
    return arg


def _iter_context() -> tuple[int | None, bool]:
    """The fixed arity the caller's unpack demands, and whether it splats into a call.

    `UNPACK_SEQUENCE`/`UNPACK_EX` pin an exact arity; `CALL_FUNCTION_EX` (`f(*x)`) has
    no local arity signal, so its iterator grows via the `_yield_budget` instead.
    """
    if not _CPYTHON:
        return None, False

    try:
        frame = sys._getframe(2)  # _iter_context -> __iter__ -> the consuming frame  # noqa: SLF001
    except ValueError:
        frame = None
    if frame is None or (i := frame.f_lasti) < 0:
        return None, False

    code = _co_code(frame.f_code)
    match dis.opname[code[i]]:
        case "UNPACK_SEQUENCE":
            return _instruction_arg(code, i), False
        case "UNPACK_EX":
            # low byte: items before the star, high byte: after; +1 feeds the star
            arg = _instruction_arg(code, i)
            return (arg & 0xFF) + (arg >> 8) + 1, False
        case "CALL_FUNCTION_EX":
            return None, True
        case _:
            return None, False


def _decide() -> bool:
    if (plan := _fork.get()) is None:
        return True
    if (value := next(plan, None)) is None:
        raise _Fork
    return value


def _decide_stable(spy: "_SpyObject", attr: str, /, *, optional: bool = False) -> int:
    # memoized per run (keyed by the fork plan) so repeats agree; else two disagreeing
    # `len(seq)` send e.g. `random.choice` into a non-terminating `_randbelow(0)`

    slot = _slot(attr)
    plan = _fork.get()

    memo: _Memo | None = getattr(spy, slot)
    if memo is not None and memo[0] is plan:
        if memo[1] is None:
            raise _AbsentError
        return memo[1]

    if optional and not _decide():
        setattr(spy, slot, (plan, None))
        spy.__optype_trace_add__(_Marker.ABSENT, (attr,), {}, None)
        raise _AbsentError

    value = spy.__optype_trace_add__(attr, (), {}, int(_decide()))
    setattr(spy, slot, (plan, value))
    return value


def _decide_keyed(
    spy: "_SpyObject",
    attr: str,
    item: object,
    /,
    *,
    keep_arg: bool,
) -> bool:
    # per-operand variant of `_decide_stable`: `y in x` forks once per distinct `y`, so
    # `y in x and y not in x` agrees within a run while `a in x` and `b in x` stay free.
    # the cache retains `item` so its `id` can't be reused by a later distinct operand
    slot = _slot(attr)
    plan = _fork.get()

    cache: dict[int, tuple[object, int]]
    memo: _KeyedMemo | None = getattr(spy, slot)
    if memo is not None and memo[0] is plan:
        cache = memo[1]
    else:
        cache = {}
        setattr(spy, slot, (plan, cache))

    key = id(item)
    if key not in cache:
        args = (item,) if keep_arg else ()
        cache[key] = item, spy.__optype_trace_add__(attr, args, {}, int(_decide()))
    return bool(cache[key][1])


class _TraceItem(NamedTuple):
    attr: str
    args: _Args
    kwargs: _Kwargs
    return_: object


type _Traces = dict[int, list[_TraceItem]]


class _Spy:
    __optype_trace__: list[_TraceItem]

    def __optype_trace_add__[OutT](
        self,
        attr: str,
        args: _Args,
        kwargs: _Kwargs,
        out: OutT,
    ) -> OutT:
        self.__optype_trace__.append(_TraceItem(attr, args, kwargs, out))
        return out

    def __init__(self, /, *_args: object, **_kwargs: object) -> None:
        self.__optype_trace__ = []


@final
class _SpyStr(str, _Spy):
    __slots__ = ()


@final
class _SpyBytes(bytes, _Spy):
    __slots__ = ()


class _SpyType(type):
    """The metaclass of every spy's unique class, so class attribute access records."""

    def __getattr__(cls, attr: str, /) -> "_SpyObject | None":
        if _internal(attr):
            return None

        if (owner := _class_spy(cls)) is None:
            msg = f"type object {cls.__name__!r} has no attribute {attr!r}"
            raise AttributeError(msg)

        out = _SpyObject()
        return owner.__optype_trace_add__(_Marker.CLASS_GETATTR, (attr,), {}, out)

    @override
    def __setattr__(cls, attr: str, value: object, /) -> None:
        # recorded without mutating, so forked reruns stay clean
        if _internal(attr) or (owner := _class_spy(cls)) is None:
            return super().__setattr__(attr, value)

        args = attr, value
        return owner.__optype_trace_add__(_Marker.CLASS_SETATTR, args, {}, None)

    @override
    def __delattr__(cls, attr: str, /) -> None:
        if _internal(attr) or (owner := _class_spy(cls)) is None:
            return super().__delattr__(attr)

        return owner.__optype_trace_add__(_Marker.CLASS_DELATTR, (attr,), {}, None)


class _SpyObject(_Spy, metaclass=_SpyType):
    __optype_element__: "_SpyObject | None" = None
    __optype_iterator__: bool = False
    __optype_arity__: "int | None" = None
    __optype_growable__: bool = False
    # spies are descriptors (`__get__`), so only ever read through the class `__dict__`
    __optype_instance__: "ClassVar[_SpyObject | None]" = None

    def __new__(cls, /, *_args: object, **_kwargs: object) -> "_SpyObject":
        if cls is not _SpyObject:
            # a `type(spy)(...)` sibling; the marker keeps it reachable from the spy
            self = super().__new__(cls)
            if (owner := _class_spy(cls)) is not None:
                owner.__optype_trace__.append(_TraceItem(_Marker.SIBLING, (), {}, self))
            return self
        # every spy gets a class of its own, so that `type(spy)` identifies the spy
        unique = cast("type[Self]", type("_SpyObject", (cls,), {}))
        self = super().__new__(unique)
        type.__setattr__(unique, "__optype_instance__", self)
        return self

    ###

    def __getattr__(self, attr: str, /) -> "_SpyObject | None":
        # TODO: specialize for known special dunder attrs, e.g. `__name__: str`
        if _internal(attr):
            return None

        return self.__optype_trace_add__("__getattr__", (attr,), {}, _SpyObject())

    @override
    def __setattr__(self, attr: str, value: object, /) -> None:
        if _internal(attr):
            return super().__setattr__(attr, value)

        return self.__optype_trace_add__("__setattr__", (attr, value), {}, None)

    @override
    def __delattr__(self, attr: str, /) -> None:
        self.__optype_trace_add__("__delattr__", (attr,), {}, None)

    @override
    def __dir__(self, /) -> "_SpyObject":
        # TODO: maybe return specialized `_SpyObject & Iterable[str]`
        return self.__optype_trace_add__("__dir__", (), {}, _SpyObject())

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

    ###

    @override
    def __hash__(self, /) -> int:
        return self.__optype_trace_add__("__hash__", (), {}, super().__hash__())

    def __bool__(self, /) -> bool:
        return bool(_decide_stable(self, "__bool__"))

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

    def __instancecheck__(self, instance: object, /) -> bool:
        return _decide_keyed(self, "__instancecheck__", instance, keep_arg=False)

    def __subclasscheck__(self, subclass: object, /) -> bool:
        return _decide_keyed(self, "__subclasscheck__", subclass, keep_arg=False)

    ###

    def __call__(self, /, *args: object, **kwargs: object) -> "_SpyObject":
        return self.__optype_trace_add__("__call__", args, kwargs, _SpyObject())

    ###

    def __len__(self, /) -> int:
        # `len()` may be probed optionally (e.g. `list()` via `length_hint`)
        return _decide_stable(self, "__len__", optional=True)

    # no need for `__length_hint__`

    def __getitem__(self, key: object, /) -> "_SpyObject":
        return self.__optype_trace_add__("__getitem__", (key,), {}, _SpyObject())

    def __setitem__(self, key: object, value: object, /) -> None:
        return self.__optype_trace_add__("__setitem__", (key, value), {}, None)

    def __delitem__(self, key: object, /) -> None:
        return self.__optype_trace_add__("__delitem__", (key,), {}, None)

    # no need for `__missing__`

    def __iter__(self, /) -> "_SpyObject":
        arity, growable = _iter_context()

        if self.__optype_iterator__:
            if arity is not None:
                self.__optype_arity__ = arity
            if growable:
                self.__optype_growable__ = True
            return self  # an iterator is its own iterable (idempotent `iter()`)

        out = _iterator_of(self)
        out.__optype_arity__ = arity
        out.__optype_growable__ = growable
        return self.__optype_trace_add__("__iter__", (), {}, out)

    def __reversed__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__reversed__", (), {}, _iterator_of(self))

    def __contains__(self, item: object, /) -> bool:
        return _decide_keyed(self, "__contains__", item, keep_arg=True)

    # return `Any` instead of `_SpyObject` to avoid an LSP error for `__dir__`
    def __next__(self, /) -> Any:
        # count from the trace, not a field, so a forked run's rollback is reflected
        served = sum(1 for item in self.__optype_trace__ if item.attr == "__next__")
        arity, growable = self.__optype_arity__, self.__optype_growable__
        limit = (
            arity
            if arity is not None
            else _yield_budget.get()
            if growable
            else _DEFAULT_YIELD
        )
        if served >= limit:
            if growable and arity is None:
                _starved.set(True)
            raise StopIteration
        return self.__optype_trace_add__("__next__", (), {}, _element_of(self))

    ###

    def __complex__(self, /) -> complex:
        return self.__optype_trace_add__("__complex__", (), {}, 0j)

    def __float__(self, /) -> float:
        return self.__optype_trace_add__("__float__", (), {}, 0.0)

    def __int__(self, /) -> int:
        return _decide_stable(self, "__int__")

    def __index__(self, /) -> int:
        return _decide_stable(self, "__index__")

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
        return self.__optype_trace_add__("__release_buffer__", (buffer,), {}, None)

    ###

    # numpy looks these up on the type, so `__getattr__` won't do
    def __array_ufunc__(
        self,
        ufunc: _AnyFunc,
        method: str,
        /,
        *inputs: object,
        **kwargs: object,
    ) -> "_SpyObject":
        return self.__optype_trace_add__("__array_ufunc__", (ufunc,), {}, _SpyObject())

    def __array_function__(
        self,
        func: _AnyFunc,
        types: object,
        args: object,
        kwargs: object,
        /,
    ) -> "_SpyObject":
        return self.__optype_trace_add__(
            "__array_function__",
            (func,),
            {},
            _SpyObject(),
        )

    ###

    def __await__(self, /) -> Generator[Any, None, "_SpyObject"]:
        out = self.__optype_trace_add__("__await__", (), {}, _SpyObject())

        def spy_generator() -> Generator[Any, None, "_SpyObject"]:
            yield from ()
            return out  # noqa: B901

        return spy_generator()

    def __aiter__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__aiter__", (), {}, _SpyObject())

    def __anext__(self, /) -> "_SpyObject":
        if any(item.attr == "__anext__" for item in self.__optype_trace__):
            raise StopAsyncIteration
        return self.__optype_trace_add__("__anext__", (), {}, _SpyObject())

    def __aenter__(self, /) -> "_SpyObject":
        return self.__optype_trace_add__("__aenter__", (), {}, _SpyObject())

    def __aexit__(self, /, *args: object) -> "_SpyObject":
        return self.__optype_trace_add__("__aexit__", args, {}, _SpyObject())


# Operators that record their positional args and return a fresh spy. Generated onto
# the type (not synthesized in `__getattr__`) because special-method lookup skips it.
_TRACED_OPS = (
    "__neg__",
    "__pos__",
    "__abs__",
    "__invert__",
    "__round__",
    "__trunc__",
    "__floor__",
    "__ceil__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__add__",
    "__sub__",
    "__mul__",
    "__matmul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__xor__",
    "__or__",
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rmatmul__",
    "__rtruediv__",
    "__rfloordiv__",
    "__rmod__",
    "__rdivmod__",
    "__rpow__",
    "__rlshift__",
    "__rrshift__",
    "__rand__",
    "__rxor__",
    "__ror__",
    "__iadd__",
    "__isub__",
    "__imul__",
    "__imatmul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__ilshift__",
    "__irshift__",
    "__iand__",
    "__ixor__",
    "__ior__",
)


def _traced_op(name: str) -> _AnyFunc:
    def op(self: _SpyObject, /, *args: object) -> _SpyObject:
        return self.__optype_trace_add__(name, args, {}, _SpyObject())

    op.__name__ = op.__qualname__ = name
    return op


for _name in _TRACED_OPS:
    # bypass the metaclass: `_class_spy` isn't defined yet, and this isn't a trace
    type.__setattr__(_SpyObject, _name, _traced_op(_name))  # noqa: PLC2801


# Free functions, not methods: a method would be an unrecorded hole in the proxy.
def _class_spy(cls: object) -> _SpyObject | None:
    """The spy whose unique class `cls` is, if any."""
    if isinstance(cls, type) and issubclass(cls, _SpyObject):
        spy = cls.__dict__.get("__optype_instance__")
        if isinstance(spy, _SpyObject):
            return spy
    return None


def _own_spy(spy: _SpyObject) -> _SpyObject:
    """The first spy of `spy`'s class: `type(spy)()` siblings collapse onto it."""
    owner = _class_spy(type(spy))  # `or spy` would trace a `__bool__` on the owner
    return spy if owner is None else owner


def as_spy(value: object) -> _SpyObject | None:
    """The (first-of-its-class) spy itself, or the spy whose class it is, if any."""
    # a `weakref.proxy` forwards `__class__`, so verify its real class is a spy's
    if isinstance(value, _SpyObject) and _class_spy(type(value)) is not None:
        return _own_spy(value)
    return _class_spy(value)


def _element_of(spy: _SpyObject) -> _SpyObject:
    if (element := spy.__optype_element__) is None:
        element = _SpyObject()
        spy.__optype_element__ = element  # ty:ignore[invalid-assignment]
    return element


def _iterator_of(spy: _SpyObject) -> _SpyObject:
    iterator = _SpyObject()
    iterator.__optype_element__ = _element_of(spy)  # ty:ignore[invalid-assignment]
    iterator.__optype_iterator__ = True
    return iterator
