"""Recording proxy objects that trace the operations performed on them."""

import dis
import mmap
import sys
from collections.abc import Callable, Generator, Iterator
from contextvars import ContextVar
from enum import StrEnum
from functools import lru_cache
from types import CodeType
from typing import Any, ClassVar, NamedTuple, Self, TypeGuard, final, override

type AnyFunc = Callable[..., object]
type Args = tuple[object, ...]
type Kwargs = dict[str, object]

# per-run memos: (fork plan, value), `None` if decided absent; the keyed memo keeps
# each operand alive so its id is not reused
type _Memo = tuple[object, int | None]
type _KeyedMemo = tuple[object, dict[int, tuple[object, int]]]


def _internal(attr: str) -> bool:
    """Whether `attr` is a spy's own `__optype*` bookkeeping attribute."""
    return attr.startswith("__optype")


def dynamic_name(attr: str) -> bool:
    """Whether `attr` is spy-derived, e.g. built from a spy's type name (#777)."""

    if isinstance(attr, Spy):
        # e.g. `SpyStr = str & Spy`
        return True

    low = attr.lower()
    return any(name in low for name in _SPY_NAMES)


class Fork(BaseException): ...


class AbsentError(TypeError):
    """A simulated missing dunder; subclasses `TypeError` so probes suppress it."""


class DynamicNameError(AttributeError):
    """A spy-derived attribute name; subclasses `AttributeError` so fallbacks run."""

    # the parameter keeps `cls(*args)` reconstruction working, e.g. unpickling
    def __init__(
        self,
        message: str = "no protocol for a dynamic attribute name",
        /,
    ) -> None:
        super().__init__(message)


class Marker(StrEnum):
    """A pseudo-operation trace marker, not a real dunder."""

    ABSENT = "__absent__"  # a simulated-missing dunder
    SIBLING = "__sibling__"  # a `type(spy)(...)` instantiation

    CLASS_DELATTR = "__class_delattr__"
    CLASS_GETATTR = "__class_getattr__"
    CLASS_SETATTR = "__class_setattr__"


fork_plan: ContextVar[Iterator[bool] | None] = ContextVar("fork_plan", default=None)

# the yield count for a star-unpack iterator (`f(*x)`) whose arity no bytecode pins:
# the arity it needs, which `explore_spies` grows into until the call works
yield_budget: ContextVar[int] = ContextVar("yield_budget", default=1)
# set when a growable star-unpack iterator hits the budget, so `explore_spies` only
# grows the budget when a fixed-arity star unpacking actually came up short
starved: ContextVar[bool] = ContextVar("starved", default=False)

# one element exercises no pairwise op, so `sorted`/`min` never reach their elements'
# `__lt__` (#686); a pair suffices, and `_render` inlines the extra typevar away
_DEFAULT_YIELD = 2

# `_explore._run` unpacks a real arg list, so a spy `__iter__` charged to its frame is a
# C builtin iterating internally, not a growable star-unpack (#723)
_driver_code: CodeType | None = None


def set_driver_code[T: Callable[..., object]](fn: T, /) -> T:
    global _driver_code  # ruff: ignore[global-statement]
    _driver_code = fn.__code__  # ty: ignore[unresolved-attribute]
    return fn


# a shared buffer into which each spy operation writes itself, so the last action
# survives a native crash for `_isolate` to report (#738)
_state_buffer: mmap.mmap | None = None


def set_state_buffer(buf: mmap.mmap | None, /) -> None:
    global _state_buffer  # ruff: ignore[global-statement]
    _state_buffer = buf


# star-unpack detection reads caller bytecode (CPython detail); else it never fires
_CPYTHON = sys.implementation.name == "cpython"


@lru_cache(maxsize=256)
def _co_code(code: CodeType) -> bytes:
    # `co_code` rebuilds a deoptimized copy on each access, so cache it per code object
    return code.co_code


def _iter_is_star_unpack() -> bool:
    """Whether the caller iterates by star-unpacking into a call, as in `f(*x)`.

    `CALL_FUNCTION_EX` has no local arity signal, so its iterator grows via the
    `yield_budget` until the call's arity is met.
    """
    if not _CPYTHON:
        return False

    try:
        frame = sys._getframe(2)  # _iter_is_star_unpack -> __iter__ -> consuming frame  # ruff: ignore[private-member-access]
    except ValueError:
        frame = None
    if frame is None or (i := frame.f_lasti) < 0 or frame.f_code is _driver_code:
        return False

    return dis.opname[_co_code(frame.f_code)[i]] == "CALL_FUNCTION_EX"


def _decide() -> bool:
    if (plan := fork_plan.get()) is None:
        return True
    if (value := next(plan, None)) is None:
        raise Fork
    return value


def _decide_stable(spy: "SpyObject", attr: str, /, *, optional: bool = False) -> int:
    # memoized per run (keyed by the fork plan) so repeats agree; else two disagreeing
    # `len(seq)` send e.g. `random.choice` into a non-terminating `_randbelow(0)`
    plan = fork_plan.get()
    memos = spy.__optype_stable__

    memo = memos.get(attr)
    if memo is not None and memo[0] is plan:
        if memo[1] is None:
            raise AbsentError
        return memo[1]

    if optional and not _decide():
        memos[attr] = plan, None
        spy.__optype_trace_add__(Marker.ABSENT, (attr,), {}, None)
        raise AbsentError

    value = spy.__optype_trace_add__(attr, (), {}, int(_decide()))
    memos[attr] = plan, value
    return value


def _decide_keyed(
    spy: "SpyObject",
    attr: str,
    item: object,
    /,
    *,
    keep_arg: bool,
) -> bool:
    # per-operand `_decide_stable`: `y in x and y not in x` agrees within a run, while
    # `a in x` and `b in x` stay free
    plan = fork_plan.get()
    memos = spy.__optype_keyed__

    cache: dict[int, tuple[object, int]]
    memo = memos.get(attr)
    if memo is not None and memo[0] is plan:
        cache = memo[1]
    else:
        cache = {}
        memos[attr] = plan, cache

    key = id(item)
    if key not in cache:
        args = (item,) if keep_arg else ()
        cache[key] = item, spy.__optype_trace_add__(attr, args, {}, int(_decide()))
    return bool(cache[key][1])


class TraceItem(NamedTuple):
    attr: str
    args: Args
    kwargs: Kwargs
    return_: object


type Traces = dict[int, list[TraceItem]]


class Spy:
    __optype_trace__: list[TraceItem]

    def __optype_trace_add__[OutT](
        self,
        attr: str,
        args: Args,
        kwargs: Kwargs,
        out: OutT,
    ) -> OutT:
        _journal_touch(self)
        item = TraceItem(attr, args, kwargs, out)
        self.__optype_trace__.append(item)
        if _state_buffer is not None:
            _record_state(item, _state_buffer)
        return out

    def __init__(self, /, *_args: object, **_kwargs: object) -> None:
        self.__optype_trace__ = []


# while a run explores, each touched spy's pre-run trace length, so `_explore` can
# undo a rejected run by truncating only the spies it actually appended to
journal: ContextVar[dict[int, tuple[Spy, int]] | None] = ContextVar(
    "journal",
    default=None,
)


def _journal_touch(spy: Spy) -> None:
    if (marks := journal.get()) is not None and id(spy) not in marks:
        marks[id(spy)] = (spy, len(spy.__optype_trace__))


def journal_rollback(marks: dict[int, tuple[Spy, int]], /) -> None:
    for spy, length in marks.values():
        del spy.__optype_trace__[length:]


@final
class SpyStr(str, Spy):
    __slots__ = ()


@final
class SpyBytes(bytes, Spy):
    __slots__ = ()


def _brief(value: Any, /) -> str:
    # no arbitrary `repr` here: a container's repr reaches the dunders of its
    # elements, recording phantom ops on any spy inside (#763)
    if isinstance(value, Spy):
        return "spy"

    cls = type(value)  # pyright: ignore[reportUnknownVariableType]
    if cls in {tuple, list}:
        left, right = "()" if cls is tuple else "[]"
        text = left + ", ".join(map(_brief, value)) + right
    elif value is None or cls in {bool, int, float, complex, bytes, str}:
        text = repr(value)
    else:
        text = cls.__name__

    return text if len(text) <= 32 else text[:31] + "..."


def _record_state(item: TraceItem, buf: mmap.mmap, /) -> None:
    # the last completed op; a crash strikes between ops, in native code (#738)
    call = f"{item.attr}({', '.join(_brief(a) for a in item.args)})"
    ret = "" if item.return_ is None else f" -> {_brief(item.return_)}"
    data = (call + ret).encode("utf-8", "replace")[: len(buf) - 1] + b"\x00"
    buf.seek(0)
    buf.write(data)


class _SpyType(type):
    """The metaclass of every spy's unique class, so class attribute access records."""

    def __getattr__(cls, attr: str, /) -> "SpyObject | None":
        if _internal(attr):
            return None
        if dynamic_name(attr):
            raise DynamicNameError

        if (owner := class_spy(cls)) is None:
            msg = f"type object {cls.__name__!r} has no attribute {attr!r}"
            raise AttributeError(msg)

        out = SpyObject()
        return owner.__optype_trace_add__(Marker.CLASS_GETATTR, (attr,), {}, out)

    @override
    def __setattr__(cls, attr: str, value: object, /) -> None:
        # recorded without mutating, so forked reruns stay clean
        if _internal(attr) or (owner := class_spy(cls)) is None:
            return super().__setattr__(attr, value)

        args = attr, value
        return owner.__optype_trace_add__(Marker.CLASS_SETATTR, args, {}, None)

    @override
    def __delattr__(cls, attr: str, /) -> None:
        if _internal(attr) or (owner := class_spy(cls)) is None:
            return super().__delattr__(attr)

        return owner.__optype_trace_add__(Marker.CLASS_DELATTR, (attr,), {}, None)


class SpyObject(Spy, metaclass=_SpyType):
    __optype_element__: Self | None = None
    __optype_iterator__: bool = False
    __optype_growable__: bool = False
    __optype_absent__: frozenset[str] = frozenset()
    # see `_decide_stable` and `_decide_keyed`
    __optype_stable__: dict[str, _Memo]
    __optype_keyed__: dict[str, _KeyedMemo]
    # spies are descriptors (`__get__`), so only ever read through the class `__dict__`
    __optype_instance__: ClassVar[Self | None] = None

    def __init__(self, /, *_args: object, **_kwargs: object) -> None:
        super().__init__()
        self.__optype_stable__ = {}
        self.__optype_keyed__ = {}

    def __new__(cls, /, *_args: object, **_kwargs: object) -> Self:
        if cls is not SpyObject:
            # a `type(spy)(...)` sibling; the marker keeps it reachable from the spy
            self = super().__new__(cls)
            if (owner := class_spy(cls)) is not None:
                _journal_touch(owner)
                owner.__optype_trace__.append(TraceItem(Marker.SIBLING, (), {}, self))
            return self

        # every spy gets a class of its own, so that `type(spy)` identifies the spy
        unique: type[Any] = type("SpyObject", (cls,), {})
        self = super().__new__(unique)
        type.__setattr__(unique, "__optype_instance__", self)
        return self

    ###

    def __getattr__(self, attr: str, /) -> "SpyObject | None":
        # TODO: specialize for known special dunder attrs, e.g. `__name__: str`
        if _internal(attr):
            return None
        if dynamic_name(attr):
            raise DynamicNameError

        if attr in self.__optype_absent__:
            # the marker survives only if a completing run tolerates the absence
            self.__optype_trace_add__(Marker.ABSENT, ("__getattr__", attr), {}, None)
            raise AttributeError(attr)
        return self.__optype_trace_add__("__getattr__", (attr,), {}, SpyObject())

    @override
    def __setattr__(self, attr: str, value: object, /) -> None:
        if _internal(attr):
            return super().__setattr__(attr, value)

        return self.__optype_trace_add__("__setattr__", (attr, value), {}, None)

    @override
    def __delattr__(self, attr: str, /) -> None:
        self.__optype_trace_add__("__delattr__", (attr,), {}, None)

    @override
    def __dir__(self, /) -> "SpyObject":
        # TODO: maybe return specialized `SpyObject & Iterable[str]`
        return self.__optype_trace_add__("__dir__", (), {}, SpyObject())

    ###

    @override
    def __repr__(self, /) -> SpyStr:
        return self.__optype_trace_add__("__repr__", (), {}, SpyStr())

    @override
    def __str__(self, /) -> SpyStr:
        return self.__optype_trace_add__("__str__", (), {}, SpyStr())

    @override
    def __format__(self, format_spec: str, /) -> SpyStr:
        return self.__optype_trace_add__("__format__", (format_spec,), {}, SpyStr())

    def __bytes__(self, /) -> SpyBytes:
        return self.__optype_trace_add__("__bytes__", (), {}, SpyBytes())

    ###

    @override
    def __eq__(self, other: object, /) -> "SpyObject":  # type:ignore[override] # pyright:ignore[reportIncompatibleMethodOverride] # ty:ignore[invalid-method-override]
        return self.__optype_trace_add__("__eq__", (other,), {}, SpyObject())

    @override
    def __ne__(self, other: object, /) -> "SpyObject":  # type:ignore[override] # pyright:ignore[reportIncompatibleMethodOverride] # ty:ignore[invalid-method-override]
        return self.__optype_trace_add__("__ne__", (other,), {}, SpyObject())

    ###

    @override
    def __hash__(self, /) -> int:
        return self.__optype_trace_add__("__hash__", (), {}, super().__hash__())

    def __bool__(self, /) -> bool:
        return bool(_decide_stable(self, "__bool__"))

    ###

    def __get__(self, instance: object, owner: type | None = None) -> "SpyObject":
        return self.__optype_trace_add__("__get__", (instance, owner), {}, SpyObject())

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

    def __call__(self, /, *args: object, **kwargs: object) -> "SpyObject":
        return self.__optype_trace_add__("__call__", args, kwargs, SpyObject())

    ###

    def __len__(self, /) -> int:
        # `len()` may be probed optionally (e.g. `list()` via `length_hint`)
        return _decide_stable(self, "__len__", optional=True)

    # no need for `__length_hint__`

    def __getitem__(self, key: object, /) -> "SpyObject":
        return self.__optype_trace_add__("__getitem__", (key,), {}, SpyObject())

    def __setitem__(self, key: object, value: object, /) -> None:
        return self.__optype_trace_add__("__setitem__", (key, value), {}, None)

    def __delitem__(self, key: object, /) -> None:
        return self.__optype_trace_add__("__delitem__", (key,), {}, None)

    # no need for `__missing__`

    def __iter__(self, /) -> "SpyObject":
        growable = _iter_is_star_unpack()

        if self.__optype_iterator__:
            if growable:
                self.__optype_growable__ = True
            return self  # an iterator is its own iterable (idempotent `iter()`)

        out = _iterator_of(self)
        out.__optype_growable__ = growable
        return self.__optype_trace_add__("__iter__", (), {}, out)

    def __reversed__(self, /) -> "SpyObject":
        return self.__optype_trace_add__("__reversed__", (), {}, _iterator_of(self))

    def __contains__(self, item: object, /) -> bool:
        return _decide_keyed(self, "__contains__", item, keep_arg=True)

    # return `Any` instead of `SpyObject` to avoid an LSP error for `__dir__`
    def __next__(self, /) -> Any:
        # count from the trace, not a field, so a forked run's rollback is reflected
        served = sum(1 for item in self.__optype_trace__ if item.attr == "__next__")
        growable = self.__optype_growable__
        limit = yield_budget.get() if growable else _DEFAULT_YIELD
        if served >= limit:
            if growable:
                starved.set(True)
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

    def __enter__(self, /) -> "SpyObject":
        return self.__optype_trace_add__("__enter__", (), {}, SpyObject())

    def __exit__(self, /, *args: object) -> None:
        # TODO: maybe fork and return falsy/truthy in case of exception??
        return self.__optype_trace_add__("__exit__", args, {}, None)

    ###

    # no `__release_buffer__`: its slot lookup fails uncatchably under cyclic GC (#739)
    def __buffer__(self, flags: int, /) -> memoryview:
        return self.__optype_trace_add__("__buffer__", (flags,), {}, memoryview(b""))

    ###

    # numpy looks these up on the type, so `__getattr__` won't do
    def __array_ufunc__(
        self,
        ufunc: AnyFunc,
        method: str,
        /,
        *inputs: object,
        **kwargs: object,
    ) -> "SpyObject":
        return self.__optype_trace_add__("__array_ufunc__", (ufunc,), {}, SpyObject())

    def __array_function__(
        self,
        func: AnyFunc,
        types: object,
        args: object,
        kwargs: object,
        /,
    ) -> "SpyObject":
        return self.__optype_trace_add__(
            "__array_function__",
            (func,),
            {},
            SpyObject(),
        )

    ###

    def __await__(self, /) -> Generator[Any, None, "SpyObject"]:
        out = self.__optype_trace_add__("__await__", (), {}, SpyObject())

        def spy_generator() -> Generator[Any, None, "SpyObject"]:
            yield from ()
            return out  # ruff: ignore[return-in-generator]

        return spy_generator()

    def __aiter__(self, /) -> "SpyObject":
        return self.__optype_trace_add__("__aiter__", (), {}, SpyObject())

    def __anext__(self, /) -> "SpyObject":
        if any(item.attr == "__anext__" for item in self.__optype_trace__):
            raise StopAsyncIteration
        return self.__optype_trace_add__("__anext__", (), {}, SpyObject())

    def __aenter__(self, /) -> "SpyObject":
        return self.__optype_trace_add__("__aenter__", (), {}, SpyObject())

    def __aexit__(self, /, *args: object) -> "SpyObject":
        return self.__optype_trace_add__("__aexit__", args, {}, SpyObject())


# the observable class names for `dynamic_name`; not the broader `Spy`, so a
# genuine `*_spy*` attribute never matches
_SPY_NAMES = tuple(
    cls.__name__.lower() for cls in (SpyStr, SpyBytes, _SpyType, SpyObject)
)

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


def _traced_op(name: str) -> AnyFunc:
    def op(self: SpyObject, /, *args: object) -> SpyObject:
        return self.__optype_trace_add__(name, args, {}, SpyObject())

    op.__name__ = op.__qualname__ = name
    return op


for _name in _TRACED_OPS:
    # bypass the metaclass: `class_spy` isn't defined yet, and this isn't a trace
    type.__setattr__(SpyObject, _name, _traced_op(_name))  # ruff: ignore[unnecessary-dunder-call]


# Free functions, not methods: a method would be an unrecorded hole in the proxy.
def class_spy(cls: object) -> SpyObject | None:
    """The spy whose unique class `cls` is, if any."""
    # the marker must point back into the mro, or it is a copy on a foreign class; the
    # exact-metaclass gate is because a foreign `__dict__` can be a metaclass property
    if type(cls) is _SpyType:
        spy = cls.__dict__.get("__optype_instance__")
        if isinstance(spy, SpyObject) and issubclass(cls, type(spy)):
            return spy
    return None


def own_spy(spy: SpyObject) -> SpyObject:
    """The first spy of `spy`'s class: `type(spy)()` siblings collapse onto it."""
    owner = class_spy(type(spy))  # `or spy` would trace a `__bool__` on the owner
    return spy if owner is None else owner


def despy_class(cls: type, /) -> type:
    """The first non-spy base of `cls`, so internal spy classes never render."""
    return next(c for c in cls.__mro__ if c.__module__ != __name__)


def isinstance_not_spy[T](
    obj: object,
    cls_or_tuple: type[T] | tuple[type[T], ...],
    /,
) -> TypeGuard[T]:
    return isinstance(obj, cls_or_tuple) and not isinstance(obj, Spy)


def as_spy(value: object) -> SpyObject | None:
    """The (first-of-its-class) spy itself, or the spy whose class it is, if any."""
    # a `weakref.proxy` forwards `__class__`, so verify its real class is a spy's
    if isinstance(value, SpyObject) and (owner := class_spy(type(value))) is not None:
        return owner
    return class_spy(value)


def _element_of(spy: SpyObject) -> SpyObject:
    if (element := spy.__optype_element__) is None:
        element = SpyObject()
        spy.__optype_element__ = element  # ty:ignore[invalid-assignment]
    return element


def _iterator_of(spy: SpyObject) -> SpyObject:
    iterator = SpyObject()
    iterator.__optype_element__ = _element_of(spy)  # ty:ignore[invalid-assignment]
    iterator.__optype_iterator__ = True
    return iterator
