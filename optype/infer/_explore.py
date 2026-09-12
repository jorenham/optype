"""Run a function against spy placeholders and record what happens."""

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

import functools
import gc
import itertools
import sys
import warnings
from collections.abc import (
    AsyncGenerator,
    Callable,
    Collection,
    Coroutine,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import suppress
from contextvars import ContextVar
from inspect import Parameter, isasyncgen, iscoroutine, isgenerator
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    MethodType,
    WrapperDescriptorType,
)
from typing import Any

from ._errors import InferError
from ._gc import cyclic_gc
from ._signature import signature
from ._spy import (
    AbsentError,
    AnyFunc,
    DynamicNameError,
    Fork,
    Marker,
    Spy,
    SpyBytes,
    SpyObject,
    SpyStr,
    TraceItem,
    Traces,
    as_spy,
    fork_plan,
    journal,
    journal_rollback,
    own_spy,
    set_driver_code,
    starved,
    yield_budget,
)
from ._values import (
    COROUTINE,
    VARIADIC_KINDS,
    Exploration,
    FnResult,
    GapKind,
    Gen,
    Rec,
    RecRef,
    RecVar,
    fn_spies,
    is_mapping,
)

_FORK_LIMIT = 64
_RUN_LIMIT = 256
_YIELD_LIMIT = 64
_KWARGS_LIMIT = 8  # max injected `**kwargs` keys

# the `*args` placeholder counts to try: exact arities first, then doubling so that
# large indices stay within reach
_VARIADIC_COUNTS = (2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024)
# yield budgets to try for a star-unpack into a fixed-arity call (e.g. `divmod`): an
# exact arity is needed, so these stay contiguous; a sparse range would skip valid ones
_YIELD_COUNTS = tuple(range(2, 17))

# the `next`-like builtins, which return their trailing argument when exhausted
_NEXT_BUILTINS = frozenset({next, anext})

# the single-arg `itertools` iterators; with no yields they render bare, not `[Never]`
_ITERTOOLS_TYPES: dict[type, str] = {
    cls: f"itertools.{cls.__qualname__}"
    for name, cls in vars(itertools).items()
    if isinstance(cls, type)
    and issubclass(cls, Iterator)
    and not name.startswith("_")
    and cls is not itertools.groupby  # 2 type args
}

# single-arg lazy iterators, mapped to their rendered name
_ITERATOR_TYPES: dict[type, str] = (
    {  # pyrefly: ignore[implicit-any-type-argument]
        enumerate: "enumerate",
        filter: "filter",
        map: "map",
        zip: "zip",
    }
    | _ITERTOOLS_TYPES
    # typeshed types `tee()` as `tuple[Iterator[T], ...]`
    | {type(itertools.tee(())[0]): "Iterator"}
)

# predicate filters that preserve the element type; a stably-truthy spy predicate makes
# `dropwhile`/`filterfalse` drop every element, so the element comes from the source
_FILTER_TYPES = frozenset({
    filter,
    itertools.dropwhile,
    itertools.filterfalse,
    itertools.takewhile,
})

# the lazy iterator returned by the 2-argument `iter(callable, sentinel)`
_CALLABLE_ITERATOR = type(iter(int, None))

# generic `functools` wrappers, by rendered name. `singledispatchmethod` is absent:
# its `__init__` eagerly calls `singledispatch`, which a spy callable cannot satisfy
_WRAPPER_TYPES: dict[type, str] = {
    cls: f"functools.{cls.__qualname__}"
    for cls in (functools.partial, functools.partialmethod, functools.cached_property)
}

# `string.templatelib` types (3.14+): rendered name, and the single-type-arg attr
if sys.version_info >= (3, 14):
    from string.templatelib import (
        Interpolation as _Interpolation,
        Template as _Template,
    )

    _TEMPLATE_TYPES: dict[type, tuple[str, str | None]] = {
        _Template: ("string.templatelib.Template", None),
        _Interpolation: ("string.templatelib.Interpolation", "value"),
    }
else:
    _TEMPLATE_TYPES: dict[type, tuple[str, str | None]] = {}


def _reachable_spies(params: Iterable[object]) -> Generator[SpyObject]:
    seen: set[int] = set()
    stack = [spy for spy in params if isinstance(spy, SpyObject)]
    while stack:
        if id(spy := stack.pop()) in seen:
            continue
        seen.add(id(spy))
        yield spy
        stack.extend(
            ret
            for item in spy.__optype_trace__
            if isinstance(ret := item.return_, SpyObject)
        )


def _snapshot(params: Iterable[SpyObject]) -> Traces:
    """Capture the traces of every spy reachable from `params`.

    An operation on a `type(spy)(...)` sibling requires it of the spy's type, so a
    sibling's trace merges into its owner's, and the markers themselves are dropped.
    """
    traces: Traces = {}
    for spy in _reachable_spies(params):
        items = (item for item in spy.__optype_trace__ if item.attr != Marker.SIBLING)
        traces.setdefault(id(own_spy(spy)), []).extend(items)
    return traces


def _parameters(func: AnyFunc) -> Mapping[str, Parameter]:
    try:
        return signature(func).parameters
    except (TypeError, ValueError) as exc:  # not callable, or no signature
        raise InferError(str(exc)) from exc


def declared_defaults(params: Mapping[str, Parameter]) -> dict[str, object]:
    """The declared parameter defaults, by name."""
    return {n: p.default for n, p in params.items() if p.default is not Parameter.empty}


def _typed_default(value: object) -> object:
    """A rejected default's type is known, but its value is not, so widen it."""
    if isinstance(value, str):
        return SpyStr(value)
    if isinstance(value, bytes):
        return SpyBytes(value)
    return value


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
    if isinstance(value, SpyObject):
        return ("spy", *(op.attr for op in value.__optype_trace__))
    return ("val", type(value).__name__)


def _yields[T](values: Iterable[T]) -> list[T]:
    seen: set[tuple[str, ...]] = set()
    out: list[T] = []
    for value in itertools.islice(values, _YIELD_LIMIT):
        if (key := _yield_key(value)) in seen:
            break
        seen.add(key)
        out.append(value)
    return out


def _sync[T](agen: AsyncGenerator[T, Any]) -> Generator[T]:
    for _ in range(_YIELD_LIMIT):
        try:
            yield _await(anext(agen))
        except StopAsyncIteration:
            return


def _ref0_is_callable() -> bool:
    # `iter(callable, sentinel)` keeps the callable as its first gc referent on CPython
    def probe() -> None: ...

    refs = gc.get_referents(iter(probe, object()))
    return len(refs) == 2 and refs[0] is probe


_CALLABLE_FIRST = _ref0_is_callable()

_FUNCTION_TYPES = (
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    MethodDescriptorType,
    WrapperDescriptorType,
)

# the (unwrapped) functions currently being explored, see `_explore_key`
_exploring: ContextVar[frozenset[int]] = ContextVar("_exploring", default=frozenset())


def _unwrap(obj: object) -> object:
    """The underlying callable of (nested) `functools.partial` wrappers."""
    while isinstance(obj, functools.partial):
        obj = obj.func
    return obj


def _explore_key(func: object) -> int:
    # closures from a single def share their code object, so a recursive function
    # factory is recognized even though it returns a fresh closure on every call
    base = _unwrap(func)
    return id(getattr(base, "__code__", base))


def _explore_func(func: AnyFunc) -> object:
    """Explore a returned function, so it renders in signature syntax."""
    if _explore_key(func) in _exploring.get():
        return func  # a recursive function type is inexpressible
    try:
        params = _parameters(func)
        if any(p.kind in VARIADIC_KINDS for p in params.values()):
            return func  # variadic parameters are not expressible (yet)
        exploration, _ = explore_lenient(func, params)
    except Exception:  # ruff: ignore[blind-except]  # an unexplorable function stays opaque
        return func
    return FnResult(
        params,
        exploration.spies,
        exploration.fixed,
        declared_defaults(params),
        exploration.results,
    )


def _wrapped_return(result: object) -> object | None:
    # only a spy is safe to call; a real callable (e.g. `print`) would actually run
    if (fn := as_spy(getattr(result, "func", None))) is None:
        return None

    # `cached_property` has no `.args`: its getter binds the instance
    args = getattr(result, "args", (SpyObject(),))
    return fn(*args, **getattr(result, "keywords", {}))


def _wrapper(
    cls: type,
    name: str,
    result: Any,
    path: dict[int, RecVar | None],
) -> object:
    """A generic `functools` wrapper, parameterized by the wrapped return type."""

    # a `partial` of a real function keeps its richer call signature; exploring a
    # spy-wrapped one would pollute the spy with signature probes
    if (
        cls is functools.partial
        and isinstance(_unwrap(result), _FUNCTION_TYPES)
        and isinstance(explored := _explore_func(result), FnResult)
    ):
        return explored

    ret = _wrapped_return(result)
    yields = [] if ret is None else [_explore_result(ret, path)]
    return Gen(yields, name, bare_when_empty=True)


def _source_element(result: object) -> SpyObject | None:
    for ref in gc.get_referents(result):
        if isinstance(ref, SpyObject) and ref.__optype_iterator__:
            return ref.__optype_element__
    return None


def _explore_result(  # ruff: ignore[complex-structure]
    result: Any,
    path: dict[int, RecVar | None] | None = None,
) -> object:
    # a function (or iterator) within the yields or a container is explored as well
    path = {} if path is None else path
    rid = id(result)
    if rid in path:
        # reuse this ancestor's binder, or create it on the first back-edge
        path[rid] = var = path[rid] or RecVar(object())
        return RecRef(var)
    path[rid] = None  # marks the ancestor chain; a `RecVar` once reached again
    cls = type(result)
    if isgenerator(result):
        out = Gen([_explore_result(v, path) for v in _yields(result)], "Generator")
    elif isasyncgen(result):
        out = Gen(
            [_explore_result(v, path) for v in _yields(_sync(result))],
            "AsyncGenerator",
        )
    elif isinstance(result, Coroutine):
        # a returned coroutine value (e.g. 2-arg `anext`'s `anext_awaitable`)
        out = Gen([_explore_result(_await(result), path)], COROUTINE)
    elif (kind := _ITERATOR_TYPES.get(cls)) is not None:
        values = _yields(result)
        if cls is enumerate:
            # `enumerate[R]` is parameterized by the element type, not the yields
            values = [item for _, item in values]
        elif not values and cls in _FILTER_TYPES:
            # the predicate dropped every element; the element type is the source's
            element = _source_element(result)
            values = [element] if element is not None else values
        out = Gen(
            [_explore_result(v, path) for v in values],
            kind,
            bare_when_empty=cls in _ITERTOOLS_TYPES,
        )
    elif (
        cls is _CALLABLE_ITERATOR
        and _CALLABLE_FIRST
        and len(refs := gc.get_referents(result)) == 2
        and (fn := as_spy(refs[0])) is not None
    ):
        # `iter(callable, sentinel)`: referent 0 is the callable, not a spy-search,
        # since the sentinel may be a spy too. Calling it is deliberate: the recorded
        # `__call__` is what renders the parameter as `() -> R`.
        out = Gen([_explore_result(fn(), path)], "Iterator")
    elif (name := _WRAPPER_TYPES.get(cls)) is not None:
        out = _wrapper(cls, name, result, path)
    elif (tpl := _TEMPLATE_TYPES.get(cls)) is not None:
        kind, attr = tpl
        yields = [] if attr is None else [_explore_result(getattr(result, attr), path)]
        out = Gen(yields, kind, bare_when_empty=True)
    elif isinstance(_unwrap(result), _FUNCTION_TYPES):
        out = _explore_func(result)
    else:
        out = _explore_container(cls, result, path)
    return Rec(var, out) if (var := path.pop(rid)) is not None else out


def _explore_container(cls: type, result: Any, path: dict[int, RecVar | None]) -> Any:
    match result:
        case tuple():
            return tuple(_explore_result(item, path) for item in result)
        case list():
            return [_explore_result(item, path) for item in result]
        case _ if is_mapping(result):
            # the keys must stay hashable, so only the values recurse
            items = {key: _explore_result(value, path) for key, value in result.items()}
            try:
                return cls(items)
            except TypeError:
                return result
        case _:
            return result


def _with_next_default(
    func: AnyFunc,
    spies: Mapping[str, SpyObject],
    results: Sequence[object],
) -> Sequence[object]:
    # `next`/`anext` return `default` on an exhaustion branch the spies never reach
    if func not in _NEXT_BUILTINS or len(spies) < 2:
        return results
    default = list(spies.values())[1]

    # `anext` resolves through an awaitable, so the default unions inside the coroutine
    merged: list[object] = []
    awaitable = False
    for r in results:
        if isinstance(r, Gen) and r.kind == COROUTINE:
            awaitable = True
            merged.append(Gen([*r.yielded, default], COROUTINE))
        else:
            merged.append(r)

    # `next` returns the value directly, so the default joins as a sibling result
    return merged if awaitable else [*results, default]


@set_driver_code
def _run(
    func: Callable[..., Any],
    args: Iterable[object],
    kwds: Mapping[str, object],
) -> tuple[Any, str | None]:
    """Call `func`, returning its (awaited) result and any deprecation message.

    A `DeprecationWarning` is recorded, not raised, so a `@deprecated` callable runs.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", category=DeprecationWarning)
        result = func(*args, **kwds)
        value = _await(result) if iscoroutine(result) else result

    message = next(
        (str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)),
        None,
    )
    return value, message


def _scrub_deprecated(message: str | None, func: AnyFunc) -> str | None:
    """Replace a leaked spy identity in `message` with the target's owner (#777)."""
    name = SpyObject.__name__
    if message is None or name not in message:
        return message

    qualname: str = getattr(func, "__qualname__", "")
    owner = qualname.rpartition(".")[0]
    module: str = getattr(func, "__module__", None) or ""
    full = f"{module}.{owner}" if module and owner else owner
    return message.replace(f"{SpyObject.__module__}.{name}", full).replace(name, owner)


def _drain_untraced() -> None:
    # the drain runs the target's finalizers; discard any spy ops they record,
    # or a rolled-back branch's `__del__` pollutes the trace
    marks: dict[int, tuple[Spy, int]] = {}
    token = journal.set(marks)
    try:
        cyclic_gc.drain()
    finally:
        journal.reset(token)
        journal_rollback(marks)


def _explore[T](  # ruff: ignore[complex-structure, too-many-branches]
    func: Callable[..., T] | Callable[..., Coroutine[Any, None, T]],
    args: Sequence[object],
    kwds: Mapping[str, object],
) -> tuple[list[T], str | None, frozenset[GapKind]]:
    results: list[T] = []
    deprecated: str | None = None
    stack: list[list[bool]] = [[]]
    dropped = False

    last_exc: BaseException | None = None
    value_exc: ValueError | None = None

    for _ in range(_RUN_LIMIT):  # caps the exponential blowup of independent forks
        if not stack:
            break
        plan = stack.pop()
        # `starved` is a per-run star-unpack flag, so no prior run leaks into this one
        starved.set(False)
        fork_token = fork_plan.set(iter(plan))
        # a rejected run rolls back its trace appends, including on closed-over spies
        marks: dict[int, tuple[Spy, int]] = {}
        journal_token = journal.set(marks)
        undo = False

        try:
            result, message = _run(func, args, kwds)
            results.append(result)
            deprecated = deprecated or _scrub_deprecated(message, func)
        except Fork:
            if len(plan) < _FORK_LIMIT:
                stack.extend(([*plan, False], [*plan, True]))
            else:
                dropped = True
        except AbsentError:
            # the dunder is genuinely needed, so this run (and its marker) never was
            undo = True
        except (InferError, IndexError, KeyError, TypeError):
            raise  # signals the driver acts on, not a rejected run
        except ValueError as exc:
            # a forked value the target rejected (e.g. `range`'s zero step); defer
            value_exc = exc
            undo = True
        except (Exception, SystemExit) as exc:  # ruff: ignore[blind-except]
            # the target rejected these spy values or exited (e.g. `exit()`); skip
            last_exc = exc
            undo = True
        finally:
            fork_plan.reset(fork_token)
            journal.reset(journal_token)
            if undo:
                journal_rollback(marks)

        _drain_untraced()

    if not results:
        if value_exc is not None:
            raise value_exc

        msg = (
            str(last_exc)
            if isinstance(last_exc, DynamicNameError)
            else "the function never ran to completion"
        )
        raise InferError(msg) from last_exc

    hits = (dropped, GapKind.BRANCH_BUDGET), (bool(stack), GapKind.RUN_BUDGET)
    gaps = frozenset(kind for hit, kind in hits if hit)
    return results, deprecated, gaps


def _fixed_self(func: AnyFunc, params: Mapping[str, Parameter]) -> dict[str, object]:
    if (
        not isinstance(func, (MethodDescriptorType, WrapperDescriptorType))
        or not params
    ):
        return {}
    cls = func.__objclass__
    # a spy argument satisfies constructors that need a buffer/index/iterable/...
    for args in ((), (SpyObject(),)):
        with suppress(Exception):
            return {next(iter(params)): cls(*args)}
    msg = f"cannot instantiate {cls.__name__!r} for {func.__qualname__!r}"
    raise InferError(msg)


def _placeholders(
    params: Mapping[str, Parameter],
    *,
    count: int,
    keys: Sequence[str],
    omit: Collection[str],
    fixed: Mapping[str, object],
) -> tuple[dict[str, SpyObject], list[object], dict[str, object]]:
    # one spy per non-omitted, non-fixed parameter, distributed over the call's
    # args and kwds
    spies = {
        name: SpyObject() for name in params if name not in omit and name not in fixed
    }
    args: list[object] = []
    kwds: dict[str, object] = {}
    gap = False  # a positional parameter after an omitted one must pass by keyword
    for name, param in params.items():
        if name in omit:
            gap = gap or param.kind is not Parameter.KEYWORD_ONLY
            continue
        value = fixed[name] if name in fixed else spies[name]
        match param.kind:
            case Parameter.VAR_POSITIONAL:
                args += [value] * count
            case Parameter.VAR_KEYWORD:
                kwds |= dict.fromkeys(map(SpyStr, keys or ("",)), value)
            case Parameter.KEYWORD_ONLY:
                kwds[name] = value
            case Parameter.POSITIONAL_ONLY if gap:
                msg = f"cannot pass {name!r} by keyword"
                raise InferError(msg)
            case _ if gap:
                kwds[name] = value
            case _:
                args.append(value)
    return spies, args, kwds


def _force_absent(
    spies: Mapping[str, SpyObject],
    absent: Mapping[str, Collection[str]],
) -> None:
    for name, attrs in absent.items():
        if (spy := spies.get(name)) is not None:
            spy.__optype_absent__ = frozenset(attrs)


def explore_spies(
    func: AnyFunc,
    params: Mapping[str, Parameter],
    omit: Collection[str] = (),
    fix: Collection[str] = (),
    absent: Mapping[str, Collection[str]] | None = None,
) -> Exploration:
    kinds = {p.kind for p in params.values()}
    forced_absent = absent or {}

    counts = iter(_VARIADIC_COUNTS)
    count = next(counts)

    # rerun with new spies when the variadic placeholders/iterator yield budget runs out
    budgets = iter(_YIELD_COUNTS)
    budget = 1

    keys: list[str] = []

    # registering `func` itself keeps a returned self-reference from recursing
    token = _exploring.set(_exploring.get() | {_explore_key(func)})

    yield_token = yield_budget.set(budget)
    starve_token = starved.set(False)
    try:
        while True:
            yield_budget.set(budget)

            # a fresh `self` instance per attempt, so a mutated one cannot leak
            fixed = _fixed_self(func, params) | {
                n: _typed_default(params[n].default) for n in fix
            }
            spies, args, kwds = _placeholders(
                params,
                count=count,
                keys=keys,
                omit=omit,
                fixed=fixed,
            )
            _force_absent(spies, forced_absent)
            try:
                results, deprecated, gaps = _explore(func, args, kwds)
                results = _with_next_default(
                    func,
                    spies,
                    [_explore_result(r) for r in results],
                )
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
                # a too-short star-unpack raises `TypeError`; gate on it so the target's
                # own error (e.g. a `ValueError`) can't churn the budget and bury itself
                if (
                    isinstance(exc, TypeError)
                    and starved.get()
                    and (budget := next(budgets, 0))
                ):
                    continue
                if Parameter.VAR_POSITIONAL not in kinds:
                    raise
                if not (count := next(counts, 0)):
                    msg = f"ran out of `*args` placeholders ({exc})"
                    raise InferError(msg) from exc
            else:
                return Exploration(
                    spies,
                    _snapshot((*spies.values(), *fn_spies(results))),
                    results,
                    count,
                    fixed,
                    deprecated,
                    gaps,
                )
    finally:
        starved.reset(starve_token)
        yield_budget.reset(yield_token)
        _exploring.reset(token)


def explore_lenient(
    func: AnyFunc,
    params: Mapping[str, Parameter],
) -> tuple[Exploration, Mapping[str, object]]:
    # if a spy placeholder is rejected, fall back to fixing the defaulted parameters,
    # and also return every parameter default for rendering
    try:
        return explore_spies(func, params), {}
    except (TypeError, ValueError):
        defaults = declared_defaults(params)
        if not defaults:
            raise
    # start from the all-fixed baseline and greedily promote one spy at a time
    fix = set(defaults)
    exploration = explore_spies(func, params, fix=fix)
    for name in defaults:
        with suppress(Exception):
            exploration = explore_spies(func, params, fix=fix - {name})
            fix.discard(name)
    # a still-fixed default widens to its type; a promoted one keeps its literal
    return exploration, {
        name: exploration.fixed.get(name, value) for name, value in defaults.items()
    }


def _op_shape(items: Iterable[TraceItem]) -> frozenset[str]:
    return frozenset(item.attr for item in items if not isinstance(item.attr, Marker))


# dunders `tuple` delegates to its elements (`repr`, `hash`, ...), not distribution
_TUPLE_DUNDERS = frozenset(
    name for klass in tuple.__mro__ for name in vars(klass) if name.startswith("__")
)


def explore_tuple_params(
    func: AnyFunc,
    params: Mapping[str, Parameter],
    exploration: Exploration,
) -> frozenset[str]:
    """The parameters that also accept a homogeneous `tuple[<bound>, ...]`.

    `isinstance`'s tuple recursion is a C-level check no spy sees, so each parameter is
    re-explored as a real tuple of spies: it distributes when its elements get its ops.
    """
    tuple_params: set[str] = set()
    for name, spy in exploration.spies.items():
        param = params.get(name)
        if param is None or param.kind in VARIADIC_KINDS:
            continue
        bare_shape = _op_shape(exploration.traces.get(id(spy), ()))
        if not bare_shape or not bare_shape.isdisjoint(_TUPLE_DUNDERS):
            continue

        spies, args, kwds = _placeholders(
            params,
            count=2,
            keys=(),
            omit=(),
            fixed=exploration.fixed,
        )
        if (target := spies.get(name)) is None:
            continue
        elems = (SpyObject(), SpyObject())
        args = [elems if a is target else a for a in args]
        kwds = {key: elems if value is target else value for key, value in kwds.items()}
        try:
            # a bare run: results aren't re-explored, so no recursion guard is needed
            _explore(func, args, kwds)
        except Exception:  # ruff: ignore[blind-except, try-except-continue]
            continue
        traces = _snapshot(elems)
        # an element skipped by short-circuit isn't traced; check only those that ran
        elem_shapes = [s for e in elems if (s := _op_shape(traces.get(id(e), ())))]
        if elem_shapes and all(s == bare_shape for s in elem_shapes):
            tuple_params.add(name)
    return frozenset(tuple_params)
