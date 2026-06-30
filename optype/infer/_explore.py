"""Run a function against spy placeholders and record what happens."""

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
from inspect import Parameter, isasyncgen, iscoroutine, isgenerator, signature
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    MethodType,
    WrapperDescriptorType,
)
from typing import Any, cast

from ._errors import InferError
from ._spy import (
    _AbsentError,
    _AnyFunc,
    _Fork,
    _fork,
    _journal,
    _Marker,
    _own_spy,
    _Spy,
    _SpyBytes,
    _SpyObject,
    _SpyStr,
    _starved,
    _TraceItem,
    _Traces,
    _yield_budget,
    as_spy,
    journal_rollback,
    set_driver_code,
)
from ._values import (
    COROUTINE,
    VARIADIC_KINDS,
    Exploration,
    GapKind,
    _Fn,
    _Gen,
    _Rec,
    _RecRef,
    _RecVar,
    fn_spies,
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

# single-arg lazy iterators, mapped to their rendered name; itertools entries are
# derived from the module, so new ones register automatically
_ITERATOR_TYPES: dict[type, str] = (
    {  # type: ignore[assignment]
        enumerate: "enumerate",
        filter: "filter",
        map: "map",
        zip: "zip",
    }
    | {
        cls: f"itertools.{cls.__qualname__}"
        for name, cls in vars(itertools).items()
        if isinstance(cls, type)
        and issubclass(cls, Iterator)
        and not name.startswith("_")
        and cls is not itertools.groupby  # 2 type args
    }
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


def _reachable(params: Iterable[object]) -> Generator[_SpyObject]:
    # every spy reachable from `params` through the recorded operations
    seen: set[int] = set()
    stack = [spy for spy in params if isinstance(spy, _SpyObject)]
    while stack:
        if id(spy := stack.pop()) in seen:
            continue
        seen.add(id(spy))
        yield spy
        stack.extend(
            ret
            for item in spy.__optype_trace__
            if isinstance(ret := item.return_, _SpyObject)
        )


def _snapshot(params: Iterable[_SpyObject]) -> _Traces:
    """Capture the traces of every spy reachable from `params`.

    An operation on a `type(spy)(...)` sibling requires it of the spy's type, so a
    sibling's trace merges into its owner's, and the markers themselves are dropped.
    """
    traces: _Traces = {}
    for spy in _reachable(params):
        items = (item for item in spy.__optype_trace__ if item.attr != _Marker.SIBLING)
        traces.setdefault(id(_own_spy(spy)), []).extend(items)
    return traces


def _parameters(func: _AnyFunc) -> Mapping[str, Parameter]:
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
        return _SpyStr(value)
    if isinstance(value, bytes):
        return _SpyBytes(value)
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
    if isinstance(value, _SpyObject):
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
    # iterate an async generator synchronously by resolving each step's awaitable
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


def _explore_func(func: _AnyFunc) -> object:
    """Explore a returned function, so it renders in signature syntax."""
    if _explore_key(func) in _exploring.get():
        return func  # a recursive function type is inexpressible
    try:
        params = _parameters(func)
        if any(p.kind in VARIADIC_KINDS for p in params.values()):
            return func  # variadic parameters are not expressible (yet)
        exploration, _ = explore_lenient(func, params)
    except Exception:  # noqa: BLE001  # an unexplorable function stays opaque
        return func
    return _Fn(
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
    args = getattr(result, "args", (_SpyObject(),))
    return fn(*args, **getattr(result, "keywords", {}))


def _wrapper(
    cls: type,
    name: str,
    result: object,
    path: dict[int, _RecVar | None],
) -> object:
    """A generic `functools` wrapper, parameterized by the wrapped return type."""

    # a `partial` of a real function keeps its richer call signature; exploring a
    # spy-wrapped one would pollute the spy with signature probes
    if (
        cls is functools.partial
        and isinstance(_unwrap(result), _FUNCTION_TYPES)
        and isinstance(explored := _explore_func(cast("_AnyFunc", result)), _Fn)
    ):
        return explored

    ret = _wrapped_return(result)
    return _Gen([] if ret is None else [_next(ret, path)], name)


def _source_element(result: object) -> _SpyObject | None:
    # the shared element spy of a predicate filter's wrapped source iterator
    for ref in gc.get_referents(result):
        if isinstance(ref, _SpyObject) and ref.__optype_iterator__:
            return ref.__optype_element__
    return None


def _next(result: object, path: dict[int, _RecVar | None] | None = None) -> object:  # noqa: C901
    # a function (or iterator) within the yields or a container is explored as well
    path = {} if path is None else path
    rid = id(result)
    if rid in path:
        # reuse this ancestor's binder, or create it on the first back-edge
        path[rid] = var = path[rid] or _RecVar(object())
        return _RecRef(var)
    path[rid] = None  # marks the ancestor chain; a `_RecVar` once reached again
    cls = type(result)
    if isgenerator(result):
        out = _Gen([_next(v, path) for v in _yields(result)], "Generator")
    elif isasyncgen(result):
        out = _Gen([_next(v, path) for v in _yields(_sync(result))], "AsyncGenerator")
    elif isinstance(result, Coroutine):
        # a returned coroutine value (e.g. 2-arg `anext`'s `anext_awaitable`)
        out = _Gen([_next(_await(result), path)], COROUTINE)
    elif (kind := _ITERATOR_TYPES.get(cls)) is not None:
        values = _yields(cast("Iterable[Any]", result))
        if cls is enumerate:
            # `enumerate[R]` is parameterized by the element type, not the yields
            values = [item for _, item in values]
        elif not values and cls in _FILTER_TYPES:
            # the predicate dropped every element; the element type is the source's
            element = _source_element(result)
            values = [element] if element is not None else values
        out = _Gen([_next(v, path) for v in values], kind)
    elif (
        cls is _CALLABLE_ITERATOR
        and _CALLABLE_FIRST
        and len(refs := gc.get_referents(result)) == 2
        and (fn := as_spy(refs[0])) is not None
    ):
        # `iter(callable, sentinel)`: call the callable for the element type; iterating
        # would stop at the sentinel and pollute it. Referent 0 is the callable (per
        # `_CALLABLE_FIRST`; not a spy-search, since the sentinel may be a spy too).
        # Calling `fn()` is deliberate: the recorded `__call__` is what renders the
        # parameter as `() -> R`, as `explore_spies` snapshots after this runs.
        out = _Gen([_next(fn(), path)], "Iterator")
    elif (name := _WRAPPER_TYPES.get(cls)) is not None:
        out = _wrapper(cls, name, result, path)
    elif (tpl := _TEMPLATE_TYPES.get(cls)) is not None:
        kind, attr = tpl
        yields = [] if attr is None else [_next(getattr(result, attr), path)]
        out = _Gen(yields, kind)
    elif isinstance(_unwrap(result), _FUNCTION_TYPES):
        out = _explore_func(cast("_AnyFunc", result))
    else:
        out = _next_container(cls, result, path)
    return _Rec(var, out) if (var := path.pop(rid)) is not None else out


def _next_container(
    cls: type[object],
    result: object,
    path: dict[int, _RecVar | None],
) -> object:
    # pyright fails miserably here when narrowing types
    match result:
        case tuple():
            return tuple(_next(item, path) for item in result)  # pyright:ignore[reportUnknownArgumentType,reportUnknownVariableType]
        case list():
            return [_next(item, path) for item in result]  # pyright:ignore[reportUnknownArgumentType,reportUnknownVariableType]
        case Mapping():
            # the keys must stay hashable, so only the values recurse
            return cls({key: _next(value, path) for key, value in result.items()})  # type:ignore[call-arg] # pyright:ignore[reportCallIssue,reportUnknownVariableType]
        case _:
            return result


def _with_next_default(
    func: _AnyFunc,
    spies: Mapping[str, _SpyObject],
    results: list[object],
) -> list[object]:
    # `next`/`anext` return `default` on an exhaustion branch the spies never reach
    if func not in _NEXT_BUILTINS or (default := spies.get("_1")) is None:
        return results

    # `anext` resolves through an awaitable, so the default unions inside the coroutine
    merged: list[object] = []
    awaitable = False
    for r in results:
        if isinstance(r, _Gen) and r.kind == COROUTINE:
            awaitable = True
            merged.append(_Gen([*r.yielded, default], COROUTINE))
        else:
            merged.append(r)

    # `next` returns the value directly, so the default joins as a sibling result
    return merged if awaitable else [*results, default]


@set_driver_code
def _run[T](
    func: Callable[..., T] | Callable[..., Coroutine[Any, None, T]],
    args: Iterable[object],
    kwds: Mapping[str, object],
) -> tuple[T, str | None]:
    """Call `func`, returning its (awaited) result and any deprecation message.

    A `DeprecationWarning` is recorded, not raised, so a `@deprecated` callable runs.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", category=DeprecationWarning)
        result = func(*args, **kwds)
        value = _await(result) if iscoroutine(result) else cast("T", result)

    message = next(
        (str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)),
        None,
    )
    return value, message


def _explore[T](  # noqa: C901
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
        # `_starved` is a per-run star-unpack flag, so no prior run leaks into this one
        _starved.set(False)
        fork_token = _fork.set(iter(plan))
        # a rejected run rolls back its trace appends, including on closed-over spies
        marks: dict[int, tuple[_Spy, int]] = {}
        journal_token = _journal.set(marks)
        undo = False

        try:
            result, message = _run(func, args, kwds)
            results.append(result)
            deprecated = deprecated or message
        except _Fork:
            if len(plan) < _FORK_LIMIT:
                stack.extend(([*plan, False], [*plan, True]))
            else:
                dropped = True
        except _AbsentError:
            # the dunder is genuinely needed, so this run (and its marker) never was
            undo = True
        except (InferError, IndexError, KeyError, TypeError):
            raise  # signals the driver acts on, not a rejected run
        except ValueError as exc:
            # a forked value the target rejected (e.g. `range`'s zero step); defer
            value_exc = exc
            undo = True
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            # the target rejected these spy values or exited (e.g. `exit()`); skip
            last_exc = exc
            undo = True
        finally:
            _fork.reset(fork_token)
            _journal.reset(journal_token)
            journal_rollback(marks, undo=undo)

    if not results:
        if value_exc is not None:
            raise value_exc

        raise InferError("the function never ran to completion") from last_exc

    hits = (dropped, GapKind.BRANCH_BUDGET), (bool(stack), GapKind.RUN_BUDGET)
    gaps = frozenset(kind for hit, kind in hits if hit)
    return results, deprecated, gaps


def _fixed_self(func: _AnyFunc, params: Mapping[str, Parameter]) -> dict[str, object]:
    if (
        not isinstance(func, (MethodDescriptorType, WrapperDescriptorType))
        or not params
    ):
        return {}
    cls = func.__objclass__
    # a spy argument satisfies constructors that need a buffer/index/iterable/...
    for args in ((), (_SpyObject(),)):
        with suppress(Exception):
            return {next(iter(params)): cls(*args)}
    msg = f"cannot instantiate {cls.__name__!r} for {func.__qualname__!r}"
    raise InferError(msg)


def _placeholders(
    params: Mapping[str, Parameter],
    count: int,
    keys: Sequence[str],
    omit: Collection[str],
    fixed: Mapping[str, object],
) -> tuple[dict[str, _SpyObject], list[object], dict[str, object]]:
    # one spy per non-omitted, non-fixed parameter, distributed over the call's
    # args and kwds
    spies = {
        name: _SpyObject() for name in params if name not in omit and name not in fixed
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
                kwds |= dict.fromkeys(map(_SpyStr, keys or ("",)), value)
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
    spies: Mapping[str, _SpyObject],
    absent: Mapping[str, Collection[str]],
) -> None:
    for name, attrs in absent.items():
        if (spy := spies.get(name)) is not None:
            spy.__optype_absent__ = frozenset(attrs)


def explore_spies(
    func: _AnyFunc,
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

    yield_token = _yield_budget.set(budget)
    starve_token = _starved.set(False)
    try:
        while True:
            _yield_budget.set(budget)

            # a fresh `self` instance per attempt, so a mutated one cannot leak
            fixed = _fixed_self(func, params) | {
                n: _typed_default(params[n].default) for n in fix
            }
            spies, args, kwds = _placeholders(params, count, keys, omit, fixed)
            _force_absent(spies, forced_absent)
            try:
                results, deprecated, gaps = _explore(func, args, kwds)
                results = _with_next_default(func, spies, [_next(r) for r in results])
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
                    and _starved.get()
                    and (budget := next(budgets, 0)) != 0
                ):
                    pass
                elif Parameter.VAR_POSITIONAL in kinds:
                    if (count := next(counts, 0)) == 0:
                        msg = f"ran out of `*args` placeholders ({exc})"
                        raise InferError(msg) from exc
                else:
                    raise
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
        _starved.reset(starve_token)
        _yield_budget.reset(yield_token)
        _exploring.reset(token)


def explore_lenient(
    func: _AnyFunc,
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


def _op_shape(items: Iterable[_TraceItem]) -> frozenset[str]:
    return frozenset(item.attr for item in items if not isinstance(item.attr, _Marker))


# dunders `tuple` delegates to its elements (`repr`, `hash`, ...), not distribution
_TUPLE_DUNDERS = frozenset(
    name for klass in tuple.__mro__ for name in vars(klass) if name.startswith("__")
)


def explore_tuple_params(
    func: _AnyFunc,
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

        spies, args, kwds = _placeholders(params, 2, [], (), exploration.fixed)
        if (target := spies.get(name)) is None:
            continue
        elems = (_SpyObject(), _SpyObject())
        args = [elems if a is target else a for a in args]
        kwds = {key: elems if value is target else value for key, value in kwds.items()}
        try:
            # a bare run: results aren't re-explored, so no recursion guard is needed
            _explore(func, args, kwds)
        except Exception:  # noqa: BLE001, S112
            continue
        traces = _snapshot(elems)
        # an element skipped by short-circuit isn't traced; check only those that ran
        elem_shapes = [s for e in elems if (s := _op_shape(traces.get(id(e), ())))]
        if elem_shapes and all(s == bare_shape for s in elem_shapes):
            tuple_params.add(name)
    return frozenset(tuple_params)
