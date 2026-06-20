"""Run a function against spy placeholders and record what happens."""

import gc
import keyword
import re
import warnings
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
from contextlib import suppress
from contextvars import ContextVar
from functools import partial
from inspect import Parameter, isasyncgen, iscoroutine, isgenerator, signature
from itertools import islice
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    MethodType,
    WrapperDescriptorType,
)
from typing import Any, NamedTuple, cast

from ._errors import InferError, InferWarning
from ._spy import (
    _AbsentError,
    _AnyFunc,
    _Fork,
    _fork,
    _Marker,
    _own_spy,
    _SpyObject,
    _SpyStr,
    _starved,
    _Traces,
    _yield_budget,
    as_spy,
)
from ._values import _Fn, _Gen, _Rec, _RecRef, _RecVar, _walk, fn_spies

_RE_DOC_SIGNATURE = re.compile(r"\b(\w+)\(([^)]*)\)")
_RE_DOC_PARAM = re.compile(r"(?:^|,)\s*\**([a-zA-Z_]\w*)")
_RE_DOC_ARROW = re.compile(r"\s*->")

_FORK_LIMIT = 64
_RUN_LIMIT = 256
_YIELD_LIMIT = 64
# the `*args` placeholder counts to try: exact arities first, then doubling so that
# large indices stay within reach
_VARIADIC_COUNTS = (2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024)
_KWARGS_LIMIT = 8  # max injected `**kwargs` keys
# yield budgets to try for a splat into a fixed-arity call (e.g. `divmod`): an exact
# arity is needed, so these stay contiguous; a sparse range would skip valid arities
_YIELD_COUNTS = tuple(range(2, 17))


class _Exploration(NamedTuple):
    """What one exploration of a function against spy placeholders produced."""

    spies: Mapping[str, _SpyObject]
    traces: _Traces
    results: list[object]
    var_count: int  # the `*args` placeholder count
    fixed: Mapping[str, object]  # parameters passed as-is, not spies
    deprecated: str | None = None  # a `DeprecationWarning` message raised when called


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


def _doc_param_names(group: str) -> list[str] | None:
    params = group.replace("[", "").replace("]", "")
    names = _RE_DOC_PARAM.findall(params)
    # a usage example like `reduce(lambda x, y: ...)` yields keywords, not params
    if names and not any(map(keyword.iskeyword, names)):
        return names
    return None


def _doc_signatures(func: _AnyFunc) -> list[list[str]]:
    """Every documented call form's parameter names, by arity (one per arity).

    Arrow-annotated forms (`name(...) -> ret`) are the real overloads; a docstring's
    usage examples lack the arrow. When none are annotated (e.g. `next`, `slice`),
    fall back to the first form. Same-arity forms infer the same structure, so only
    the first of each arity is kept (its parameter names win).
    """
    name = getattr(func, "__name__", "")
    if not name:
        return []
    doc = func.__doc__ or ""
    arrow_forms: list[list[str]] = []
    first: list[str] | None = None
    for match in _RE_DOC_SIGNATURE.finditer(doc):
        if match[1] != name or (names := _doc_param_names(match[2])) is None:
            continue
        if first is None:
            first = names
        if _RE_DOC_ARROW.match(doc, match.end()):
            arrow_forms.append(names)
    forms = arrow_forms or ([first] if first is not None else [])
    by_arity: dict[int, list[str]] = {}
    for f in forms:
        by_arity.setdefault(len(f), f)
    return list(by_arity.values())


def _parameter_forms(func: _AnyFunc) -> list[Mapping[str, Parameter]]:
    # the parameters of every call form: the real signature's, else the docstring's
    try:
        return [signature(func).parameters]
    except TypeError as exc:  # not callable
        raise InferError(str(exc)) from exc
    except ValueError as exc:  # no signature
        if not (forms := _doc_signatures(func)):
            raise InferError(str(exc)) from exc
        return [
            {n: Parameter(n, Parameter.POSITIONAL_OR_KEYWORD) for n in names}
            for names in forms
        ]


def _parameters(func: _AnyFunc) -> Mapping[str, Parameter]:
    return _parameter_forms(func)[0]  # the first call form


def _declared_defaults(params: Mapping[str, Parameter]) -> dict[str, object]:
    """The declared parameter defaults, by name."""
    return {n: p.default for n, p in params.items() if p.default is not Parameter.empty}


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


# lazy builtin iterators with a single type argument
_ITERATOR_TYPES = frozenset({enumerate, filter, map, zip})

# the lazy iterator returned by the 2-argument `iter(callable, sentinel)`
_CALLABLE_ITERATOR = type(iter(int, None))


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
_VARIADIC_KINDS = frozenset({Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD})

# the (unwrapped) functions currently being explored, see `_explore_key`
_exploring: ContextVar[frozenset[int]] = ContextVar("_exploring", default=frozenset())


def _unwrap(obj: object) -> object:
    """The underlying callable of (nested) `functools.partial` wrappers."""
    while isinstance(obj, partial):
        obj = obj.func
    return obj


def _explore_key(func: object) -> int:
    # closures from a single def share their code object, so a recursive function
    # factory is recognized even though it returns a fresh closure on every call
    base = _unwrap(func)
    return id(getattr(base, "__code__", base))


def _closed_over(func: object, seen: set[int] | None = None) -> Generator[object]:
    # every value a function can reach besides its arguments, transitively
    seen = set() if seen is None else seen
    if id(func) in seen:
        return
    seen.add(id(func))
    values: list[object] = []
    while isinstance(func, partial):
        values += func.args
        values += func.keywords.values()
        func = func.func
    values += getattr(func, "__defaults__", None) or ()
    values += (getattr(func, "__kwdefaults__", None) or {}).values()
    for cell in getattr(func, "__closure__", None) or ():
        with suppress(ValueError):  # an unset cell
            values.append(cell.cell_contents)
    for value in values:
        for node in _walk(value):
            yield node
            if isinstance(_unwrap(node), _FUNCTION_TYPES):
                yield from _closed_over(node, seen)


def _explore_func(func: _AnyFunc) -> object:
    """Explore a returned function, so it renders in signature syntax."""
    if _explore_key(func) in _exploring.get():
        return func  # a recursive function type is inexpressible
    try:
        params = _parameters(func)
        if any(p.kind in _VARIADIC_KINDS for p in params.values()):
            return func  # variadic parameters are not expressible (yet)
        exploration, _ = _explore_lenient(func, params)
    except Exception:  # noqa: BLE001  # an unexplorable function stays opaque
        return func
    return _Fn(
        params,
        exploration.spies,
        exploration.fixed,
        _declared_defaults(params),
        exploration.results,
    )


def _next(result: object, path: dict[int, _RecVar | None] | None = None) -> object:
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
    elif cls in _ITERATOR_TYPES:
        values = _yields(cast("Iterable[object]", result))
        if cls is enumerate:
            # `enumerate[R]` is parameterized by the element type, not the yields
            values = [item for _, item in cast("list[tuple[int, object]]", values)]
        out = _Gen([_next(v, path) for v in values], cls.__name__)
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
        # parameter as `() -> R`, as `_explore_spies` snapshots after this runs.
        out = _Gen([_next(fn(), path)], "Iterator")
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


def _rollback(marks: Iterable[tuple[_SpyObject, int]]) -> None:
    for spy, length in marks:
        del spy.__optype_trace__[length:]


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


def _explore[T](
    func: Callable[..., T] | Callable[..., Coroutine[Any, None, T]],
    args: Sequence[object],
    kwds: Mapping[str, object],
) -> tuple[list[T], str | None]:
    results: list[T] = []
    deprecated: str | None = None
    stack: list[list[bool]] = [[]]
    dropped = False
    last_exc: Exception | None = None
    for _ in range(_RUN_LIMIT):  # caps the exponential blowup of independent forks
        if not stack:
            break
        plan = stack.pop()
        # a failed run must also roll back its traces on the closed-over spies
        marks = [
            (spy, len(spy.__optype_trace__))
            for spy in _reachable((*args, *kwds.values(), *_closed_over(func)))
        ]
        # `_starved` is a per-run splat signal, so a prior run cannot leak into this one
        _starved.set(False)
        token = _fork.set(iter(plan))

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
            _rollback(marks)
        except (InferError, IndexError, KeyError, TypeError, ValueError):
            raise  # signals the driver acts on, not a rejected run
        except Exception as exc:  # noqa: BLE001
            # the target rejected these spy values (assert, zero-division, ...); skip
            last_exc = exc
            _rollback(marks)
        finally:
            _fork.reset(token)
    if not results:
        raise InferError("the function never ran to completion") from last_exc
    if dropped or stack:
        warnings.warn("not every branch was explored", InferWarning, stacklevel=3)
    return results, deprecated


def _fixed_self(func: _AnyFunc, params: Mapping[str, Parameter]) -> dict[str, object]:
    if not isinstance(func, MethodDescriptorType | WrapperDescriptorType) or not params:
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


def _explore_spies(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    omit: Collection[str] = (),
    fix: Collection[str] = (),
) -> _Exploration:
    kinds = {p.kind for p in params.values()}

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
            fixed = _fixed_self(func, params) | {n: params[n].default for n in fix}
            spies, args, kwds = _placeholders(params, count, keys, omit, fixed)
            try:
                results, deprecated = _explore(func, args, kwds)
                results = [_next(r) for r in results]
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
                # a too-short splat raises `TypeError`; gate on it so the target's own
                # error (e.g. a `ValueError`) can't churn the budget and bury itself
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
                return _Exploration(
                    spies,
                    _snapshot((*spies.values(), *fn_spies(results))),
                    results,
                    count,
                    fixed,
                    deprecated,
                )
    finally:
        _starved.reset(starve_token)
        _yield_budget.reset(yield_token)
        _exploring.reset(token)


def _explore_lenient(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
) -> tuple[_Exploration, Mapping[str, object]]:
    # if a spy placeholder is rejected, fall back to fixing the defaulted parameters,
    # and also return every parameter default for rendering
    try:
        return _explore_spies(func, params), {}
    except (TypeError, ValueError):
        defaults = _declared_defaults(params)
        if not defaults:
            raise
    # start from the all-fixed baseline and greedily promote one spy at a time
    fix = set(defaults)
    exploration = _explore_spies(func, params, fix=fix)
    for name in defaults:
        with suppress(Exception):
            exploration = _explore_spies(func, params, fix=fix - {name})
            fix.discard(name)
    return exploration, defaults
