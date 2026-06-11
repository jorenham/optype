"""Run a function against spy placeholders and record what happens."""

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
from inspect import Parameter, isasyncgen, iscoroutine, isgenerator, signature
from itertools import islice
from typing import Any, NamedTuple, cast, overload

from ._errors import InferError, InferWarning
from ._spy import (
    _AbsentError,
    _AnyFunc,
    _Fork,
    _fork,
    _SpyObject,
    _SpyStr,
    _TraceItem,
)

__all__ = ("_Gen", "_Recon", "_Traces", "_doc_params", "_explore_spies", "_parameters")

_RE_DOC_SIGNATURE = re.compile(r"\b(\w+)\(([^)]*)\)")
_RE_DOC_PARAM = re.compile(r"(?:^|,)\s*\**([a-zA-Z_]\w*)")

_FORK_LIMIT = 64
_RUN_LIMIT = 256
_YIELD_LIMIT = 64
# the `*args` placeholder counts to try: exact arities first, then doubling so that
# large indices stay within reach
_VARIADIC_COUNTS = (2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024)
_KWARGS_LIMIT = 8  # max injected `**kwargs` keys

type _Traces = dict[int, list[_TraceItem]]

# the spies, their traces, the results, and the `*args` placeholder count
type _Recon = tuple[Mapping[str, _SpyObject], _Traces, list[object], int]


class _Gen(NamedTuple):
    yielded: list[object]
    is_async: bool

    @property
    def kind(self) -> str:
        return "AsyncGenerator" if self.is_async else "Generator"


def _reachable(params: Iterable[_SpyObject]) -> Generator[_SpyObject]:
    # every spy reachable from `params` through the recorded operations
    seen: set[int] = set()
    stack = list(params)
    while stack:
        spy = stack.pop()
        if id(spy) in seen:
            continue
        seen.add(id(spy))
        yield spy
        stack.extend(
            ret
            for item in spy.__optype_trace__
            if isinstance(ret := item.return_, _SpyObject)
        )


def _snapshot(params: Iterable[_SpyObject]) -> _Traces:
    """Capture the traces of every spy reachable from `params`."""
    return {id(spy): list(spy.__optype_trace__) for spy in _reachable(params)}


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
        marks = [
            (spy, len(spy.__optype_trace__))
            for spy in _reachable((*args, *kwds.values()))
        ]
        token = _fork.set(iter(plan))
        try:
            result = func(*args, **kwds)
            results.append(_await(result) if iscoroutine(result) else cast("T", result))
        except _Fork:
            if len(plan) < _FORK_LIMIT:
                stack.extend(([*plan, False], [*plan, True]))
            else:
                dropped = True
        except _AbsentError:
            # the dunder is genuinely needed, so this run (and its marker) never was
            for spy, length in marks:
                del spy.__optype_trace__[length:]
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
