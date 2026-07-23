"""The shared data shapes of an exploration: its record, and the explored results."""

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from collections.abc import Callable, Generator, Iterable, Mapping
from contextvars import Context
from enum import StrEnum
from inspect import Parameter
from itertools import chain
from typing import Any, NamedTuple, NewType

from ._spy import _Spy, _SpyObject, _Traces

VARIADIC_KINDS = frozenset({Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD})


class GapKind(StrEnum):
    """A reason the exploration could not cover every path."""

    BRANCH_BUDGET = "branch budget exhausted"
    RUN_BUDGET = "run budget exhausted"


class Exploration(NamedTuple):
    """What one exploration of a function against spy placeholders produced."""

    spies: Mapping[str, _SpyObject]
    traces: _Traces
    results: list[object]
    var_count: int  # the `*args` placeholder count
    fixed: Mapping[str, object]  # parameters passed as-is, not spies
    deprecated: str | None = None  # a `DeprecationWarning` message raised when called
    gaps: frozenset[GapKind] = frozenset()  # kinds of unexplored path
    tuple_params: frozenset[str] = frozenset()  # params also accepting a tuple of self


class _Gen(NamedTuple):
    """An explored generator, iterator, or coroutine result, e.g. `Generator[R]`."""

    yielded: list[object]
    kind: str


# the `_Gen.kind` of an awaited coroutine, rendered as `Coroutine[object, None, R]`
COROUTINE = "Coroutine"


class _Fn(NamedTuple):
    """An explored function result, rendered in signature syntax."""

    params: Mapping[str, Parameter]
    spies: Mapping[str, _SpyObject]
    fixed: Mapping[str, object]
    defaults: Mapping[str, object]
    results: list[object]


# the shared identity of a recursive `_Rec` binder and its `_RecRef` uses
_RecVar = NewType("_RecVar", object)


class _Rec(NamedTuple):
    """A result that reaches itself, rendered as a recursive typevar bound."""

    var: _RecVar  # the identity shared with this binder's `_RecRef` uses
    body: Any


class _RecRef(NamedTuple):
    """A reference to the enclosing `_Rec` binder of the same `var`."""

    var: _RecVar


def _children(value: Any) -> Iterable[Any]:
    """The values directly contained in an explored result."""
    if isinstance(value, _Spy):
        # a spy is a leaf; its unique class defeats the `Mapping` check's negative cache
        return ()

    match value:
        case _Gen():
            out: Iterable[object] = value.yielded
        case _Fn():
            out = value.results
        case _Rec():
            out = (value.body,)
        case _RecRef():
            out = ()
        case tuple() | list() | set() | frozenset():
            out = value
        case Mapping() if not isinstance(value, Context):
            # `dict` and `frozendict` (py315+); a `Context` is a leaf, since its items
            # are just whatever context vars happen to be set (gh-769)
            out = chain.from_iterable(value.items())
        case slice():
            out = value.start, value.stop, value.step
        case _:
            out = ()
    return out


def _walk(value: object) -> Generator[object]:
    yield value
    for child in _children(value):
        yield from _walk(child)


def map_values(value: Any, leaf: Callable[[Any], Any]) -> Any:  # noqa: C901, PLR0912
    """Rebuild `value` with each non-composite leaf replaced via `leaf`.

    Recurses into the same shapes as `_children`, but a `tuple` subclass (namedtuple)
    is a leaf, and a `dict` subclass (e.g. `defaultdict`) collapses to a plain `dict`.
    """

    if isinstance(value, _Spy):
        # a spy is a leaf; see `_children`
        return leaf(value)

    match value:
        case _Gen():
            yielded = [map_values(item, leaf) for item in value.yielded]
            out: object = value._replace(yielded=yielded)
        case _Fn():
            results = [map_values(item, leaf) for item in value.results]
            out = value._replace(results=results)
        case _Rec():
            out = value._replace(body=map_values(value.body, leaf))
        case _RecRef():
            out = value
        case tuple() if type(value) is tuple:
            out = tuple(map_values(item, leaf) for item in value)
        case list():
            out = [map_values(item, leaf) for item in value]
        case set() | frozenset():
            items = {map_values(item, leaf) for item in value}
            out = frozenset(items) if isinstance(value, frozenset) else items
        case Mapping() if not isinstance(value, Context):
            mapping = value
            rebuilt = {
                map_values(k, leaf): map_values(v, leaf) for k, v in mapping.items()
            }
            if isinstance(value, dict):
                out = rebuilt  # any `dict` subclass collapses to a plain `dict`
            else:  # the `frozendict` builtin rebuilds as itself
                ctor = type(value)
                try:
                    out = ctor(rebuilt)  # type:ignore[call-arg]  # pyright:ignore[reportCallIssue]
                except TypeError:
                    out = mapping
        case slice():
            out = slice(
                map_values(value.start, leaf),
                map_values(value.stop, leaf),
                map_values(value.step, leaf),
            )
        case _:
            out = leaf(value)
    return out


def fn_spies(results: Iterable[object]) -> Generator[_SpyObject]:
    # every parameter spy of the explored function results, in signature order
    for result in results:
        for node in _walk(result):
            if isinstance(node, _Fn):
                yield from node.spies.values()
