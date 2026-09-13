"""The shared data shapes of an exploration: its record, and the explored results."""

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextvars import Context
from dataclasses import dataclass, replace
from enum import StrEnum
from inspect import Parameter
from itertools import chain
from typing import Any, NamedTuple, NewType, TypeGuard

from ._spy import Spy, SpyObject, Traces

VARIADIC_KINDS = frozenset({Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD})


class GapKind(StrEnum):
    """A reason the exploration could not cover every path."""

    BRANCH_BUDGET = "branch budget exhausted"
    RUN_BUDGET = "run budget exhausted"


class Exploration(NamedTuple):
    """What one exploration of a function against spy placeholders produced."""

    spies: Mapping[str, SpyObject]
    traces: Traces
    results: Sequence[object]
    var_count: int  # the `*args` placeholder count
    fixed: Mapping[str, object]  # parameters passed as-is, not spies
    deprecated: str | None = None  # a `DeprecationWarning` message raised when called
    gaps: frozenset[GapKind] = frozenset()  # kinds of unexplored path
    tuple_params: frozenset[str] = frozenset()  # params also accepting a tuple of self


# not tuples: a `tuple()` match over a result tree must not match these


@dataclass(frozen=True, slots=True)
class Gen:
    """An explored generator, iterator, or coroutine result, e.g. `Generator[R]`."""

    yielded: Sequence[object]
    kind: str
    bare_when_empty: bool = False


# the `Gen.kind` of an awaited coroutine, rendered as `Coroutine[object, None, R]`
COROUTINE = "Coroutine"


@dataclass(frozen=True, slots=True)
class FnResult:
    """An explored function result, rendered in signature syntax."""

    params: Mapping[str, Parameter]
    spies: Mapping[str, SpyObject]
    fixed: Mapping[str, object]
    defaults: Mapping[str, object]
    results: Sequence[object]


# the shared identity of a recursive `Rec` binder and its `RecRef` uses
RecVar = NewType("RecVar", object)


@dataclass(frozen=True, slots=True)
class Rec:
    """A result that reaches itself, rendered as a recursive typevar bound."""

    var: RecVar  # the identity shared with this binder's `RecRef` uses
    body: Any


@dataclass(frozen=True, slots=True)
class RecRef:
    """A reference to the enclosing `Rec` binder of the same `var`."""

    var: RecVar


def is_mapping(value: object, /) -> TypeGuard[Mapping[Any, Any]]:
    """A `Mapping` that is not a `Context`; a `Context` is a leaf (gh-769)."""
    return isinstance(value, Mapping) and not isinstance(value, Context)


def children(value: Any) -> Iterable[Any]:
    """The values directly contained in an explored result."""
    if isinstance(value, Spy):
        # a spy is a leaf; its unique class defeats the `Mapping` check's negative cache
        return ()

    match value:
        case Gen():
            out: Iterable[object] = value.yielded
        case FnResult():
            out = value.results
        case Rec():
            out = (value.body,)
        case RecRef():
            out = ()
        case tuple() | list() | set() | frozenset():
            out = value
        case _ if is_mapping(value):
            out = chain.from_iterable(value.items())
        case slice():
            out = value.start, value.stop, value.step
        case _:
            out = ()
    return out


def walk(value: object) -> Generator[object]:
    yield value
    for child in children(value):
        yield from walk(child)


def map_values(value: Any, leaf: Callable[[Any], Any]) -> Any:  # ruff: ignore[complex-structure, too-many-branches]
    """Rebuild `value` with each non-composite leaf replaced via `leaf`.

    Recurses into the same shapes as `children`, but a `tuple` subclass (namedtuple)
    is a leaf, and a `dict` subclass (e.g. `defaultdict`) collapses to a plain `dict`.
    """

    if isinstance(value, Spy):
        # a spy is a leaf; see `children`
        return leaf(value)

    match value:
        case Gen():
            yielded = [map_values(item, leaf) for item in value.yielded]
            out = replace(value, yielded=yielded)
        case FnResult():
            results = [map_values(item, leaf) for item in value.results]
            out = replace(value, results=results)
        case Rec():
            out = replace(value, body=map_values(value.body, leaf))
        case RecRef():
            out = value
        case tuple() if type(value) is tuple:
            out = tuple(map_values(item, leaf) for item in value)
        case list():
            out = [map_values(item, leaf) for item in value]
        case set() | frozenset():
            items = {map_values(item, leaf) for item in value}
            out = frozenset(items) if isinstance(value, frozenset) else items
        case _ if is_mapping(value):
            mapping = value
            rebuilt = {
                map_values(k, leaf): map_values(v, leaf) for k, v in mapping.items()
            }
            if isinstance(value, dict):
                out = rebuilt  # any `dict` subclass collapses to a plain `dict`
            else:  # the `frozendict` builtin rebuilds as itself
                ctor = type(value)
                try:
                    out = ctor(rebuilt)  # type:ignore[call-arg]  # pyright:ignore[reportCallIssue]  # ty:ignore[too-many-positional-arguments]
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


def fn_spies(results: Iterable[object]) -> Generator[SpyObject]:
    for result in results:
        for node in walk(result):
            if isinstance(node, FnResult):
                yield from node.spies.values()
