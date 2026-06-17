"""The composite shapes of explored results, and the traversal over them."""

from collections.abc import Collection, Generator, Iterable, Mapping
from inspect import Parameter
from itertools import chain
from typing import NamedTuple, NewType, cast

from ._spy import _SpyObject


class _Gen(NamedTuple):
    """An explored generator or lazy iterator result, e.g. `Generator[R]`."""

    yielded: list[object]
    kind: str


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
    body: object


class _RecRef(NamedTuple):
    """A reference to the enclosing `_Rec` binder of the same `var`."""

    var: _RecVar


def _children(value: object) -> Iterable[object]:
    """The values directly contained in an explored result."""
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
            out = cast("Collection[object]", value)
        case Mapping():  # `dict`, and the `frozendict` builtin on Python 3.15+
            out = chain.from_iterable(cast("Mapping[object, object]", value).items())
        case _:
            out = ()
    return out


def _walk(value: object) -> Generator[object]:
    yield value
    for child in _children(value):
        yield from _walk(child)


def fn_spies(results: Iterable[object]) -> Generator[_SpyObject]:
    # every parameter spy of the explored function results, in signature order
    for result in results:
        for node in _walk(result):
            if isinstance(node, _Fn):
                yield from node.spies.values()
