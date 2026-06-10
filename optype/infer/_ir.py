"""A minimal algebraic representation of inferred type expressions."""

from collections.abc import Iterable
from dataclasses import dataclass

type Node = Lit | Name | App | Union | Inter


@dataclass(frozen=True, slots=True)
class Lit:
    """A `Literal[...]` type of one or more literal values."""

    values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class Name:
    """An opaque type expression, e.g. `int`, a typevar, or `dict[str, int]`."""

    name: str


@dataclass(frozen=True, slots=True)
class Arg:
    """A keyword-labeled type argument, e.g. the `a=Literal[1]` in `CanCall`."""

    key: str
    value: Node


@dataclass(frozen=True, slots=True)
class App:
    """A (subscripted) named type, e.g. `CanAdd[Literal[1], R]`."""

    base: str
    args: tuple[Node | Arg, ...]


@dataclass(frozen=True, slots=True)
class Union:
    """A `|`-union of two or more types."""

    parts: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Inter:
    """An `&`-intersection of two or more types."""

    parts: tuple[Node, ...]


def union(parts: Iterable[Node]) -> Node | None:
    """The flat union of `parts`, unwrapped if singular, or `None` if empty."""
    flat: dict[Node, None] = {}
    for part in parts:
        if isinstance(part, Union):
            flat.update(dict.fromkeys(part.parts))
        else:
            flat[part] = None
    if not flat:
        return None
    return next(iter(flat)) if len(flat) == 1 else Union(tuple(flat))


def inter(parts: Iterable[Node]) -> Node | None:
    """The flat intersection of `parts`, unwrapped if singular, or `None`."""
    flat: dict[Node, None] = {}
    for part in parts:
        if isinstance(part, Inter):
            flat.update(dict.fromkeys(part.parts))
        else:
            flat[part] = None
    if not flat:
        return None
    return next(iter(flat)) if len(flat) == 1 else Inter(tuple(flat))


def render(node: Node) -> str:
    """Format a type expression, parenthesized where precedence requires."""
    match node:
        case Lit(values):
            return f"Literal[{', '.join(map(repr, values))}]"
        case Name(name):
            return name
        case App(base, args):
            if not args:
                return base
            parts = [
                f"{arg.key}={render(arg.value)}"
                if isinstance(arg, Arg)
                else render(arg)
                for arg in args
            ]
            return f"{base}[{', '.join(parts)}]"
        case Union(parts):
            return " | ".join(
                f"({render(part)})" if isinstance(part, Inter) else render(part)
                for part in parts
            )
        case Inter(parts):
            return " & ".join(
                f"({render(part)})" if isinstance(part, Union) else render(part)
                for part in parts
            )
