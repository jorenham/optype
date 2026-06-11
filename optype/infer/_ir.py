"""A minimal algebraic representation of inferred type expressions."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import override

type Node = Lit | Type | Name | App | Union | Inter | Not

_NOT = "~"  # the type complement prefix

# variance per type argument ("+" co, "-" contra); the last entry repeats variadically
_VARIANCES = {
    "AsyncGenerator": "+-",
    "Generator": "+-+",
    "frozenset": "+",
    "tuple": "+",
}


def _keys(values: Iterable[object]) -> tuple[tuple[type, object], ...]:
    """Type-sensitive: `True == 1`, but `Literal[True]` is not `Literal[1]`."""
    return tuple((type(value), value) for value in values)


@dataclass(frozen=True, slots=True, eq=False)
class Lit:
    """A `Literal[...]` type of one or more literal values."""

    values: tuple[object, ...]

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Lit) and _keys(self.values) == _keys(other.values)

    @override
    def __hash__(self) -> int:
        return hash(_keys(self.values))


@dataclass(frozen=True, slots=True)
class Type:
    """A concrete runtime type, e.g. `int` or `np.float64`."""

    cls: type


@dataclass(frozen=True, slots=True)
class Name:
    """An opaque type expression, e.g. a typevar, a protocol, or `None`."""

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
class Not:
    """A type complement, e.g. the `~None` of everything but `None`."""

    part: Node


@dataclass(frozen=True, slots=True)
class Union:
    """A `|`-union of two or more types."""

    parts: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Inter:
    """An `&`-intersection of two or more types."""

    parts: tuple[Node, ...]


def subtype(sub: Node | Arg, sup: Node | Arg) -> bool:
    """Whether `sub` is assignable to `sup`, as far as can be told from the nodes."""
    if sub == sup or sub == Name("Never") or sup in {Name("object"), Type(object)}:
        return True
    match sub, sup:
        case Union(parts), _:
            result = all(subtype(part, sup) for part in parts)
        case _, Union(parts):
            result = any(subtype(sub, part) for part in parts)
        case Lit(values), Lit(wider):
            result = set(_keys(values)) <= set(_keys(wider))
        case Lit(values), Type(cls):
            result = all(isinstance(value, cls) for value in values)
        case Type(cls), Type(wider):
            result = issubclass(cls, wider)
        case App(base, args), App(wider, wider_args) if base == wider:
            variances = _VARIANCES.get(base, "")
            result = (
                bool(variances)
                and len(args) == len(wider_args)
                and all(
                    subtype(arg, wide)
                    if variances[min(i, len(variances) - 1)] == "+"
                    else subtype(wide, arg)
                    for i, (arg, wide) in enumerate(zip(args, wider_args, strict=True))
                )
            )
        case _:
            result = False
    return result


def _absorb(nodes: list[Node]) -> list[Node]:
    """Drop the union members (and literal values) that another member covers."""
    atoms: list[Node] = []
    for node in nodes:
        if isinstance(node, Lit):
            atoms += (Lit((value,)) for value in node.values)
        else:
            atoms.append(node)

    kept: list[Node] = []
    for atom in atoms:
        if not any(subtype(atom, wide) for wide in kept):
            kept = [*(k for k in kept if not subtype(k, atom)), atom]

    merged: list[Node] = []
    for node in kept:
        if isinstance(node, Lit) and merged and isinstance(last := merged[-1], Lit):
            merged[-1] = Lit(last.values + node.values)
        else:
            merged.append(node)
    return merged


def union(parts: Iterable[Node]) -> Node | None:
    """The simplified flat union of `parts`, unwrapped if singular, or `None`."""
    flat: dict[Node, None] = {}
    for part in parts:
        if isinstance(part, Union):
            flat.update(dict.fromkeys(part.parts))
        else:
            flat[part] = None
    if not flat:
        return None
    nodes = _absorb(list(flat))
    return nodes[0] if len(nodes) == 1 else Union(tuple(nodes))


def exclude(base: Node | None, part: Node) -> Node:
    """The intersection of `base` (if any) with the complement of `part`."""
    neg = Not(part)
    return neg if base is None else inter((base, neg)) or neg


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
        case Type(cls):
            prefix = "np." if cls.__module__.partition(".")[0] == "numpy" else ""
            return prefix + cls.__name__
        case Name(name):
            return name
        case App(base, args):
            parts = [
                f"{arg.key}={render(arg.value)}"
                if isinstance(arg, Arg)
                else render(arg)
                for arg in args
            ]
            return f"{base}[{', '.join(parts)}]" if parts else base
        case Not(part):
            inner = render(part)
            return (
                f"{_NOT}({inner})" if isinstance(part, Union | Inter) else _NOT + inner
            )
        case Union(parts) | Inter(parts):
            sep, dual = (" | ", Inter) if isinstance(node, Union) else (" & ", Union)
            return sep.join(
                f"({render(part)})" if isinstance(part, dual) else render(part)
                for part in parts
            )
