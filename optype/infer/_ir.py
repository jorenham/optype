"""A minimal algebraic representation of inferred type expressions and signatures."""

import builtins
import sys
import types
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import override

type Node = (
    Lit | Type | Name | App | Fn | Union | Inter | Not | Variance | Unpack | Dots
)

# the shared variance signs: covariant (read-only) and contravariant (write-only)
COVARIANT = "+"
CONTRAVARIANT = "-"

# variance per type argument; the last entry repeats variadically
_VARIANCES = {
    "AsyncGenerator": COVARIANT + CONTRAVARIANT,
    "Generator": COVARIANT + CONTRAVARIANT + COVARIANT,
    "frozenset": COVARIANT,
    "tuple": COVARIANT,
    "type": COVARIANT,
    # `enumerate`, `filter`, and `map` are invariant in typeshed; only `zip` is not
    "zip": COVARIANT,
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
    """An optionally keyword-labeled parameter, e.g. `a: Literal[1]` or `T = 1`."""

    key: str | None
    value: Node
    default: tuple[object] | None = None  # the boxed default value, if any


@dataclass(frozen=True, slots=True)
class App:
    """A (subscripted) named type, e.g. `CanAdd[Literal[1], R]`."""

    base: str
    args: tuple[Node | Arg, ...]


@dataclass(frozen=True, slots=True)
class Fn:
    """A function type in signature syntax, e.g. `(x: T) -> R` or `(T) -> R`."""

    params: tuple[Node | Arg, ...]
    ret: Node


def _param_type(param: Node | Arg) -> Node:
    """The type of a (possibly keyword-labeled) parameter."""
    return param.value if isinstance(param, Arg) else param


@dataclass(frozen=True, slots=True)
class Not:
    """A type complement, e.g. the `~None` of everything but `None`."""

    part: Node


@dataclass(frozen=True, slots=True)
class Variance:
    """A variance-marked type: covariant (read-only) or contravariant (write-only)."""

    sign: str
    part: Node


@dataclass(frozen=True, slots=True)
class Unpack:
    """A PEP 646 unpacking, e.g. `*tuple[T, ...]` or a `*Ts` typevar tuple."""

    part: Node


@dataclass(frozen=True, slots=True)
class Dots:
    """The `...` ellipsis, as in `tuple[X, ...]` or a `(...) -> R` parameter list."""


@dataclass(frozen=True, slots=True)
class Union:
    """A `|`-union of two or more types."""

    parts: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Inter:
    """An `&`-intersection of two or more types."""

    parts: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class TypeParam:
    """One PEP 695 type parameter: `T`, `T: Bound`, `T = Default`, or `*Ts`."""

    name: str
    bound: Node | None = None
    default: Node | None = None
    unpack: bool = False


@dataclass(frozen=True, slots=True)
class Param:
    """One signature parameter: a prefix, name, type, and optional default suffix."""

    name: str
    node: Node
    prefix: str = ""  # "", "*", or "**"
    nameless: bool = False  # positional-only
    default: tuple[object] | None = None  # the boxed default value, if any


@dataclass(frozen=True, slots=True)
class Signature:
    """A fully analyzed, backend-agnostic `def` signature."""

    type_params: tuple[TypeParam, ...]
    params: tuple[Param, ...]
    ret: Node
    deprecated: str | None = None  # the `@deprecated` message, if any


def tuple_node(parts: Iterable[Node]) -> App:
    return App("tuple", tuple(parts))


def tuple_node_variadic(element: Node) -> App:
    return tuple_node((element, Dots()))


def subtype(sub: Node | Arg, sup: Node | Arg) -> bool:
    """Whether `sub` is a subtype of `sup`, as far as can be told from the nodes."""
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
            result = len(args) == len(wider_args) and (
                # an all-`Never` container holds only `[]`, a member of any same base
                (bool(args) and all(arg == Name("Never") for arg in args))
                or (
                    bool(variances)
                    and all(
                        subtype(arg, wide)
                        if variances[min(i, len(variances) - 1)] == COVARIANT
                        else subtype(wide, arg)
                        for i, (arg, wide) in enumerate(
                            zip(args, wider_args, strict=True),
                        )
                    )
                )
            )
        case Fn(params, ret), Fn(wider_params, wider_ret):
            # parameters are contravariant (and positionally matched), the return
            # type is covariant
            result = (
                len(params) == len(wider_params)
                and subtype(ret, wider_ret)
                and all(
                    subtype(_param_type(wide), _param_type(param))
                    for param, wide in zip(params, wider_params, strict=True)
                )
            )
        case Unpack(part), Unpack(wider):
            result = subtype(part, wider)
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


_UNION_TUPLE_LIMIT = 8  # keep positional correlation below this; collapse a wider union


def _fixed_tuple_arity(node: Node) -> int | None:
    """The arity of a fixed-length `tuple[...]`, or `None` if not one."""
    if not isinstance(node, App) or node.base != "tuple" or not node.args:
        return None
    if any(isinstance(arg, (Arg, Dots, Unpack)) for arg in node.args):
        return None  # a variadic `tuple[X, ...]` or `tuple[*Ts]` has no fixed arity
    return len(node.args)


def _collapse_tuples(nodes: list[Node]) -> list[Node]:
    """Merge each large group of same-arity tuples into one per-position union.

    `tuple[A, B] | tuple[C, D]` widens to `tuple[A | C, B | D]`; only a group wider
    than `_UNION_TUPLE_LIMIT` collapses, so a small union keeps its positional
    correlation (e.g. `tuple[int, str] | tuple[str, int]`).
    """
    groups: dict[int, list[App]] = {}
    for node in nodes:
        if isinstance(node, App) and (arity := _fixed_tuple_arity(node)) is not None:
            groups.setdefault(arity, []).append(node)
    collapse = {a for a, group in groups.items() if len(group) > _UNION_TUPLE_LIMIT}
    if not collapse:
        return nodes

    out: list[Node] = []
    done: set[int] = set()
    for node in nodes:
        arity = _fixed_tuple_arity(node)
        if arity is None or arity not in collapse:
            out.append(node)
            continue
        if arity in done:
            continue
        done.add(arity)
        group = groups[arity]
        # `_fixed_tuple_arity` already excluded any `Arg`, so `_param_type` is a no-op
        columns = (
            union([_param_type(g.args[i]) for g in group], tuples=True) or Name("Never")
            for i in range(arity)
        )
        out.append(tuple_node(columns))
    return out


def union(parts: Iterable[Node], *, tuples: bool = False) -> Node | None:
    """The simplified flat union of `parts`, unwrapped if singular, or `None`.

    With `tuples=True`, a wide union of same-arity tuples collapses per position;
    pass it only in covariant positions, where widening a tuple stays sound.
    """
    flat: dict[Node, None] = {}
    for part in parts:
        if isinstance(part, Union):
            flat.update(dict.fromkeys(part.parts))
        else:
            flat[part] = None
    if not flat:
        return None
    nodes = _absorb(list(flat))
    if tuples:
        nodes = _collapse_tuples(nodes)
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


def names(node: Node | Arg) -> Generator[str]:
    # every type-name leaf, in order, so typevar uses can be counted
    match node:
        case Name(name):
            yield name
        case Arg(value=value):
            yield from names(value)
        case App(args=parts) | Union(parts) | Inter(parts):
            for part in parts:
                yield from names(part)
        case Fn(params, ret):
            for part in (*params, ret):
                yield from names(part)
        case Not(part) | Variance(part=part) | Unpack(part):
            yield from names(part)
        case Lit() | Type() | Dots():
            return


# the two `types` members that alias another's type, so each type maps to one name
_TYPE_ALIASES = "LambdaType", "BuiltinMethodType"

# cpython-internal `__name__`s (`ModuleType.__name__ == "module"`) to importable names
_TYPES_NAMES: dict[type, str] = {
    tp: name
    for name, tp in vars(types).items()
    if isinstance(tp, type) and tp.__name__ != name and name not in _TYPE_ALIASES
}


def is_sentinel(x: object, /) -> bool:
    # the getattr works around a pyrefly (1.0.0) bug
    return sys.version_info >= (3, 15) and isinstance(x, getattr(builtins, "sentinel"))  # noqa: B009


def type_name(cls: type) -> str:
    """The canonical importable name of a type."""
    if alias := _TYPES_NAMES.get(cls):
        return alias
    prefix = "np." if cls.__module__.partition(".")[0] == "numpy" else ""
    return prefix + cls.__name__
