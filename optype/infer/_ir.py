"""A minimal algebraic representation of inferred type expressions and signatures."""

import builtins
import sys
import types
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass
from typing import Final, override

type Node = (
    Lit | Type | Name | App | Fn | Union | Intersection | Not | Variance | Unpack | Dots
)
type Term = Node | Arg
type Terms = tuple[Term, ...]

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


OBJECT: Final[Name] = Name("object")
NEVER: Final[Name] = Name("Never")
NONE: Final[Name] = Name("None")

_TOP = frozenset({OBJECT, Type(object)})


@dataclass(frozen=True, slots=True)
class Arg:
    """An optionally keyword-labeled parameter, e.g. `a: Literal[1]` or `T = 1`."""

    key: str | None
    value: Node
    default: tuple[object] | None = None  # the boxed default value, if any


@dataclass(frozen=True, slots=True)
class App:
    """A (subscripted) named type, e.g. `CanAdd[Literal[1], R]`."""

    origin: str
    args: Terms


@dataclass(frozen=True, slots=True)
class Fn:
    """A function type in signature syntax, e.g. `(x: T) -> R` or `(T) -> R`."""

    params: Terms
    ret: Node


def _param_type(param: Term) -> Node:
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
class Intersection:
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


def _subtype_args(origin: str, args: Terms, wider: Terms) -> bool:
    """Whether same-`origin` applications relate, argument by argument."""
    if len(args) != len(wider):
        return False
    # an all-`Never` container holds only `[]`, a member of any same origin
    if args and all(arg == NEVER for arg in args):
        return True
    if not (variances := _VARIANCES.get(origin, "")):
        return False
    signs = variances.ljust(len(args), variances[-1])
    return all(
        subtype(arg, wide) if sign == COVARIANT else subtype(wide, arg)
        for arg, wide, sign in zip(args, wider, signs, strict=False)
    )


def subtype(sub: Term, sup: Term) -> bool:
    """Whether `sub` is a subtype of `sup`, as far as can be told from the nodes."""

    # a set would hash `sub`, which could have unhashable defaults
    if sub in (sup, NEVER) or sup in _TOP:  # noqa: PLR6201
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
        case App(origin, args), App(wider, wider_args) if origin == wider:
            result = _subtype_args(origin, args, wider_args)
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
    if not isinstance(node, App) or node.origin != "tuple" or not node.args:
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
            union([_param_type(g.args[i]) for g in group], tuples=True) or NEVER
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
    return neg if base is None else intersection((base, neg)) or neg


def intersection(parts: Iterable[Node]) -> Node | None:
    """The flat intersection of `parts`, unwrapped if singular, or `None`."""
    flat: dict[Node, None] = {}
    for part in parts:
        if isinstance(part, Intersection):
            flat.update(dict.fromkeys(part.parts))
        else:
            flat[part] = None
    if not flat:
        return None
    return next(iter(flat)) if len(flat) == 1 else Intersection(tuple(flat))


def names(node: Term) -> Generator[str]:
    # every type-name leaf, in order, so typevar uses can be counted
    match node:
        case Name(name):
            yield name
        case Arg(value=part) | Not(part) | Variance(part=part) | Unpack(part):
            yield from names(part)
        case App(args=parts) | Union(parts) | Intersection(parts):
            for part in parts:
                yield from names(part)
        case Fn(params, ret):
            for part in (*params, ret):
                yield from names(part)
        case Lit() | Type() | Dots():
            return


def subst(node: Node, m: Mapping[str, Node], *, dedup: bool = False) -> Node:
    """Replace every `Name(n)` with `m[n]`."""
    if not m:
        return node

    match node:
        case Name(name):
            out = m.get(name, node)
        case App(origin, args):
            out = App(origin, tuple(subst_term(a, m, dedup=dedup) for a in args))
        case Fn(params, ret):
            terms = tuple(subst_term(p, m, dedup=dedup) for p in params)
            out = Fn(terms, subst(ret, m, dedup=dedup))
        case Union(parts) | Intersection(parts):
            new = tuple(subst(p, m, dedup=dedup) for p in parts)
            if dedup:
                new = tuple(dict.fromkeys(new))
            out = new[0] if dedup and len(new) == 1 else type(node)(new)
        case Not(part) | Unpack(part):
            out = type(node)(subst(part, m, dedup=dedup))
        case Variance(sign, part):
            out = Variance(sign, subst(part, m, dedup=dedup))
        case _:
            out = node
    return out


def subst_term(term: Term, m: Mapping[str, Node], *, dedup: bool = False) -> Term:
    """`subst`, keeping any `Arg` wrapper of an `App`/`Fn` member."""
    if isinstance(term, Arg):
        return Arg(term.key, subst(term.value, m, dedup=dedup), term.default)
    return subst(term, m, dedup=dedup)


def rename(node: Node, m: Mapping[str, str]) -> Node:
    """Simultaneously rename every `Name(n)` to `Name(m[n])`."""
    return subst(node, {old: Name(new) for old, new in m.items()}, dedup=True)


def placeholder_name(n: int) -> str:
    """A placeholder name that cannot collide with a real identifier."""
    return f"\x00{n}"


def _canonical_renaming(node: Node) -> dict[str, str]:
    """Relabel each `Name` by first-occurrence order, to canonicalize via `rename`."""
    m: dict[str, str] = {}
    for name in names(node):
        m.setdefault(name, placeholder_name(len(m)))
    return m


def alpha_equal(a: Node, b: Node) -> dict[str, str] | None:
    """A `Name` bijection making `a` and `b` identical up to renaming, or `None`."""
    ca, cb = _canonical_renaming(a), _canonical_renaming(b)
    if rename(a, ca) != rename(b, cb):
        return None
    inv = {label: name for name, label in cb.items()}
    return {name: inv[label] for name, label in ca.items()}


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


def _public_module(cls: type) -> str | None:
    """The module `cls` is importable from, or `None` for a local class."""
    if (module := cls.__module__) == "builtins":
        return None

    # a private extension module defers to its public face (`_io` -> `io`)
    for candidate in (module.removeprefix("_"), module):
        if getattr(sys.modules.get(candidate), cls.__name__, None) is cls:
            return candidate

    return None


def type_name(cls: type) -> str:
    """The canonical importable name of a type."""
    if alias := _TYPES_NAMES.get(cls):
        return alias

    if cls.__module__.partition(".")[0] == "numpy":
        return f"np.{cls.__name__}"

    module = _public_module(cls)
    return f"{module}.{cls.__name__}" if module else cls.__name__


_TYVAR_LETTERS = "TUVWXYZ"


def tyvar_name(n: int) -> str:
    """The `n`-th generic parameter name: `T, U, ..., Z`, then `T7, T8, ...`."""
    return _TYVAR_LETTERS[n] if n < len(_TYVAR_LETTERS) else f"T{n}"


def tyvar_index(name: str) -> int | None:
    """The index of a generated typevar `name` (`T..Z` or `T<n>`), or `None`."""
    if len(name) == 1:
        return _TYVAR_LETTERS.index(name) if name in _TYVAR_LETTERS else None
    return int(name[1:]) if name[0] == "T" and name[1:].isdigit() else None
