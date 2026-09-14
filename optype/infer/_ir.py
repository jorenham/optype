"""A minimal algebraic representation of inferred type expressions and signatures."""

import builtins
import sys
import types
from collections.abc import Collection, Generator, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import starmap
from typing import Final, override

type Node = (
    Lit
    | Type
    | Name
    | App
    | Has
    | Fn
    | Union
    | Intersection
    | Not
    | Covariant
    | Contravariant
    | Unpack
    | Dots
)
type Term = Node | Arg
type Terms = tuple[Term, ...]


class Variance(StrEnum):
    """The variance of a type's argument position: covariant (`+`) or contravariant
    (`-`)."""

    COVARIANT = "+"
    CONTRAVARIANT = "-"


COVARIANT: Final = Variance.COVARIANT
CONTRAVARIANT: Final = Variance.CONTRAVARIANT

# variance per type argument; the last entry repeats variadically
_VARIANCES: dict[str, tuple[Variance, ...]] = {
    "AsyncGenerator": (COVARIANT, CONTRAVARIANT),
    "Generator": (COVARIANT, CONTRAVARIANT, COVARIANT),
    "frozenset": (COVARIANT,),
    "tuple": (COVARIANT,),
    "type": (COVARIANT,),
    # `enumerate`, `filter`, and `map` are invariant in typeshed; only `zip` is not
    "zip": (COVARIANT,),
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

_TOP: Final = OBJECT, Type(object)


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
class Has:
    """An attribute requirement, e.g. `Has['name', -T, +R]`: `-T` write, `+R` read,
    `Fn` method, or `ClassVar[...]`."""

    attr: str
    args: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Fn:
    """A function type in signature syntax, e.g. `(x: T) -> R` or `(T) -> R`."""

    params: Terms
    ret: Node


def term_node(term: Term) -> Node:
    """The node of a term, unwrapping any keyword-labeled `Arg`."""
    return term.value if isinstance(term, Arg) else term


@dataclass(frozen=True, slots=True)
class Not:
    """A type complement, e.g. the `~None` of everything but `None`."""

    part: Node


@dataclass(frozen=True, slots=True)
class Covariant:
    part: Node


@dataclass(frozen=True, slots=True)
class Contravariant:
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
    pos_only: bool = False
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
    if not (variances := _VARIANCES.get(origin)):
        # an all-`Never` invariant container holds only `[]`, a member of any same
        # origin; otherwise each argument has to be the same type, as written or not
        if args and all(arg == NEVER for arg in args):
            return True
        return _equivalent_all(args, wider)
    variances += variances[-1:] * (len(args) - len(variances))
    return all(
        subtype(arg, wide) if variance == COVARIANT else subtype(wide, arg)
        for arg, wide, variance in zip(args, wider, variances, strict=False)
    )


def _equivalent_all(xs: Iterable[Term], ys: Iterable[Term]) -> bool:
    return all(starmap(equivalent, zip(xs, ys, strict=True)))


def equivalent(a: Term, b: Term) -> bool:  # ruff: ignore[too-many-return-statements, too-many-locals]
    """Whether `a` and `b` are one type, up to the order of literal values and of
    union parts."""
    if a is b:
        return True
    match a, b:
        case Lit(values), Lit(others):
            return set(_keys(values)) == set(_keys(others))
        case App(origin, args), App(other, other_args):
            same = origin == other and len(args) == len(other_args)
            return same and _equivalent_all(args, other_args)
        case Union(parts), Union(others):
            return len(parts) == len(others) and all(
                any(equivalent(p, q) for q in others) for p in parts
            )
        case Fn(params, ret), Fn(other_params, other_ret):
            same = len(params) == len(other_params) and equivalent(ret, other_ret)
            return same and _equivalent_all(params, other_params)
        case Arg(key, value, default), Arg(other_key, other_value, other_default):
            same = key == other_key and default == other_default
            return same and equivalent(value, other_value)
        case Unpack(part), Unpack(other):
            return equivalent(part, other)
        case _:
            return a == b


def subtype(sub: Term, sup: Term) -> bool:
    """Whether `sub` is a subtype of `sup`, as far as can be told from the nodes."""

    # a set would hash the operands, which can carry an unhashable default
    if sub in (sup, NEVER) or sup in _TOP:  # ruff: ignore[literal-membership]
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
                    starmap(_param_subtype, zip(params, wider_params, strict=True)),
                )
            )
        case Unpack(part), Unpack(wider):
            result = subtype(part, wider)
        case _:
            result = False
    return result


def _param_subtype(param: Term, wider: Term) -> bool:
    """Whether `param` takes every argument its `wider` counterpart takes."""
    key, default = (
        (param.key, param.default) if isinstance(param, Arg) else (None, None)
    )
    if isinstance(wider, Arg):
        # the keyword must match, and an omission the wider one allows must be allowed
        if key != wider.key or (wider.default is not None and default is None):
            return False
    elif key is not None:
        return False
    return subtype(term_node(wider), term_node(param))


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
        # `_fixed_tuple_arity` already excluded any `Arg`, so `term_node` is a no-op
        columns = (
            union([term_node(g.args[i]) for g in group], tuples=True) or NEVER
            for i in range(arity)
        )
        out.append(tuple_node(columns))
    return out


def distinct(nodes: Iterable[Node]) -> list[Node]:
    """The distinct `nodes` by equality: a callable's default may be unhashable."""
    out: list[Node] = []
    for node in nodes:
        if node not in out:
            out.append(node)
    return out


def _flatten(parts: Iterable[Node], kind: type[Union | Intersection]) -> list[Node]:
    """The distinct members of `parts`, with every nested `kind` opened up."""
    flat: list[Node] = []
    stack = list(parts)[::-1]
    while stack:
        part = stack.pop()
        if isinstance(part, kind):
            stack.extend(reversed(part.parts))
        elif part not in flat:
            flat.append(part)
    return flat


def union(parts: Iterable[Node], *, tuples: bool = False) -> Node | None:
    """The simplified flat union of `parts`, unwrapped if singular, or `None`.

    With `tuples=True`, a wide union of same-arity tuples collapses per position;
    pass it only in covariant positions, where widening a tuple stays sound.
    """
    flat = _flatten(parts, Union)
    if not flat:
        return None
    nodes = _absorb(flat)
    if tuples:
        nodes = _collapse_tuples(nodes)
    return nodes[0] if len(nodes) == 1 else Union(tuple(nodes))


def exclude(base: Node | None, part: Node) -> Node:
    """The intersection of `base` (if any) with the complement of `part`."""
    neg = Not(part)
    return neg if base is None else intersection((base, neg)) or neg


def intersection(parts: Iterable[Node]) -> Node | None:
    """The flat intersection of `parts`, unwrapped if singular, or `None`."""
    flat = _flatten(parts, Intersection)
    if not flat:
        return None
    return flat[0] if len(flat) == 1 else Intersection(tuple(flat))


def names(node: Term) -> Generator[str]:
    # in order, so typevar uses can be counted
    match node:
        case Name(name):
            yield name
        case (
            Arg(value=part)
            | Not(part)
            | Covariant(part)
            | Contravariant(part)
            | Unpack(part)
        ):
            yield from names(part)
        case App(args=parts) | Has(args=parts) | Union(parts) | Intersection(parts):
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
        case Has(attr, args):
            out = Has(attr, tuple(subst(a, m, dedup=dedup) for a in args))
        case Fn(params, ret):
            terms = tuple(subst_term(p, m, dedup=dedup) for p in params)
            out = Fn(terms, subst(ret, m, dedup=dedup))
        case Union(parts) | Intersection(parts):
            new = tuple(subst(p, m, dedup=dedup) for p in parts)
            if dedup:
                new = tuple(distinct(new))
            out = new[0] if dedup and len(new) == 1 else type(node)(new)
        case Not(part) | Covariant(part) | Contravariant(part) | Unpack(part):
            out = type(node)(subst(part, m, dedup=dedup))
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


def _canonical_renaming(node: Node, tyvars: Collection[str]) -> dict[str, str]:
    """Relabel each of the `tyvars` by first-occurrence order, to canonicalize via
    `rename`."""
    m: dict[str, str] = {}
    for name in names(node):
        if name in tyvars:
            m.setdefault(name, placeholder_name(len(m)))
    return m


def alpha_equal(a: Node, b: Node, tyvars: Collection[str]) -> dict[str, str] | None:
    """A bijection over the `tyvars` making `a` and `b` identical, or `None`.

    Any other name, such as `None` or `object`, has to match as it is.
    """
    ca, cb = _canonical_renaming(a, tyvars), _canonical_renaming(b, tyvars)
    if rename(a, ca) != rename(b, cb):
        return None
    inv = {label: name for name, label in cb.items()}
    return {name: inv[label] for name, label in ca.items()}


def rename_signature(sig: Signature, m: Mapping[str, str]) -> Signature:
    """`rename` across a signature's type parameters, parameters, and return type."""

    def opt(node: Node | None) -> Node | None:
        return None if node is None else rename(node, m)

    typars = tuple(
        replace(
            typar,
            name=m.get(typar.name, typar.name),
            bound=opt(typar.bound),
            default=opt(typar.default),
        )
        for typar in sig.type_params
    )
    params = tuple(replace(p, node=rename(p.node, m)) for p in sig.params)
    return replace(sig, type_params=typars, params=params, ret=rename(sig.ret, m))


def alpha_equal_signatures(a: Signature, b: Signature) -> bool:
    """Whether `a` and `b` are one signature up to their type parameters' names."""
    canon = [
        rename_signature(
            sig,
            {
                typar.name: placeholder_name(i)
                for i, typar in enumerate(sig.type_params)
            },
        )
        for sig in (a, b)
    ]
    return canon[0] == canon[1]


# the two `types` members that alias another's type, so each type maps to one name
_TYPE_ALIASES = "LambdaType", "BuiltinMethodType"

# cpython-internal `__name__`s (`ModuleType.__name__ == "module"`) to importable names
_TYPES_NAMES: dict[type, str] = {
    cls: name
    for name, cls in vars(types).items()
    if isinstance(cls, type) and cls.__name__ != name and name not in _TYPE_ALIASES
}


def is_sentinel(x: object, /) -> bool:
    # the getattr works around a pyrefly (1.0.0) bug
    return sys.version_info >= (3, 15) and isinstance(x, getattr(builtins, "sentinel"))  # ruff: ignore[get-attr-with-constant]


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
