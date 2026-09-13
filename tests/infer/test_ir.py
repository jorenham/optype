"""The `_ir` type algebra: subtyping, union and intersection simplification,
substitution, and naming, on hand-built nodes."""

import enum
import io
import types
from typing import Any

import pytest

from optype.infer._ir import (
    COVARIANT,
    NEVER,
    NONE,
    OBJECT,
    App,
    Arg,
    Dots,
    Fn,
    Has,
    Intersection,
    Lit,
    Name,
    Node,
    Not,
    Type,
    Union,
    Unpack,
    Variance,
    alpha_equal,
    exclude,
    intersection,
    names,
    rename,
    subst,
    subst_term,
    subtype,
    tuple_node,
    tuple_node_variadic,
    type_name,
    tyvar_index,
    tyvar_name,
    union,
)

A, B, C = Name("A"), Name("B"), Name("C")


FN_LIST_DEFAULT = Fn((Arg(None, Type(int), ([1],)),), Type(int))

SUBTYPE_CASES: list[tuple[Any, Any, bool]] = [
    (Type(bool), Type(int), True),
    (Type(int), Type(bool), False),
    (Type(int), Type(float), False),  # PEP 484's numeric tower is not real
    (Type(FileNotFoundError), Type(OSError), True),
    (Type(OSError), Type(FileNotFoundError), False),
    (Name("Never"), Type(str), True),
    (Type(str), Name("object"), True),
    (Type(str), Type(object), True),
    (Lit((True, 1)), Type(int), True),
    (Lit((1, "a")), Type(int), False),
    (Lit((1,)), Lit((1, 2)), True),
    (Lit((True,)), Lit((1,)), False),  # Literal[True] is not Literal[1]
    # the yield type is covariant
    (App("Generator", (Type(bool),)), App("Generator", (Type(int),)), True),
    (App("Generator", (Type(int),)), App("Generator", (Type(bool),)), False),
    # the send type is contravariant
    (
        App("Generator", (Type(int), Type(int), Type(int))),
        App("Generator", (Type(int), Type(bool), Type(int))),
        True,
    ),
    (
        App("Generator", (Type(int), Type(bool), Type(int))),
        App("Generator", (Type(int), Type(int), Type(int))),
        False,
    ),
    # tuple is covariant in every element; list is invariant
    (App("tuple", (Type(bool), Lit((1,)))), App("tuple", (Type(int),) * 2), True),
    (App("tuple", (Type(bool),)), App("tuple", (Type(int),) * 2), False),
    (App("list", (Type(bool),)), App("list", (Type(int),)), False),
    # an empty container is the bottom of its base, even when invariant
    (App("list", (Name("Never"),)), App("list", (Type(int),)), True),
    (App("list", (Type(int),)), App("list", (Name("Never"),)), False),
    (App("set", (Name("Never"),)), App("set", (Type(int),)), True),
    (App("dict", (Name("Never"),) * 2), App("dict", (Type(str), Type(int))), True),
    (App("list", (Name("Never"),)), App("set", (Type(int),)), False),
    # type is covariant
    (App("type", (Type(bool),)), App("type", (Type(int),)), True),
    (App("type", (Type(int),)), App("type", (Type(bool),)), False),
    # a function's parameters are contravariant, its return type is covariant
    (Fn((), Type(bool)), Fn((), Type(int)), True),
    (Fn((), Type(int)), Fn((), Type(bool)), False),
    (
        Fn((Arg("x", Type(int)),), Type(int)),
        Fn((Arg("x", Type(bool)),), Type(int)),
        True,
    ),
    (
        Fn((Arg("x", Type(bool)),), Type(int)),
        Fn((Arg("x", Type(int)),), Type(int)),
        False,
    ),
    (Fn((), Type(int)), Fn((Arg("x", Type(int)),), Type(int)), False),
    # a keyword must match by name, and an omission the wider one allows must be allowed
    (
        Fn((Arg("x", Type(int)),), Type(int)),
        Fn((Arg("y", Type(int)),), Type(int)),
        False,
    ),
    (Fn((Arg("x", Type(int)),), Type(int)), Fn((Type(int),), Type(int)), False),
    (Fn((Type(int),), Type(int)), Fn((Arg("x", Type(int)),), Type(int)), False),
    (
        Fn((Arg("x", Type(int)),), Type(int)),
        Fn((Arg("x", Type(int), (0,)),), Type(int)),
        False,
    ),
    (
        Fn((Arg("x", Type(int), (0,)),), Type(int)),
        Fn((Arg("x", Type(int)),), Type(int)),
        True,
    ),
    # the default's value is no part of the type
    (
        Fn((Arg("x", Type(int), (0,)),), Type(int)),
        Fn((Arg("x", Type(int), (1,)),), Type(int)),
        True,
    ),
    (
        Fn((Arg("x", Type(int), (1,)),), Type(int)),
        Fn((Arg("x", Type(int), (0,)),), Type(int)),
        True,
    ),
    (FN_LIST_DEFAULT, Fn((Arg(None, Type(int), ([2],)),), Type(int)), True),
    (Fn((Arg(None, Type(int), ([2],)),), Type(int)), FN_LIST_DEFAULT, True),
    # `zip` is covariant in typeshed; `map`, `filter`, and `enumerate` are invariant
    (App("zip", (Type(bool),)), App("zip", (Type(int),)), True),
    (App("map", (Type(bool),)), App("map", (Type(int),)), False),
    (App("enumerate", (Type(bool),)), App("enumerate", (Type(int),)), False),
    # an all-`Never` argument list is the empty container only where nothing is known
    # about the variance; a declared contravariant position decides for itself
    (App("Generator", (Name("Never"),) * 3), App("Generator", (Type(int),) * 3), False),
    (App("Generator", (Name("Never"),)), App("Generator", (Type(int),)), True),
    (App("tuple", (Name("Never"),)), App("tuple", (Type(int),)), True),
    # a union operand: every member below, any member above
    (Union((Type(bool), Lit((1,)))), Type(int), True),
    (Union((Type(bool), Type(str))), Type(int), False),
    (Type(bool), Union((Type(int), Type(str))), True),
    (Type(int), Union((Type(bool), Type(str))), False),
    # an unpacked element is covariant
    (Unpack(Type(bool)), Unpack(Type(int)), True),
    (Unpack(Type(int)), Unpack(Type(bool)), False),
    # a callable's unhashable default must not trip the identity check
    (FN_LIST_DEFAULT, FN_LIST_DEFAULT, True),
    (FN_LIST_DEFAULT, Fn((Arg(None, Type(bool), ([1],)),), Type(int)), True),
    (Fn((Arg(None, Type(bool), ([1],)),), Type(int)), FN_LIST_DEFAULT, False),
]


@pytest.mark.parametrize(("sub", "sup", "expected"), SUBTYPE_CASES)
def test_subtype(sub: Any, sup: Any, expected: bool) -> None:
    assert subtype(sub, sup) is expected


def test_lit_equality_is_type_sensitive() -> None:
    # `True == 1`, but `Literal[True]` is not `Literal[1]`
    assert Lit((1,)) == Lit((1,))
    assert hash(Lit((1,))) == hash(Lit((1,)))
    assert Lit((True,)) != Lit((1,))
    assert len({Lit((True,)), Lit((1,))}) == 2


def test_union_empty_and_singular() -> None:
    assert union([]) is None
    assert union([A]) == A


def test_union_flattens_and_dedups() -> None:
    assert union([Union((A, B)), Union((B, C))]) == Union((A, B, C))


def test_union_and_intersection_accept_unhashable_defaults() -> None:
    fn = FN_LIST_DEFAULT
    assert union([fn, fn]) == fn
    assert intersection([fn, fn]) == fn
    assert union([fn, A]) == Union((fn, A))
    assert subst(Union((fn, A)), {"A": B}, dedup=True) == Union((fn, B))


def test_union_and_intersection_flatten_every_level() -> None:
    assert union([Union((A, Union((B, C))))]) == Union((A, B, C))
    nested = Intersection((A, Intersection((B, C))))
    assert intersection([nested]) == Intersection((A, B, C))


def test_union_absorbs_subtypes() -> None:
    # a literal is covered by its type, a subclass by its parent, `Never` by anything
    assert union([Lit((1,)), Type(int)]) == Type(int)
    assert union([Type(int), Lit((1,))]) == Type(int)
    assert union([Type(bool), Type(int)]) == Type(int)
    assert union([NEVER, A]) == A
    assert union([NEVER]) == NEVER
    # either top absorbs everything
    assert union([A, OBJECT]) == OBJECT
    assert union([A, Type(object)]) == Type(object)


def test_union_absorbs_literals_one_value_at_a_time() -> None:
    # only the covered values of a literal group drop; the rest stay a literal
    cases: list[list[Node]] = [[Lit((1, "a")), Type(int)], [Type(int), Lit((1, "a"))]]
    for parts in cases:
        node = union(parts)
        assert isinstance(node, Union)
        assert set(node.parts) == {Lit(("a",)), Type(int)}


def test_union_merges_literals() -> None:
    assert union([Lit((1,)), Lit((2,))]) == Lit((1, 2))
    assert union([Lit((1, 2)), Lit((2, 3))]) == Lit((1, 2, 3))
    assert union([Lit((True,)), Lit((1,))]) == Lit((True, 1))


def test_union_tuple_collapse() -> None:
    # a wide union of same-arity tuples (like `colorsys.hls_to_rgb`) collapses per
    # position; a small one keeps its correlation
    def pair(i: int) -> App:
        return App("tuple", (Name(f"A{i}"), Name(f"B{i}")))

    def triple(i: int) -> App:
        return App("tuple", (Name(f"C{i}"), Name(f"D{i}"), Name(f"E{i}")))

    def column(prefix: str) -> Node:
        return Union(tuple(Name(f"{prefix}{i}") for i in range(9)))

    wide2 = App("tuple", (column("A"), column("B")))

    small: list[Node] = [pair(0), pair(1)]
    assert union(small, tuples=True) == Union((pair(0), pair(1)))

    wide: list[Node] = [pair(i) for i in range(9)]
    assert union(wide, tuples=True) == wide2
    assert union(wide, tuples=False) == Union(tuple(wide))
    # only a group strictly wider than the limit collapses
    assert union(wide[:8], tuples=True) == Union(tuple(wide[:8]))

    # each arity collapses on its own; a wider triple group folds independently
    wide3 = App("tuple", (column("C"), column("D"), column("E")))
    mixed = wide + [triple(i) for i in range(9)]
    assert union(mixed, tuples=True) == Union((wide2, wide3))

    # a non-tuple member and a variable-length tuple stay untouched beside the collapse
    assert union([*wide, Name("X")], tuples=True) == Union((wide2, Name("X")))
    variadic = App("tuple", (Name("V"), Dots()))
    assert union([*wide, variadic], tuples=True) == Union((wide2, variadic))
    unpacked = App("tuple", (Name("H"), Unpack(Name("Ts"))))
    assert union([*wide, unpacked], tuples=True) == Union((wide2, unpacked))


def test_intersection() -> None:
    assert intersection([]) is None
    assert intersection([A]) == A
    assert intersection([Intersection((A, B)), B, C]) == Intersection((A, B, C))


def test_exclude() -> None:
    assert exclude(None, A) == Not(A)
    assert exclude(B, A) == Intersection((B, Not(A)))
    assert exclude(Intersection((B, C)), A) == Intersection((B, C, Not(A)))


def test_subst() -> None:
    node = App("X", (A, Arg("k", A)))
    assert subst(node, {"A": B}) == App("X", (B, Arg("k", B)))
    assert subst(A, {}) is A
    # members that collapse together are only merged when `dedup` asks for it
    assert subst(Union((A, B)), {"A": B}) == Union((B, B))
    assert subst(Union((A, B)), {"A": B}, dedup=True) == B
    assert subst_term(Arg("k", A, (1,)), {"A": B}) == Arg("k", B, (1,))


def test_subst_through_every_wrapper() -> None:
    # attribute names, argument keys and defaults, and variance signs all survive
    nested = Fn(
        (Arg("k", A, (1,)), Unpack(A)),
        Has("s", (Variance(COVARIANT, A), Not(Intersection((A, B))))),
    )
    assert subst(nested, {"A": C}) == Fn(
        (Arg("k", C, (1,)), Unpack(C)),
        Has("s", (Variance(COVARIANT, C), Not(Intersection((C, B))))),
    )


def test_names_in_order_with_repeats() -> None:
    node = Fn((Arg("k", A), Union((B, A))), Has("s", (C,)))
    assert list(names(node)) == ["A", "B", "A", "C"]
    assert list(names(Lit((1,)))) == []
    assert list(names(Type(int))) == []
    assert list(names(Dots())) == []


def test_alpha_equal_and_rename() -> None:
    # the fold's primitives: structural equality up to a consistent name bijection,
    # and a simultaneous rename that drops union members which collapse together
    a = App("CanAdd", (Name("A"), App("CanMul", (Name("B"), Name("A")))))
    b = App("CanAdd", (Name("X"), App("CanMul", (Name("Y"), Name("X")))))
    tyvars = {"A", "B", "X", "Y"}
    assert alpha_equal(a, b, tyvars) == {"A": "X", "B": "Y"}
    assert alpha_equal(a, App("CanAdd", (Name("X"), Name("X"))), tyvars) is None
    assert rename(a, {"A": "X", "B": "Y"}) == b
    assert rename(Union((Name("A"), Name("B"))), {"A": "C", "B": "C"}) == Name("C")


def test_alpha_equal_is_a_bijection() -> None:
    # the same shape with a name used twice on one side only is not a bijection
    tyvars = {"A", "B", "C"}
    assert alpha_equal(App("X", (A, A)), App("X", (B, C)), tyvars) is None
    assert alpha_equal(App("X", (B, C)), App("X", (A, A)), tyvars) is None
    # a swap is a bijection, and renames simultaneously
    assert alpha_equal(App("X", (A, B)), App("X", (B, A)), tyvars) == {
        "A": "B",
        "B": "A",
    }
    assert rename(App("X", (A, B)), {"A": "B", "B": "A"}) == App("X", (B, A))


def test_alpha_equal_keeps_callable_metadata() -> None:
    # an argument's keyword and default are structure, not names, so they must match
    fn = Fn((Arg("x", A, (0,)),), B)
    tyvars = {"A", "B", "C"}
    assert alpha_equal(fn, Fn((Arg("x", C, (0,)),), A), tyvars) == {"A": "C", "B": "A"}
    assert alpha_equal(fn, Fn((Arg("y", A, (0,)),), B), tyvars) is None
    assert alpha_equal(fn, Fn((Arg("x", A, (1,)),), B), tyvars) is None
    assert alpha_equal(fn, Fn((Arg("x", A),), B), tyvars) is None


def test_alpha_equal_renames_only_type_parameters() -> None:
    # `None` and `object` are names too, but not ones that a renaming may touch
    a, b = App("CanAdd", (NONE, A)), App("CanAdd", (OBJECT, B))
    assert alpha_equal(a, b, {"A", "B"}) is None
    assert alpha_equal(a, App("CanAdd", (NONE, B)), {"A", "B"}) == {"A": "B"}


def test_tuple_nodes() -> None:
    assert tuple_node([A, B]) == App("tuple", (A, B))
    assert tuple_node_variadic(A) == App("tuple", (A, Dots()))


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (int, "int"),
        (io.BytesIO, "io.BytesIO"),  # `_io` defers to its public face
        (types.ModuleType, "ModuleType"),  # not the cpython-internal `module`
        (types.FunctionType, "FunctionType"),
        (enum.Enum, "enum.Enum"),
        (type("Local", (), {}), "Local"),  # a local class is not importable
    ],
)
def test_type_name(cls: type, expected: str) -> None:
    assert type_name(cls) == expected


def test_tyvar_names_round_trip() -> None:
    assert [tyvar_name(i) for i in range(9)] == [*"TUVWXYZ", "T7", "T8"]
    assert [tyvar_index(tyvar_name(i)) for i in range(9)] == list(range(9))
    assert tyvar_index("R") is None
    assert tyvar_index("Ts") is None
