"""The shared backend text helpers, and the terse renderer's precedence rules."""

import enum
import http
import io
import types

import pytest

from optype.infer._backends._base import (
    default_text,
    qualified_default_text,
    qualified_value_text,
    value_text,
)
from optype.infer._backends._terse import TERSE
from optype.infer._ir import (
    CONTRAVARIANT,
    COVARIANT,
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
    Param,
    Signature,
    Type,
    TypeParam,
    Union,
    Unpack,
    Variance,
)

A, B, C = Name("A"), Name("B"), Name("C")
FN = Fn((A,), B)


class Color(enum.Enum):
    RED = 1


class Flag(enum.Flag):
    X = 1
    Y = 2


def test_value_text() -> None:
    assert value_text(1) == "1"
    assert value_text("s") == "'s'"
    # an importable enum member renders as its bare member path
    assert value_text(http.HTTPStatus.NOT_FOUND) == "HTTPStatus.NOT_FOUND"
    assert value_text(Color.RED) == "Color.RED"
    # a composite flag has no member name, so it falls back to its data value
    assert value_text(Flag.X | Flag.Y) == "3"


def test_value_text_local_enum() -> None:
    class Local(enum.Enum):
        A = "a"

    assert value_text(Local.A) == "'a'"
    assert default_text(Local.A) == "..."


def test_default_text() -> None:
    assert default_text(None) == "None"
    assert default_text(1.5) == "1.5"
    assert default_text(2j) == "2j"
    assert default_text(float("inf")) == "..."
    assert default_text(float("nan")) == "..."
    assert default_text(complex(1, float("inf"))) == "..."


def test_default_text_float_subclass() -> None:
    class Odd(float):
        def __complex__(self) -> complex:
            raise NotImplementedError

    assert default_text(Odd(1.5)) == "1.5"
    assert default_text(Odd("inf")) == "..."
    assert default_text(b"x") == "b'x'"
    assert default_text(Color.RED) == "Color.RED"
    # only a literal repr is stub-safe; anything else elides
    assert default_text([1]) == "..."
    assert default_text(object()) == "..."


def test_qualified_text_records_import_path() -> None:
    recorded: list[str] = []
    status = http.HTTPStatus.NOT_FOUND
    assert qualified_value_text(status, recorded.append) == "http.HTTPStatus.NOT_FOUND"
    assert (
        qualified_default_text(status, recorded.append) == "http.HTTPStatus.NOT_FOUND"
    )
    assert qualified_default_text(1, recorded.append) == "1"
    assert recorded == ["http.HTTPStatus", "http.HTTPStatus"]


TERSE_CASES: list[tuple[Node, str]] = [
    (Lit((1, "a", True, None, b"x")), "Literal[1, 'a', True, None, b'x']"),
    (Type(io.BytesIO), "io.BytesIO"),
    (Type(types.ModuleType), "ModuleType"),
    (Name("None"), "None"),
    (App("tuple", ()), "tuple[()]"),
    (App("tuple", (A, Dots())), "tuple[A, ...]"),
    (App("X", (Arg("k", A),)), "X[k=A]"),
    (Has("spam", ()), "Has['spam']"),
    (
        Has("spam", (Variance(CONTRAVARIANT, A), Variance(COVARIANT, B))),
        "Has['spam', -A, +B]",
    ),
    (Fn((Arg("k", A, (1,)), Arg(None, B)), C), "(k: A = 1, B) -> C"),
    (Fn((Dots(),), C), "(...) -> C"),
    # a prefix binds tighter than an infix, so an infix operand is parenthesized
    (Not(A), "~A"),
    (Not(Union((A, B))), "~(A | B)"),
    (Variance(COVARIANT, Intersection((A, B))), "+(A & B)"),
    (Unpack(FN), "*((A) -> B)"),
    (Union((Not(A), B)), "~A | B"),
    # the dual infix and a function type are parenthesized inside an infix
    (Union((Intersection((A, B)), C)), "(A & B) | C"),
    (Intersection((Union((A, B)), C)), "(A | B) & C"),
    (Union((FN, C)), "((A) -> B) | C"),
    (Intersection((FN, C)), "((A) -> B) & C"),
]


@pytest.mark.parametrize(
    ("node", "expected"),
    TERSE_CASES,
    ids=[expected for _, expected in TERSE_CASES],
)
def test_terse_node(node: Node, expected: str) -> None:
    assert TERSE.render([Signature((), (), node)]) == f"() -> {expected}"


def test_terse_signature() -> None:
    type_params = (
        TypeParam("Ts", unpack=True),
        TypeParam("T", bound=Type(int), default=Lit((0,))),
    )
    params = (
        Param("x", Name("T"), nameless=True, default=(0,)),
        Param("args", Unpack(Name("Ts")), prefix="*"),
        Param("k", A, default=([1],)),
    )
    sig = Signature(type_params, params, App("tuple", (Unpack(Name("Ts")),)), "old")
    assert TERSE.render([sig]) == (
        "@deprecated('old')\n"
        "[*Ts, T: int = Literal[0]](T = 0, *args: *Ts, k: A = ...) -> tuple[*Ts]"
    )


def test_terse_dedups_rendered_lines() -> None:
    # two positional-only parameters render alike whatever their hidden names
    sigs = [
        Signature((), (Param("x", A, nameless=True),), B),
        Signature((), (Param("y", A, nameless=True),), B),
        Signature((), (Param("x", A),), C),
    ]
    assert TERSE.render(sigs) == "(A) -> B\n(x: A) -> C"
