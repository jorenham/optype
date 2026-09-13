"""The compat lowering, from hand-built `Signature`s to `.pyi` text."""

import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import pytest

from optype.infer import InferError
from optype.infer._backends._compat import COMPAT
from optype.infer._backends._compat._lower import Lowerer
from optype.infer._backends._compat._model import (
    Alias,
    Attr,
    Member,
    Method,
    Module,
    ProtocolDef,
    combine_name,
    components,
    cyclic_names,
    free_tyvars,
    resolution_order,
)
from optype.infer._backends._compat._print import OPTYPE
from optype.infer._ir import (
    CONTRAVARIANT,
    COVARIANT,
    NONE,
    OBJECT,
    App,
    Arg,
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

R, T, U, X = Name("R"), Name("T"), Name("U"), Name("X")
ZERO = Lit((0,))
TS = Unpack(Name("Ts"))


def _sig(type_params: tuple[TypeParam, ...], node: Node, ret: Node = R) -> Signature:
    return Signature(type_params, (Param("x", node),), ret)


def _neg_pos(node: Node) -> Node:
    return Intersection((App("CanNeg", (node,)), App("CanPos", (node,))))


# (label, signatures, rendered stub)
TEXT_CASES: list[tuple[str, tuple[Signature, ...], str]] = [
    (
        "intersection",
        (
            _sig(
                (TypeParam("R"),),
                Intersection((
                    App("CanGt", (ZERO, App("CanBool", ()))),
                    App("CanNeg", (R,)),
                )),
            ),
        ),
        (
            "from typing import Literal, Protocol\n"
            "from optype import CanBool, CanGt, CanNeg\n\n"
            "class CanGtNeg[T](CanGt[Literal[0], CanBool], CanNeg[T], Protocol): ..."
            "\n\n"
            "def f[R](x: CanGtNeg[R]) -> R: ..."
        ),
    ),
    (
        # `T & CanNeg[R]` constrains `T`; the acyclic bound then substitutes in place
        "intersection with typevar",
        (
            _sig(
                (TypeParam("T"), TypeParam("R")),
                Intersection((T, App("CanNeg", (R,)))),
            ),
        ),
        "from optype import CanNeg\n\ndef f[R](x: CanNeg[R]) -> R: ...",
    ),
    (
        "intersection distributes over union",
        (
            _sig(
                (TypeParam("R"),),
                Intersection((
                    Union((App("CanNeg", (R,)), App("CanPos", (R,)))),
                    App("CanBool", ()),
                )),
            ),
        ),
        (
            "from typing import Protocol\n"
            "from optype import CanBool, CanNeg, CanPos\n\n"
            "class CanBoolNeg[T](CanBool, CanNeg[T], Protocol): ...\n"
            "class CanBoolPos[T](CanBool, CanPos[T], Protocol): ...\n\n"
            "def f[R](x: CanBoolNeg[R] | CanBoolPos[R]) -> R: ..."
        ),
    ),
    (
        # one helper per distinct definition, whatever the binders are named
        "helper reuse",
        (
            _sig((TypeParam("R"),), _neg_pos(R)),
            Signature((TypeParam("X"),), (Param("y", _neg_pos(X)),), X),
            Signature(
                (TypeParam("R"),),
                (
                    Param(
                        "z",
                        Intersection((
                            App("CanNeg", (R,)),
                            App("CanPos", (Lit((1,)),)),
                        )),
                    ),
                ),
                R,
            ),
        ),
        (
            "from typing import Literal, Protocol, overload\n"
            "from optype import CanNeg, CanPos\n\n"
            "class CanNegPos[T](CanNeg[T], CanPos[T], Protocol): ...\n"
            "class CanNegPos2[T](CanNeg[T], CanPos[Literal[1]], Protocol): ...\n\n"
            "@overload\n"
            "def f[R](x: CanNegPos[R]) -> R: ...\n"
            "@overload\n"
            "def f[X](y: CanNegPos[X]) -> X: ...\n"
            "@overload\n"
            "def f[R](z: CanNegPos2[R]) -> R: ..."
        ),
    ),
    (
        # positional params fit `Callable`; a keyword or default needs a `__call__`
        "callable",
        (_sig((TypeParam("T"), TypeParam("R")), Fn((T,), R)),),
        (
            "from collections.abc import Callable\n\n"
            "def f[T, R](x: Callable[[T], R]) -> R: ..."
        ),
    ),
    (
        "callable with keyword",
        (_sig((TypeParam("R"),), Fn((Arg(None, ZERO), Arg("b", Lit((2,)))), R)),),
        (
            "from typing import Literal, Protocol\n\n"
            "class CanCallP[T](Protocol):\n"
            "    def __call__(self, _0: Literal[0], /, b: Literal[2]) -> T: ...\n\n"
            "def f[R](x: CanCallP[R]) -> R: ..."
        ),
    ),
    (
        # PEP 695 forbids a bound that references a type parameter; a cycle hoists
        "cyclic protocol bound",
        (_sig((TypeParam("T", App("CanLt", (T, App("CanBool", ())))),), T, T),),
        (
            "from typing import Protocol\n"
            "from optype import CanBool, CanLt\n\n"
            "class CanLt2(CanLt[CanLt2, CanBool], Protocol): ...\n\n"
            "def f(x: CanLt2) -> CanLt2: ..."
        ),
    ),
    (
        "cyclic concrete bound",
        (_sig((TypeParam("T", App("list", (T,))),), T, T),),
        "type list2 = list[list2]\n\ndef f(x: list2) -> list2: ...",
    ),
    (
        # an eliminated binder is substituted where a default mentions it
        "default of an eliminated binder",
        (
            Signature(
                (
                    TypeParam("T", App("CanNeg", (R,))),
                    TypeParam("R"),
                    TypeParam("U", default=T),
                ),
                (Param("x", T), Param("y", U)),
                R,
            ),
        ),
        (
            "from optype import CanNeg\n\n"
            "def f[R, U = CanNeg[R]](x: CanNeg[R], y: U) -> R: ..."
        ),
    ),
    (
        # a hoisted bound sees the substitution of the acyclic binder it mentions
        "recursive bound of an eliminated binder",
        (
            Signature(
                (
                    TypeParam("T", App("CanAdd", (T, U))),
                    TypeParam("U", App("CanNeg", (R,))),
                    TypeParam("R"),
                ),
                (Param("x", T),),
                R,
            ),
        ),
        (
            "from typing import Protocol\n"
            "from optype import CanAdd, CanNeg\n\n"
            "class CanAdd2[T](CanAdd[CanAdd2[T], CanNeg[T]], Protocol): ...\n\n"
            "def f[R](x: CanAdd2[R]) -> R: ..."
        ),
    ),
    (
        # a cycle, an acyclic binder, and another cycle resolve in dependency order
        "chained bounds",
        (
            Signature(
                (
                    TypeParam("T", App("CanAdd", (T, U))),
                    TypeParam("U", App("CanNeg", (X,))),
                    TypeParam("X", App("list", (X,))),
                ),
                (Param("x", T),),
                NONE,
            ),
        ),
        (
            "from typing import Protocol\n"
            "from optype import CanAdd, CanNeg\n\n"
            "type list2 = list[list2]\n"
            "class CanAdd2(CanAdd[CanAdd2, CanNeg[list2]], Protocol): ...\n\n"
            "def f(x: CanAdd2) -> None: ..."
        ),
    ),
    (
        # a mutually recursive pair hoists together, keeping the free typevar as an arg
        "mutually recursive bounds",
        (
            Signature(
                (
                    TypeParam("T", App("CanAdd", (U, X))),
                    TypeParam("U", App("CanMul", (T, X))),
                    TypeParam("X"),
                ),
                (Param("x", T), Param("y", U), Param("z", X)),
                X,
            ),
        ),
        (
            "from typing import Protocol\n"
            "from optype import CanAdd, CanMul\n\n"
            "class CanAdd2[T](CanAdd[CanMul2[T], T], Protocol): ...\n"
            "class CanMul2[T](CanMul[CanAdd2[T], T], Protocol): ...\n\n"
            "def f[X](x: CanAdd2[X], y: CanMul2[X], z: X) -> X: ..."
        ),
    ),
    (
        # an acyclic bound substitutes in place, transitively
        "acyclic bounds",
        (
            Signature(
                (
                    TypeParam("T", App("CanAdd", (U, R))),
                    TypeParam("U", App("CanNeg", (R,))),
                    TypeParam("R"),
                ),
                (Param("x", T),),
                R,
            ),
        ),
        (
            "from optype import CanAdd, CanNeg\n\n"
            "def f[R](x: CanAdd[CanNeg[R], R]) -> R: ..."
        ),
    ),
    (
        # a non-generic protocol drops its excess argument
        "protocol arity",
        (
            Signature(
                (TypeParam("R"),),
                (Param("x", App("CanLen", (R,))), Param("y", R)),
                R,
            ),
        ),
        "from optype import CanLen\n\ndef f[R](x: CanLen, y: R) -> R: ...",
    ),
    (
        # a bounded protocol parameter hands its bound to the inferred typevar
        "protocol bound",
        (
            Signature(
                (TypeParam("T"), TypeParam("R")),
                (Param("x", App("CanSequence", (T, R))), Param("i", T)),
                R,
            ),
        ),
        (
            "import typing\n"
            "from optype import CanSequence\n\n"
            "def f[T: typing.SupportsIndex | slice, R](x: CanSequence[T, R], i: T)"
            " -> R: ..."
        ),
    ),
    (
        # a protocol-typed bound is applied, not flattened to its arguments
        "protocol bound from a shipped generic",
        (_sig((TypeParam("R"),), App("CanIter", (R,))),),
        (
            "from optype import CanIter, CanNext\n\n"
            "def f[R: CanNext[object]](x: CanIter[R]) -> R: ..."
        ),
    ),
    (
        # a bound from outside `optype` is applied and imported like any other type
        "protocol bound from the standard library in a helper",
        (
            _sig(
                (TypeParam("R"),),
                Intersection((App("CanDir", (R,)), App("CanBool", ()))),
            ),
        ),
        (
            "import typing\n"
            "from collections.abc import Iterable\n"
            "from typing import Protocol\n"
            "from optype import CanBool, CanDir\n\n"
            "class CanDirBool[T: Iterable[typing.Any]]"
            "(CanDir[T], CanBool, Protocol): ...\n\n"
            "def f[R: Iterable[typing.Any]](x: CanDirBool[R]) -> R: ..."
        ),
    ),
    (
        # a helper inherits the bound of the shipped protocol parameter it fills
        "protocol bound in helper",
        (
            Signature(
                (TypeParam("T"), TypeParam("R")),
                (
                    Param(
                        "x",
                        Intersection((App("CanSequence", (T, R)), App("CanBool", ()))),
                    ),
                    Param("i", T),
                ),
                R,
            ),
        ),
        (
            "import typing\n"
            "from typing import Protocol\n"
            "from optype import CanBool, CanSequence\n\n"
            "class CanSequenceBool[T: typing.SupportsIndex | slice, U]"
            "(CanSequence[T, U], CanBool, Protocol): ...\n\n"
            "def f[T: typing.SupportsIndex | slice, R](x: CanSequenceBool[T, R], i: T)"
            " -> R: ..."
        ),
    ),
    (
        "settable property",
        (_sig((), Has("spam", (Variance(CONTRAVARIANT, ZERO),)), NONE),),
        (
            "from typing import Literal, Protocol\n\n"
            "class HasSpam(Protocol):\n"
            "    @property\n"
            "    def spam(self) -> object: ...\n"
            "    @spam.setter\n"
            "    def spam(self, value: Literal[0], /) -> None: ...\n\n"
            "def f(x: HasSpam) -> None: ..."
        ),
    ),
    (
        "shadowed setter type",
        (
            _sig(
                (),
                Has("list", (Variance(CONTRAVARIANT, App("list", (Name("Never"),))),)),
                NONE,
            ),
        ),
        (
            "import builtins\n"
            "from typing import Never, Protocol\n\n"
            "class HasList(Protocol):\n"
            "    @property\n"
            "    def list(self) -> object: ...\n"
            "    @list.setter\n"
            "    def list(self, value: builtins.list[Never], /) -> None: ...\n\n"
            "def f(x: HasList) -> None: ..."
        ),
    ),
    (
        "method and write",
        (
            _sig(
                (),
                Intersection((
                    Has("spam", (Fn((), OBJECT),)),
                    Has("spam", (Variance(CONTRAVARIANT, Fn((), NONE)),)),
                )),
                NONE,
            ),
        ),
        (
            "from collections.abc import Callable\n"
            "from typing import Protocol\n\n"
            "class HasSpam(Protocol):\n"
            "    @property\n"
            "    def spam(self) -> Callable[[], object]: ...\n"
            "    @spam.setter\n"
            "    def spam(self, value: Callable[[], None], /) -> None: ...\n\n"
            "def f(x: HasSpam) -> None: ..."
        ),
    ),
    (
        "member named like a type parameter",
        (_sig((TypeParam("T"),), Has("T", (Variance(CONTRAVARIANT, T),)), NONE),),
        (
            "from typing import Protocol\n\n"
            "class HasT[U](Protocol):\n"
            "    @property\n"
            "    def T(self) -> object: ...\n"
            "    @T.setter\n"
            "    def T(self, value: U, /) -> None: ...\n\n"
            "def f[T](x: HasT[T]) -> None: ..."
        ),
    ),
    (
        # `~` has no Python meaning, and a variance sign is only presentational
        "fictional forms dropped",
        (
            _sig(
                (TypeParam("T"),),
                Intersection((T, Not(NONE))),
                Variance(COVARIANT, T),
            ),
        ),
        "def f[T](x: T) -> T: ...",
    ),
    (
        "typevar tuple",
        (
            Signature(
                (TypeParam("Ts", unpack=True),),
                (Param("args", TS, prefix="*"),),
                App("tuple", (TS,)),
            ),
        ),
        "def f[*Ts](*args: *Ts) -> tuple[*Ts]: ...",
    ),
    (
        "typevar default",
        (
            Signature(
                (TypeParam("T", default=ZERO),),
                (Param("x", T, nameless=True, default=(0,)),),
                T,
                deprecated="old",
            ),
        ),
        (
            "from typing import Literal\n"
            "from typing_extensions import deprecated\n\n"
            "@deprecated('old')\n"
            "def f[T = Literal[0]](_0: T = 0, /) -> T: ..."
        ),
    ),
]


@pytest.mark.parametrize(
    ("sigs", "expected"),
    [(sigs, expected) for _, sigs, expected in TEXT_CASES],
    ids=[label for label, _, _ in TEXT_CASES],
)
def test_text(sigs: tuple[Signature, ...], expected: str) -> None:
    assert COMPAT.render(sigs) == expected


def test_text_typechecks(tmp_path: Path) -> None:
    # each rendered stub must be valid, self-contained, type-checkable Python
    if shutil.which("basedpyright") is None:
        pytest.skip("basedpyright is not installed")
    for label, sigs, _ in TEXT_CASES:
        stub = tmp_path / f"{label.replace(' ', '_')}.pyi"
        stub.write_text(f"{COMPAT.render(sigs)}\n")
    # run from `tmp_path` so the stubs are checked apart from the project's settings
    out = subprocess.run(
        ["basedpyright", "."],  # ruff: ignore[start-process-with-partial-path]
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout


def _protocols(module: Module) -> list[ProtocolDef]:
    protocols = [h for h in module.helpers if isinstance(h, ProtocolDef)]
    assert len(protocols) == len(module.helpers)
    return protocols


def _has_member(*signed: Node) -> Member:
    module = Lowerer().module([_sig((TypeParam("R"),), Has("spam", signed))])
    (helper,) = module.helpers
    assert isinstance(helper, ProtocolDef)
    (member,) = helper.members
    return member


def test_has_read_is_a_property() -> None:
    assert _has_member(Variance(COVARIANT, R)) == Attr("spam", T, readonly=True)


def test_has_read_write_is_a_settable_property() -> None:
    # a settable property is satisfied by a plain attribute and by a property alike
    signed = Variance(CONTRAVARIANT, R), Variance(COVARIANT, R)
    assert _has_member(*signed) == Attr("spam", T, setter=T)


def test_has_write_is_a_settable_property() -> None:
    # any attribute that accepts the written type will do, so the getter is `object`
    assert _has_member(Variance(CONTRAVARIANT, ZERO)) == Attr(
        "spam",
        OBJECT,
        setter=ZERO,
    )
    # a read of another type keeps both: an asymmetric property
    signed = Variance(CONTRAVARIANT, ZERO), Variance(COVARIANT, R)
    assert _has_member(*signed) == Attr("spam", T, setter=ZERO)


def test_has_presence_is_a_read_only_property() -> None:
    assert _has_member() == Attr("spam", OBJECT, readonly=True)


@pytest.mark.parametrize("reverse", [False, True])
def test_has_method_merges_with_a_write_as_a_callable_read(reverse: bool) -> None:
    parts = (
        Has("spam", (Fn((), OBJECT),)),
        Has("spam", (Variance(CONTRAVARIANT, ZERO),)),
    )
    node = Intersection(parts[::-1] if reverse else parts)
    module = Lowerer().module([_sig((), node, NONE)])
    assert [h.members for h in _protocols(module)] == [
        (Attr("spam", Fn((), OBJECT), setter=ZERO),),
    ]


def test_has_of_one_attribute_merge_within_an_intersection() -> None:
    # a separate read and write of one attribute lower to one member, not two bases
    node = Intersection((
        Has("spam", (Variance(CONTRAVARIANT, ZERO),)),
        Has("spam", (Variance(COVARIANT, R),)),
    ))
    module = Lowerer().module([_sig((TypeParam("R"),), node)])
    (helper,), (func,) = _protocols(module), module.funcs
    assert helper.members == (Attr("spam", T, setter=ZERO),)
    assert func.params[0].node == App(helper.name, (R,))


def test_has_method() -> None:
    method = Method("spam", (ZERO,), T)
    assert _has_member(Fn((ZERO,), Variance(COVARIANT, R))) == method


def test_has_classvar() -> None:
    signed = App("ClassVar", (Variance(COVARIANT, Type(int)),))
    assert _has_member(signed) == Attr("spam", Type(int), classvar=True)
    # a `ClassVar` cannot hold a typevar, so a generic one demotes to the instance form
    signed = App("ClassVar", (Variance(COVARIANT, R),))
    assert _has_member(signed) == Attr("spam", T, readonly=True)
    signed = App("ClassVar", (R,))
    assert _has_member(signed) == Attr("spam", T)
    signed = App(
        "ClassVar",
        (Variance(COVARIANT, R), Variance(CONTRAVARIANT, Type(int))),
    )
    assert _has_member(signed) == Attr("spam", T, setter=Type(int))
    # a class attribute keeps its read type
    signed = App(
        "ClassVar",
        (Variance(COVARIANT, Type(int)), Variance(CONTRAVARIANT, ZERO)),
    )
    assert _has_member(signed) == Attr("spam", Type(int), classvar=True)


def test_has_helper_name_avoids_shipped_protocol() -> None:
    # `HasName` is an `optype` import, so the synthesized helper takes another name
    module = Lowerer().module([
        _sig((TypeParam("R"),), Has("name", (Variance(COVARIANT, R),))),
    ])
    (helper,), (func,) = module.helpers, module.funcs
    assert helper.name not in OPTYPE
    assert func.params[0].node == App(helper.name, (R,))


def test_intersection_distributes_over_every_union() -> None:
    # `(A | B) & (C | D)` spreads to the four pairwise combinations
    node = Intersection((
        Union((App("CanNeg", (R,)), App("CanPos", (R,)))),
        Union((App("CanAbs", (R,)), App("CanInvert", (R,)))),
    ))
    module = Lowerer().module([_sig((TypeParam("R"),), node)])
    helpers = _protocols(module)
    assert {frozenset(h.bases) for h in helpers} == {
        frozenset({App(left, (T,)), App(right, (T,))})
        for left in ("CanNeg", "CanPos")
        for right in ("CanAbs", "CanInvert")
    }
    lowered = module.funcs[0].params[0].node
    assert isinstance(lowered, Union)
    assert set(lowered.parts) == {App(h.name, (R,)) for h in helpers}


def test_nested_argument_lowers_into_its_own_helper() -> None:
    # a helper's method parameter is lowered too, under the helper's own binders
    node = Fn((Arg("k", Has("spam", (Variance(COVARIANT, X),))),), R)
    sig = Signature((TypeParam("X"), TypeParam("R")), (Param("f", node),), R)
    module = Lowerer().module([sig])
    helpers = _protocols(module)
    (attr,) = [h for h in helpers if h.members == (Attr("spam", T, readonly=True),)]
    (call,) = [h for h in helpers if h is not attr]
    assert call.type_params == (TypeParam("T"), TypeParam("U"))
    assert call.members == (Method("__call__", (Arg("k", App(attr.name, (T,))),), U),)
    assert module.funcs[0].params[0].node == App(call.name, (X, R))


def test_helpers_are_keyed_on_callable_parameters() -> None:
    # a keyword name or a default is part of a `__call__` definition
    def call(key: str, default: tuple[object] | None = None) -> Node:
        return Fn((Arg(key, Type(int), default),), R)

    module = Lowerer().module([
        Signature((TypeParam("R"),), (Param("f", call("a")),), R),
        Signature((TypeParam("R"),), (Param("g", call("b")),), R),
        Signature((TypeParam("R"),), (Param("h", call("a", (0,))),), R),
        Signature((TypeParam("X"),), (Param("i", Fn((Arg("a", Type(int)),), X)),), X),
    ])
    by_members = {h.members: h for h in _protocols(module)}
    assert len(by_members) == 3
    a = by_members[Method("__call__", (Arg("a", Type(int)),), T),]
    b = by_members[Method("__call__", (Arg("b", Type(int)),), T),]
    a0 = by_members[Method("__call__", (Arg("a", Type(int), (0,)),), T),]
    assert [f.params[0].node for f in module.funcs] == [
        App(a.name, (R,)),
        App(b.name, (R,)),
        App(a0.name, (R,)),
        App(a.name, (X,)),
    ]


def test_intersected_callables_become_call_overloads() -> None:
    # every callable member lifts into the helper, one `__call__` overload each
    node = Intersection((Fn((Type(int),), Type(int)), Fn((Type(str),), Type(str))))
    module = Lowerer().module([Signature((), (Param("f", node),), NONE)])
    (helper,), (func,) = _protocols(module), module.funcs
    assert helper.members == (
        Method("__call__", (Type(int),), Type(int)),
        Method("__call__", (Type(str),), Type(str)),
    )
    assert func.params[0].node == App(helper.name, ())


def test_type_parameter_default_is_lowered() -> None:
    # a default is a type expression too, so a fictional one becomes a helper
    default = Has("spam", (Variance(COVARIANT, Type(int)),))
    sig = Signature((TypeParam("T", default=default),), (Param("x", T),), T)
    module = Lowerer().module([sig])
    (helper,), (func,) = _protocols(module), module.funcs
    assert helper.members == (Attr("spam", Type(int), readonly=True),)
    assert func.type_params == (TypeParam("T", default=App(helper.name, ())),)


def test_helpers_are_keyed_on_their_members() -> None:
    # one registry: another attribute or another access is another helper, while the
    # same definition under other binder names is the same helper
    read = Variance(COVARIANT, R)
    module = Lowerer().module([
        Signature((TypeParam("R"),), (Param("a", Has("spam", (read,))),), R),
        Signature((TypeParam("R"),), (Param("b", Has("ham", (read,))),), R),
        Signature(
            (TypeParam("R"),),
            (Param("c", Has("spam", (Variance(CONTRAVARIANT, R), read))),),
            R,
        ),
        Signature(
            (TypeParam("X"),),
            (Param("d", Has("spam", (Variance(COVARIANT, X),))),),
            X,
        ),
    ])
    by_members = {h.members: h for h in _protocols(module)}
    assert len(by_members) == 3
    spam = by_members[Attr("spam", T, readonly=True),]
    ham = by_members[Attr("ham", T, readonly=True),]
    spam_rw = by_members[Attr("spam", T, setter=T),]
    assert len({spam.name, ham.name, spam_rw.name}) == 3
    assert [f.params[0].node for f in module.funcs] == [
        App(spam.name, (R,)),
        App(ham.name, (R,)),
        App(spam_rw.name, (R,)),
        App(spam.name, (X,)),
    ]


@pytest.mark.parametrize("attr", ["a-b", "class"])
def test_has_non_identifier_attr_is_rejected(attr: str) -> None:
    with pytest.raises(InferError, match="cannot render attribute"):
        COMPAT.render([_sig((), Has(attr, ()), NONE)])


def test_recursive_helper_reuse_across_binders() -> None:
    # alpha-equal cyclic bounds hoist into one helper, whatever the binders are named
    first = Signature(
        (TypeParam("T", App("CanAdd", (T, R))), TypeParam("R")),
        (Param("x", T),),
        R,
    )
    second = Signature(
        (TypeParam("U", App("CanAdd", (U, X))), TypeParam("X")),
        (Param("y", U),),
        X,
    )
    module = Lowerer().module([first, second])
    (helper,) = module.helpers
    assert helper == ProtocolDef(
        helper.name,
        (TypeParam("T"),),
        (App("CanAdd", (App(helper.name, (T,)), T)),),
        (),
    )
    assert [f.params[0].node for f in module.funcs] == [
        App(helper.name, (R,)),
        App(helper.name, (X,)),
    ]


def test_callable_default_needs_call_protocol() -> None:
    # `Callable` cannot express a default, so it lifts into a `__call__` that keeps it
    module = Lowerer().module([
        _sig((TypeParam("R"),), Fn((Arg(None, ZERO, (0,)),), R)),
    ])
    (helper,), (func,) = _protocols(module), module.funcs
    assert helper.type_params == (TypeParam("T"),)
    assert helper.members == (Method("__call__", (Arg(None, ZERO, (0,)),), T),)
    assert func.params[0].node == App(helper.name, (R,))


def test_callable_intersected_with_protocol_lifts_into_call() -> None:
    # a callable is not a valid base, so it becomes the `__call__` member instead
    node = Intersection((Fn((T,), R), App("CanBool", ())))
    module = Lowerer().module([_sig((TypeParam("T"), TypeParam("R")), node)])
    (helper,), (func,) = _protocols(module), module.funcs
    assert helper == ProtocolDef(
        helper.name,
        (TypeParam("T"), TypeParam("U")),
        (App("CanBool", ()),),
        (Method("__call__", (T,), U),),
    )
    assert func.params[0].node == App(helper.name, (T, R))


def test_explicit_bound_merges_with_inferred_constraint() -> None:
    # a declared bound and a lifted intersection member end up in one helper
    node = Intersection((T, App("CanNeg", (R,))))
    sig = _sig((TypeParam("T", App("CanBool", ())), TypeParam("R")), node)
    module = Lowerer().module([sig])
    (helper,), (func,) = _protocols(module), module.funcs
    assert helper.bases == (App("CanBool", ()), App("CanNeg", (T,)))
    assert func.type_params == (TypeParam("R"),)
    assert func.params[0].node == App(helper.name, (R,))


def test_acyclic_bound_referencing_a_cyclic_one() -> None:
    # the cycle hoists into an alias, which the acyclic bound then substitutes with
    type_params = TypeParam("T", App("CanNeg", (U,))), TypeParam("U", App("list", (U,)))
    module = Lowerer().module([_sig(type_params, T, T)])
    (alias,), (func,) = module.helpers, module.funcs
    assert alias == Alias(alias.name, (), App("list", (App(alias.name, ()),)))
    assert func.type_params == ()
    assert func.params[0].node == func.ret == App("CanNeg", (App(alias.name, ()),))


def test_resolution_order_does_not_depend_on_the_hash_seed() -> None:
    # competing cyclic bounds resolve in one order whatever the seed
    script = """
from optype.infer._ir import App, Name, Param, Signature, TypeParam
from optype.infer._backends._compat import COMPAT
T, U, X = Name("T"), Name("U"), Name("X")
type_params = (
    TypeParam("T", App("tuple", (U, X))),
    TypeParam("U", App("list", (U,))),
    TypeParam("X", App("list", (App("tuple", (X, X)),))),
)
print(COMPAT.render([Signature(type_params, (Param("x", T),), T)]))
"""
    outputs = {
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("0", "1", "2")
    }
    assert len(outputs) == 1


def test_graph_helpers() -> None:
    deps = {
        "a": frozenset({"b"}),
        "b": frozenset({"a"}),
        "c": frozenset({"c", "a"}),  # reaches the `a`/`b` cycle, but not back
        "d": frozenset({"a"}),
        "e": frozenset[str](),
    }
    cyclic = cyclic_names(deps)
    assert cyclic == {"a", "b", "c"}
    groups = components(cyclic, deps)
    assert set(groups) == {frozenset({"a", "b"}), frozenset({"c"})}
    # every unit comes after the units it depends on
    order = resolution_order(groups, {"d", "e"}, deps)
    assert sorted(order, key=sorted) == [
        frozenset({"a", "b"}),
        frozenset({"c"}),
        frozenset({"d"}),
        frozenset({"e"}),
    ]
    assert order.index(frozenset({"a", "b"})) < order.index(frozenset({"c"}))
    assert order.index(frozenset({"a", "b"})) < order.index(frozenset({"d"}))


def test_naming_helpers() -> None:
    assert combine_name(["CanNeg", "CanRAdd"]) == "CanNegRAdd"
    assert combine_name(["HasA", "HasB"]) == "HasAB"
    assert combine_name(["CanNeg", "HasName"]) == "CanNegHasName"
    assert not combine_name([])
    nodes = App("X", (Name("B"), Name("A"), Name("B"))), Arg("k", Name("C"))
    assert free_tyvars(nodes, frozenset({"A", "B"})) == ["B", "A"]
