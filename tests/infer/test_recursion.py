"""`collapse_recursive`: folding an unrolled loop's typevar chain back into one."""

import pytest

from optype.infer._ir import App, Name, Node, Param, Signature, TypeParam, Unpack
from optype.infer._recursion import collapse_recursive


def _add(a: str, b: str) -> Node:
    return App("CanAdd", (Name(a), Name(b)))


def _mul(a: str, b: str) -> Node:
    return App("CanMul", (Name(a), Name(b)))


# an unrolled four-pass loop: each copy's bound is the previous one, shifted
CHAIN = (
    TypeParam("T", _add("T7", "U")),
    TypeParam("U", _add("T8", "V")),
    TypeParam("V", _add("T9", "W")),
    TypeParam("W", _add("T10", "T11")),
    *(TypeParam(leaf) for leaf in ("T7", "T8", "T9", "T10", "T11")),
)


def test_collapse_recursive_reroll() -> None:
    # #736: an unrolled loop's run of identical bounds folds to one recursive typevar
    def link(leaf: str, nxt: str) -> Node:
        return App("CanAdd", (Name(leaf), Name(nxt)))

    type_params = [
        TypeParam("T", link("T7", "U")),
        TypeParam("U", link("T8", "V")),
        TypeParam("V", link("T9", "W")),
        TypeParam("W", link("T10", "T11")),  # the last copy points at the loop's exit
        *(TypeParam(leaf) for leaf in ("T7", "T8", "T9", "T10", "T11")),
    ]
    sig = Signature(tuple(type_params), (Param("x", Name("T")),), Name("T"))
    folded = collapse_recursive(sig)
    # T, U, V, W collapse onto a single self-referential T; spent leaves are dropped
    assert folded.type_params == (
        TypeParam("T", App("CanAdd", (Name("U"), Name("T")))),
        TypeParam("U"),  # the surviving per-iteration leaf, renumbered gaplessly
    )
    assert folded.params == sig.params
    assert folded.ret == Name("T")


def test_collapse_recursive_keeps_short_runs() -> None:
    # #736: below the loop threshold, similar typevars are left intact (no false fold)
    type_params = [
        TypeParam("T", App("CanAdd", (Name("T7"), Name("U")))),
        TypeParam("U", App("CanAdd", (Name("T8"), Name("T9")))),
        *(TypeParam(leaf) for leaf in ("T7", "T8", "T9")),
    ]
    sig = Signature(tuple(type_params), (Param("x", Name("T")),), Name("T"))
    assert collapse_recursive(sig) == sig


def test_collapse_recursive_pair() -> None:
    # two variables that go round the loop together fold into a mutually recursive pair
    type_params = (
        TypeParam("T", _add("U", "V")),
        TypeParam("U", _mul("T", "W")),
        TypeParam("V", _add("W", "X")),
        TypeParam("W", _mul("V", "Y")),
        TypeParam("X", _add("Y", "Z")),
        TypeParam("Y", _mul("X", "T7")),
        TypeParam("Z", _add("T7", "T8")),  # the last copy points at the loop's exit
        TypeParam("T7", _mul("Z", "T9")),
        TypeParam("T8"),
        TypeParam("T9"),
    )
    params = Param("x", Name("T")), Param("y", Name("U"))
    folded = collapse_recursive(Signature(type_params, params, Name("T")))
    assert folded.type_params == (
        TypeParam("T", _add("U", "T")),
        TypeParam("U", _mul("T", "U")),
    )
    assert folded.params == params
    assert folded.ret == Name("T")


def test_collapse_recursive_keeps_typevar_tuple_and_deprecation() -> None:
    # a `*Ts` binder has no bound to fold, and stays reachable through its parameter
    params = Param("x", Name("T")), Param("args", Unpack(Name("Ts")), prefix="*")
    sig = Signature((TypeParam("Ts", unpack=True), *CHAIN), params, Name("T"), "old")
    assert collapse_recursive(sig) == Signature(
        (TypeParam("Ts", unpack=True), TypeParam("T", _add("U", "T")), TypeParam("U")),
        params,
        Name("T"),
        "old",
    )


def test_collapse_recursive_without_bounds_is_identity() -> None:
    type_params = TypeParam("T"), TypeParam("U")
    sig = Signature(type_params, (Param("x", Name("T")),), Name("T"))
    assert collapse_recursive(sig) == sig


NO_FOLD_CASES: list[tuple[str, tuple[TypeParam, ...]]] = [
    (
        # three alike bounds, but only `T -> U` advances: a two-copy run is too short
        "short run",
        (
            TypeParam("T", _add("T7", "U")),
            TypeParam("U", _add("T8", "V")),
            TypeParam("V", _add("T9", "T10")),
            *(TypeParam(leaf) for leaf in ("T7", "T8", "T9", "T10")),
        ),
    ),
    (
        # alike bounds that only share `X`; the renaming sends `X` to an unbounded leaf,
        # so it is not a step through the loop
        "shared name",
        (
            TypeParam("T", _add("X", "T7")),
            TypeParam("U", _add("X", "T8")),
            TypeParam("V", _add("X", "T9")),
            TypeParam("X", _add("T11", "T12")),
            *(TypeParam(leaf) for leaf in ("T7", "T8", "T9", "T11", "T12")),
        ),
    ),
    (
        # alike bounds around one shared, bounded name, which is a dependency they have
        # in common, not a step through the loop
        "shared dependency",
        (
            TypeParam("T", _add("X", "T7")),
            TypeParam("U", _add("X", "T8")),
            TypeParam("V", _add("X", "T9")),
            TypeParam("X", _add("X", "T10")),
            *(TypeParam(leaf) for leaf in ("T7", "T8", "T9", "T10")),
        ),
    ),
    (
        # an already recursive pair links both ways, and has nothing left to fold
        "already recursive",
        (
            TypeParam("T", _add("U", "T")),
            TypeParam("U", _add("T", "U")),
            TypeParam("V", App("CanNeg", (Name("W"),))),
            TypeParam("W"),
        ),
    ),
]


@pytest.mark.parametrize(
    ("type_params"),
    [case for _, case in NO_FOLD_CASES],
    ids=[label for label, _ in NO_FOLD_CASES],
)
def test_collapse_recursive_no_fold(type_params: tuple[TypeParam, ...]) -> None:
    params = tuple(Param(name, Name(name.upper())) for name in ("t", "u", "v"))
    sig = Signature(type_params, params, Name("T"))
    assert collapse_recursive(sig) == sig


def test_collapse_recursive_renames_through_bounds_defaults_and_return() -> None:
    # whatever mentions a folded copy or a renumbered leaf follows the fold
    params = Param("x", Name("T")), Param("y", Name("Y")), Param("z", Name("Z"))
    extra = (
        TypeParam("Y", bound=App("list", (Name("V"),))),
        TypeParam("Z", default=Name("V")),
    )
    ret = App("tuple", (Name("V"), Name("T7")))
    folded = collapse_recursive(Signature((*CHAIN, *extra), params, ret))
    assert folded.type_params == (
        TypeParam("T", _add("U", "T")),
        TypeParam("U"),
        TypeParam("V", bound=App("list", (Name("T"),))),
        TypeParam("W", default=Name("T")),
    )
    assert folded.params == (
        Param("x", Name("T")),
        Param("y", Name("V")),
        Param("z", Name("W")),
    )
    assert folded.ret == App("tuple", (Name("T"), Name("U")))


def test_collapse_recursive_keeps_return_only_binder() -> None:
    # the return type is a reachability root of its own
    sig = Signature((*CHAIN, TypeParam("R")), (Param("x", Name("T")),), Name("R"))
    folded = collapse_recursive(sig)
    assert folded.type_params == (
        TypeParam("T", _add("U", "T")),
        TypeParam("U"),
        TypeParam("R"),
    )
    assert folded.ret == Name("R")


def test_collapse_recursive_keeps_default_only_reference() -> None:
    # a binder reached only through another's default is still in use
    extra = TypeParam("Z", default=Name("D")), TypeParam("D")
    params = Param("x", Name("T")), Param("z", Name("Z"))
    folded = collapse_recursive(Signature((*CHAIN, *extra), params, Name("T")))
    assert folded.type_params == (
        TypeParam("T", _add("U", "T")),
        TypeParam("U"),
        TypeParam("V", default=Name("D")),
        TypeParam("D"),
    )
    assert folded.params == (Param("x", Name("T")), Param("z", Name("V")))
