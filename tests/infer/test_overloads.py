"""The defaults decision of `resolve_defaults`: renders compare up to typevar names."""

from optype.infer._ir import App, Name, Param, Signature, TypeParam
from optype.infer._overloads import _distinct, _same


def _sig(t: str, r: str) -> Signature:
    return Signature(
        (TypeParam(t), TypeParam(r)),
        (Param("x", App("CanAdd", (Name(t), Name(r)))), Param("y", Name(t))),
        Name(r),
    )


def test_same_up_to_typevar_names() -> None:
    assert _same([_sig("T", "R")], [_sig("A", "B")])
    assert _same([_sig("T", "R"), _sig("A", "B")], [_sig("U", "V")])
    assert not _same([_sig("T", "R")], [])
    concrete = Signature((), (Param("x", Name("int")),), Name("int"))
    assert not _same([_sig("T", "R")], [concrete])


def test_distinct_keeps_the_first_of_alike_renders_in_order() -> None:
    other = Signature((TypeParam("T"),), (Param("x", Name("T")),), Name("T"))
    assert _distinct([_sig("T", "R"), other, _sig("A", "B")]) == [_sig("T", "R"), other]
