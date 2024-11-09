import pytest

from optype.array_api import _sctypes_numpy as sct


@pytest.mark.parametrize("v", [False, 0, 0.0, 0j])
def test_no_builtins(v: complex) -> None:
    b: sct.Bool = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8: sct.Float64 = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16: sct.Complex128 = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert not isinstance(v, sct.Bool)
    assert not isinstance(v, sct.Float64)
    assert not isinstance(v, sct.Complex128)

    iu: sct.Integer = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f: sct.Floating = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c: sct.ComplexFloating = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert not isinstance(v, sct.Integer)
    assert not isinstance(v, sct.Floating)
    assert not isinstance(v, sct.ComplexFloating)

    iuf: sct.RealNumber = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    iufc: sct.Number = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert not isinstance(v, sct.RealNumber)
    assert not isinstance(v, sct.Number)

    biu: sct.Integer_co = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    biuf: sct.Floating_co = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    biufc: sct.ComplexFloating_co = v  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert not isinstance(v, sct.Integer_co)
    assert not isinstance(v, sct.Floating_co)
    assert not isinstance(v, sct.ComplexFloating_co)
