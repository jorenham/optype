import sys
from typing import Any

import numpy as np
import pytest

from optype.array_api import _sctypes_numpy as sct


def test_bool() -> None:
    i1_x: sct.Bool = np.int8()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    u1_x: sct.Bool = np.uint8()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f2_x: sct.Bool = np.float16()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f4_x: sct.Bool = np.float32()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8_x: sct.Bool = np.float64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c8_x: sct.Bool = np.complex64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_x: sct.Bool = np.complex128()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    v = np.bool_()
    b1: sct.Bool = v

    assert isinstance(v, sct.ComplexFloating_co)
    assert isinstance(v, sct.Floating_co)
    assert isinstance(v, sct.Integer_co)
    assert isinstance(v, sct.Bool)

    assert not isinstance(v, sct.Complex128)
    assert not isinstance(v, sct.Float64)

    assert not isinstance(v, sct.Number)
    assert not isinstance(v, sct.ComplexFloating)
    assert not isinstance(v, sct.Floating)
    assert not isinstance(v, sct.Integer)


def test_float64() -> None:
    b1_x: sct.Float64 = np.bool_()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i8_x: sct.Float64 = np.int64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    u8_x: sct.Float64 = np.uint64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c8_x: sct.Float64 = np.complex64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_x: sct.Float64 = np.complex128()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    # mypy false negatives
    f2_x: sct.Float64 = np.float16()  # pyright: ignore[reportAssignmentType]
    f4_x: sct.Float64 = np.float32()  # pyright: ignore[reportAssignmentType]

    v = np.float64()
    f8: sct.Float64 = v

    assert isinstance(v, sct.Number)
    assert isinstance(v, sct.ComplexFloating_co)
    assert isinstance(v, sct.Floating_co)
    assert isinstance(v, sct.Floating)
    assert isinstance(v, sct.Float64)

    if sys.version_info >= (3, 11):
        assert not isinstance(v, sct.Complex128)
    assert not isinstance(v, sct.Integer_co)
    assert not isinstance(v, sct.Integer)


def test_complex128() -> None:
    b1_x: sct.Complex128 = np.bool_()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f2_x: sct.Complex128 = np.float16()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f4_x: sct.Complex128 = np.float32()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8_x: sct.Complex128 = np.float64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i8_x: sct.Complex128 = np.int64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    u8_x: sct.Complex128 = np.uint64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    # mypy false negative
    c8_x: sct.Complex128 = np.complex64()  # pyright: ignore[reportAssignmentType]

    c16: sct.Complex128 = np.complex128()

    assert isinstance(c16, sct.Number)
    assert isinstance(c16, sct.ComplexFloating_co)
    assert isinstance(c16, sct.Complex128)

    assert not isinstance(c16, sct.Float64)
    assert not isinstance(c16, sct.Floating)
    assert not isinstance(c16, sct.Integer)
    assert not isinstance(c16, sct.Integer_co)
    assert not isinstance(c16, sct.RealNumber)


@pytest.mark.parametrize(
    "sctype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64],
)
def test_integer(sctype: type[np.integer[Any]]) -> None:
    b1_x: sct.Integer = np.bool_()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f2_x: sct.Integer = np.float16()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f4_x: sct.Integer = np.float32()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8_x: sct.Integer = np.float64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c8_x: sct.Integer = np.complex64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_x: sct.Integer = np.complex128()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    iu: sct.Integer = sctype()

    assert isinstance(iu, sct.Number)
    assert isinstance(iu, sct.ComplexFloating_co)
    assert isinstance(iu, sct.Floating_co)
    assert isinstance(iu, sct.Integer_co)
    assert isinstance(iu, sct.Integer)

    assert not isinstance(iu, sct.Complex128)
    assert not isinstance(iu, sct.Float64)
    assert not isinstance(iu, sct.Floating)


@pytest.mark.parametrize("sctype", [np.float16, np.float32, np.float64, np.longdouble])
def test_floating(sctype: type[np.floating[Any]]) -> None:
    b1_x: sct.Floating = np.bool_()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i8_x: sct.Floating = np.int64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    u8_x: sct.Floating = np.uint64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c8_x: sct.Floating = np.complex64()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_x: sct.Floating = np.complex128()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    f: sct.Floating = sctype()

    assert isinstance(f, sct.Number)
    assert isinstance(f, sct.RealNumber)
    assert isinstance(f, sct.ComplexFloating_co)
    assert isinstance(f, sct.Floating_co)
    assert isinstance(f, sct.Floating)

    if sys.version_info >= (3, 11):
        assert not isinstance(f, sct.Complex128)
    assert not isinstance(f, sct.Integer_co)
    assert not isinstance(f, sct.Integer)


@pytest.mark.parametrize("sctype", [np.complex64, np.complex128, np.clongdouble])
def test_complexfloating(sctype: type[np.complexfloating[Any, Any]]) -> None:
    # NOTE: The negative assignment type-tests all fail in `numpy<2.2`

    c: sct.ComplexFloating = sctype()

    assert isinstance(c, sct.ComplexFloating_co)
    assert isinstance(c, sct.ComplexFloating)
    assert isinstance(c, sct.Number)

    assert not isinstance(c, sct.Bool)
    assert not isinstance(c, sct.Integer)
    assert not isinstance(c, sct.Integer_co)
    assert not isinstance(c, sct.Float64)
    assert not isinstance(c, sct.Floating)
    assert not isinstance(c, sct.RealNumber)
