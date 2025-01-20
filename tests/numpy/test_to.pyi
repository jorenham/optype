# ruff: noqa: PYI048

from collections.abc import Sequence as Seq
from typing import Any, TypeAlias

import numpy as np
from numpy._typing import _64Bit, _96Bit, _128Bit

import optype.numpy as onp

__all__ = ()

# setup

_NBit_g: TypeAlias = _64Bit | _96Bit | _128Bit

_Sca_x: TypeAlias = np.generic
_Sca_b: TypeAlias = np.bool_
_Sca_i: TypeAlias = np.integer[Any]
_Sca_f: TypeAlias = np.floating[_NBit_g] | np.float64 | np.float32 | np.float16
_Sca_c: TypeAlias = np.complexfloating[_NBit_g, _NBit_g] | np.complex128 | np.complex64
_Sca_f8: TypeAlias = np.float64
_Sca_c16: TypeAlias = np.complex128
_Sca_i_co: TypeAlias = _Sca_i | _Sca_b
_Sca_f_co: TypeAlias = _Sca_f | _Sca_i_co
_Sca_c_co: TypeAlias = _Sca_c | _Sca_f_co
_Sca_f8_co: TypeAlias = _Sca_f8 | np.float32 | np.float16 | _Sca_i_co
_Sca_c16_co: TypeAlias = _Sca_c16 | np.complex64 | _Sca_f8_co

_Val_x: TypeAlias = _Sca_x | complex | bytes | str
_Val_b: TypeAlias = _Sca_b | bool
_Val_i: TypeAlias = _Sca_i | int
_Val_f: TypeAlias = _Sca_f | float
_Val_c: TypeAlias = _Sca_c | complex
_Val_f8: TypeAlias = _Sca_f8 | float
_Val_c16: TypeAlias = _Sca_c16 | complex
_Val_i_co: TypeAlias = _Sca_i_co | int
_Val_f_co: TypeAlias = _Sca_f_co | float
_Val_c_co: TypeAlias = _Sca_c_co | complex
_Val_f8_co: TypeAlias = _Sca_f8_co | float
_Val_c16_co: TypeAlias = _Sca_c16_co | complex

_Arr0_x_co: TypeAlias = onp.CanArray0D[_Sca_x]
_Arr0_b: TypeAlias = onp.CanArray0D[_Sca_b]
_Arr0_i: TypeAlias = onp.CanArray0D[_Sca_i]
_Arr0_f: TypeAlias = onp.CanArray0D[_Sca_f]
_Arr0_c: TypeAlias = onp.CanArray0D[_Sca_c]
_Arr0_f8: TypeAlias = onp.CanArray0D[_Sca_f8]
_Arr0_c16: TypeAlias = onp.CanArray0D[_Sca_c16]
_Arr0_i_co: TypeAlias = onp.CanArray0D[_Sca_i_co]
_Arr0_f_co: TypeAlias = onp.CanArray0D[_Sca_f_co]
_Arr0_c_co: TypeAlias = onp.CanArray0D[_Sca_c_co]
_Arr0_f8_co: TypeAlias = onp.CanArray0D[_Sca_f8_co]
_Arr0_c16_co: TypeAlias = onp.CanArray0D[_Sca_c16_co]

_Arr1_x: TypeAlias = onp.CanArray1D[_Sca_x] | Seq[_Arr0_x_co | _Val_x]
_Arr1_b: TypeAlias = onp.CanArray1D[_Sca_b] | Seq[_Arr0_b | _Val_b]
_Arr1_i: TypeAlias = onp.CanArray1D[_Sca_i] | Seq[_Arr0_i | _Val_i]
_Arr1_f: TypeAlias = onp.CanArray1D[_Sca_f] | Seq[_Arr0_f | _Val_f]
_Arr1_c: TypeAlias = onp.CanArray1D[_Sca_c] | Seq[_Arr0_c | _Val_c]
_Arr1_f8: TypeAlias = onp.CanArray1D[_Sca_f8] | Seq[_Arr0_f8 | _Val_f8]
_Arr1_c16: TypeAlias = onp.CanArray1D[_Sca_c16] | Seq[_Arr0_c16 | _Val_c16]
_Arr1_i_co: TypeAlias = onp.CanArray1D[_Sca_i_co] | Seq[_Arr0_i_co | _Val_i_co]
_Arr1_f_co: TypeAlias = onp.CanArray1D[_Sca_f_co] | Seq[_Arr0_f_co | _Val_f_co]
_Arr1_c_co: TypeAlias = onp.CanArray1D[_Sca_c_co] | Seq[_Arr0_c_co | _Val_c_co]
_Arr1_f8_co: TypeAlias = onp.CanArray1D[_Sca_f8_co] | Seq[_Arr0_f8_co | _Val_f8_co]
_Arr1_c16_co: TypeAlias = onp.CanArray1D[_Sca_c16_co] | Seq[_Arr0_c16_co | _Val_c16_co]

_Arr2_x: TypeAlias = onp.CanArray2D[_Sca_x] | Seq[_Arr1_x]
_Arr2_b: TypeAlias = onp.CanArray2D[_Sca_b] | Seq[_Arr1_b]
_Arr2_i: TypeAlias = onp.CanArray2D[_Sca_i] | Seq[_Arr1_i]
_Arr2_f: TypeAlias = onp.CanArray2D[_Sca_f] | Seq[_Arr1_f]
_Arr2_c: TypeAlias = onp.CanArray2D[_Sca_c] | Seq[_Arr1_c]
_Arr2_f8: TypeAlias = onp.CanArray2D[_Sca_f8] | Seq[_Arr1_f8]
_Arr2_c16: TypeAlias = onp.CanArray2D[_Sca_c16] | Seq[_Arr1_c16]
_Arr2_i_co: TypeAlias = onp.CanArray2D[_Sca_i_co] | Seq[_Arr1_i_co]
_Arr2_f_co: TypeAlias = onp.CanArray2D[_Sca_f_co] | Seq[_Arr1_f_co]
_Arr2_c_co: TypeAlias = onp.CanArray2D[_Sca_c_co] | Seq[_Arr1_c_co]
_Arr2_f8_co: TypeAlias = onp.CanArray2D[_Sca_f8_co] | Seq[_Arr1_f8_co]
_Arr2_c16_co: TypeAlias = onp.CanArray2D[_Sca_c16_co] | Seq[_Arr1_c16_co]

_Arr3_x: TypeAlias = onp.CanArray3D[_Sca_x] | Seq[_Arr2_x]
_Arr3_b: TypeAlias = onp.CanArray3D[_Sca_b] | Seq[_Arr2_b]
_Arr3_i: TypeAlias = onp.CanArray3D[_Sca_i] | Seq[_Arr2_i]
_Arr3_f: TypeAlias = onp.CanArray3D[_Sca_f] | Seq[_Arr2_f]
_Arr3_c: TypeAlias = onp.CanArray3D[_Sca_c] | Seq[_Arr2_c]
_Arr3_f8: TypeAlias = onp.CanArray3D[_Sca_f8] | Seq[_Arr2_f8]
_Arr3_c16: TypeAlias = onp.CanArray3D[_Sca_c16] | Seq[_Arr2_c16]
_Arr3_i_co: TypeAlias = onp.CanArray3D[_Sca_i_co] | Seq[_Arr2_i_co]
_Arr3_f_co: TypeAlias = onp.CanArray3D[_Sca_f_co] | Seq[_Arr2_f_co]
_Arr3_c_co: TypeAlias = onp.CanArray3D[_Sca_c_co] | Seq[_Arr2_c_co]
_Arr3_f8_co: TypeAlias = onp.CanArray3D[_Sca_f8_co] | Seq[_Arr2_f8_co]
_Arr3_c16_co: TypeAlias = onp.CanArray3D[_Sca_c16_co] | Seq[_Arr2_c16_co]

x_: _Val_x
b_: _Val_b
i_: _Val_i
f_: _Val_f
c_: _Val_c
f8: _Val_f8
c16: _Val_c16
i_co: _Val_i_co
f_co: _Val_f_co
c_co: _Val_c_co
f8_co: _Val_f8_co
c16_co: _Val_c16_co

x_0d: _Arr0_x_co | _Val_x
b_0d: _Arr0_b | _Val_b
i_0d: _Arr0_i | _Val_i
f_0d: _Arr0_f | _Val_f
c_0d: _Arr0_c | _Val_c
f8_0d: _Arr0_f8 | _Val_f8
c16_0d: _Arr0_c16 | _Val_c16
i_co_0d: _Arr0_i_co | _Val_i_co
f_co_0d: _Arr0_f_co | _Val_f_co
c_co_0d: _Arr0_c_co | _Val_c_co
f8_co_0d: _Arr0_f8_co | _Val_f8_co
c16_co_0d: _Arr0_c16_co | _Val_c16_co

x_1d: _Arr1_x
b_1d: _Arr1_b
i_1d: _Arr1_i
f_1d: _Arr1_f
c_1d: _Arr1_c
f8_1d: _Arr1_f8
c16_1d: _Arr1_c16
i_co_1d: _Arr1_i_co
f_co_1d: _Arr1_f_co
c_co_1d: _Arr1_c_co
f8_co_1d: _Arr1_f8_co
c16_co_1d: _Arr1_c16_co

x_2d: _Arr2_x
b_2d: _Arr2_b
i_2d: _Arr2_i
f_2d: _Arr2_f
c_2d: _Arr2_c
f8_2d: _Arr2_f8
c16_2d: _Arr2_c16
i_co_2d: _Arr2_i_co
f_co_2d: _Arr2_f_co
c_co_2d: _Arr2_c_co
f8_co_2d: _Arr2_f8_co
c16_co_2d: _Arr2_c16_co

b_3d: _Arr3_b
x_3d: _Arr3_x
i_3d: _Arr3_i
f_3d: _Arr3_f
c_3d: _Arr3_c
f8_3d: _Arr3_f8
c16_3d: _Arr3_c16
i_co_3d: _Arr3_i_co
f_co_3d: _Arr3_f_co
c_co_3d: _Arr3_c_co
f8_co_3d: _Arr3_f8_co
c16_co_3d: _Arr3_c16_co

# scalar

def sca_sca() -> None:
    x__x: onp.ToScalar = x_
    b__b: onp.ToBool = b_

    i__i: onp.ToJustInt = i_
    i__b: onp.ToJustInt = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i__i_co: onp.ToJustInt = i_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    f8__f8: onp.ToJustFloat64 = f8
    f8__b: onp.ToJustFloat64 = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8__i: onp.ToJustFloat64 = i_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8__f: onp.ToJustFloat64 = f_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8__f_co: onp.ToJustFloat64 = f_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    f__f: onp.ToJustFloat = f_
    f__b: onp.ToJustFloat = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f__i: onp.ToJustFloat = i_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f__f_co: onp.ToJustFloat = f_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    c16__c16: onp.ToJustComplex128 = c16
    c16__b: onp.ToJustComplex128 = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16__i: onp.ToJustComplex128 = i_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16__f: onp.ToJustComplex128 = f_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16__c: onp.ToJustComplex128 = c_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16__c_co: onp.ToJustComplex128 = c_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    c__c: onp.ToJustComplex = c_
    c__b: onp.ToJustComplex = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c__i: onp.ToJustComplex = i_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c__f: onp.ToJustComplex = f_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c__c_co: onp.ToJustComplex = c_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    i_co__i_co: onp.ToInt = i_co
    f_co__f_co: onp.ToFloat = f_co
    c_co__c_co: onp.ToComplex = c_co

def sca_a0d() -> None:
    x__x: onp.ToScalar = x_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    b__b: onp.ToBool = b_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i__i: onp.ToJustInt = i_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f__f: onp.ToJustFloat = f_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c__c: onp.ToJustComplex = c_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8__f8: onp.ToJustFloat64 = f8_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16__c16: onp.ToJustComplex128 = c16_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i_co__i_co: onp.ToInt = i_co_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f_co__f_co: onp.ToFloat = f_co_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c_co__c_co: onp.ToComplex = c_co_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8_co__f8_co: onp.ToFloat64 = f8_co_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_co__c16_co: onp.ToComplex128 = c16_co_0d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

# 1-d

def a1d_sca() -> None:
    x__x: onp.ToArray1D = x_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    b__b: onp.ToBool1D = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i__i: onp.ToJustInt1D = i_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f__f: onp.ToJustFloat1D = f_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c__c: onp.ToJustComplex1D = c_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8__f8: onp.ToJustFloat64_1D = f8  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16__c16: onp.ToJustComplex128_1D = c16  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i_co__i_co: onp.ToInt1D = i_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f_co__f_co: onp.ToFloat1D = f_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c_co__c_co: onp.ToComplex1D = c_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8_co__f8_co: onp.ToFloat64_1D = f8_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_co__c16_co: onp.ToComplex128_1D = c16_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

def a1d_a1d() -> None:
    x__x: onp.ToArray1D = x_1d
    b__b: onp.ToBool1D = b_1d
    i__i: onp.ToJustInt1D = i_1d
    f__f: onp.ToJustFloat1D = f_1d
    c__c: onp.ToJustComplex1D = c_1d
    f8__f8: onp.ToJustFloat64_1D = f8_1d
    c16__c16: onp.ToJustComplex128_1D = c16_1d
    i_co__i: onp.ToInt1D = i_co_1d
    f_co__f: onp.ToFloat1D = f_co_1d
    c_co__c: onp.ToComplex1D = c_co_1d
    f8_co__f8: onp.ToFloat64_1D = f8_co_1d
    c16_co__c16: onp.ToComplex128_1D = c16_co_1d

def s1d_a1d() -> None:
    x__x: onp.ToArrayStrict1D = x_1d
    b__b: onp.ToBoolStrict1D = b_1d
    i__i: onp.ToJustIntStrict1D = i_1d
    f__f: onp.ToJustFloatStrict1D = f_1d
    c__c: onp.ToJustComplexStrict1D = c_1d
    f8__f8: onp.ToJustFloat64Strict1D = f8_1d
    c16__c16: onp.ToJustComplex128Strict1D = c16_1d
    i_co__i: onp.ToIntStrict1D = i_co_1d
    f_co__f: onp.ToFloatStrict1D = f_co_1d
    c_co__c: onp.ToComplexStrict1D = c_co_1d
    f8_co__f8: onp.ToFloat64Strict1D = f8_co_1d
    c16_co__c16: onp.ToComplex128Strict1D = c16_co_1d

# 2-d

def a2d_sca() -> None:
    x__x: onp.ToArray2D = x_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    b__b: onp.ToBool2D = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i__i: onp.ToJustInt2D = i_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f__f: onp.ToJustFloat2D = f_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c__c: onp.ToJustComplex2D = c_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8__f8: onp.ToJustFloat64_2D = f8  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16__c16: onp.ToJustComplex128_2D = c16  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i_co__i_co: onp.ToInt2D = i_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f_co__f_co: onp.ToFloat2D = f_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c_co__c_co: onp.ToComplex2D = c_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8_co__f8_co: onp.ToFloat64_2D = f8_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_co__c16_co: onp.ToComplex128_2D = c16_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

def a2d_a2d() -> None:
    x__x: onp.ToArray2D = x_2d
    b__b: onp.ToBool2D = b_2d
    i__i: onp.ToJustInt2D = i_2d
    f__f: onp.ToJustFloat2D = f_2d
    c__c: onp.ToJustComplex2D = c_2d
    f8__f8: onp.ToJustFloat64_2D = f8_2d
    c16__c16: onp.ToJustComplex128_2D = c16_2d
    i_co__i: onp.ToInt2D = i_co_2d
    f_co__f: onp.ToFloat2D = f_co_2d
    c_co__c: onp.ToComplex2D = c_co_2d
    f8_co__f8: onp.ToFloat64_2D = f8_co_2d
    c16_co__c16: onp.ToComplex128_2D = c16_co_2d
    f8_co__f: onp.ToFloat64_2D = f_co_2d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_co__c: onp.ToComplex128_2D = c_co_2d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

def s2d_a2d() -> None:
    x__x: onp.ToArrayStrict2D = x_2d
    b__b: onp.ToBoolStrict2D = b_2d
    i__i: onp.ToJustIntStrict2D = i_2d
    f__f: onp.ToJustFloatStrict2D = f_2d
    c__c: onp.ToJustComplexStrict2D = c_2d
    f8__f8: onp.ToJustFloat64Strict2D = f8_2d
    c16__c16: onp.ToJustComplex128Strict2D = c16_2d
    i_co__i: onp.ToIntStrict2D = i_co_2d
    f_co__f: onp.ToFloatStrict2D = f_co_2d
    c_co__c: onp.ToComplexStrict2D = c_co_2d
    f8_co__f8: onp.ToFloat64Strict2D = f8_co_2d
    c16_co__c16: onp.ToComplex128Strict2D = c16_co_2d
    f8_co__f: onp.ToFloat64Strict2D = f_co_2d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_co__c: onp.ToComplex128Strict2D = c_co_2d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

# 3-d

def a3d_sca() -> None:
    x__x: onp.ToArray3D = x_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    b__b: onp.ToBool3D = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i__i: onp.ToJustInt3D = i_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f__f: onp.ToJustFloat3D = f_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c__c: onp.ToJustComplex3D = c_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8__f8: onp.ToJustFloat64_3D = f8  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16__c16: onp.ToJustComplex128_3D = c16  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    i_co__i_co: onp.ToInt3D = i_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f_co__f_co: onp.ToFloat3D = f_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c_co__c_co: onp.ToComplex3D = c_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    f8_co__f8_co: onp.ToFloat64_3D = f8_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_co__c16_co: onp.ToComplex128_3D = c16_co  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

def a3d_a3d() -> None:
    x__x: onp.ToArray3D = x_3d
    b__b: onp.ToBool3D = b_3d
    i__i: onp.ToJustInt3D = i_3d
    f__f: onp.ToJustFloat3D = f_3d
    c__c: onp.ToJustComplex3D = c_3d
    f8__f8: onp.ToJustFloat64_3D = f8_3d
    c16__c16: onp.ToJustComplex128_3D = c16_3d
    i_co__i: onp.ToInt3D = i_co_3d
    f_co__f: onp.ToFloat3D = f_co_3d
    c_co__c: onp.ToComplex3D = c_co_3d
    f8_co__f8: onp.ToFloat64_3D = f8_co_3d
    c16_co__c16: onp.ToComplex128_3D = c16_co_3d
    f8_co__f: onp.ToFloat64_3D = f_co_3d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_co__c: onp.ToComplex128_3D = c_co_3d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

def s3d_a3d() -> None:
    x__x: onp.ToArrayStrict3D = x_3d
    b__b: onp.ToBoolStrict3D = b_3d
    i__i: onp.ToJustIntStrict3D = i_3d
    f__f: onp.ToJustFloatStrict3D = f_3d
    c__c: onp.ToJustComplexStrict3D = c_3d
    f8__f8: onp.ToJustFloat64Strict3D = f8_3d
    c16__c16: onp.ToJustComplex128Strict3D = c16_3d
    i_co__i: onp.ToIntStrict3D = i_co_3d
    f_co__f: onp.ToFloatStrict3D = f_co_3d
    c_co__c: onp.ToComplexStrict3D = c_co_3d
    f8_co__f8: onp.ToFloat64Strict3D = f8_co_3d
    c16_co__c16: onp.ToComplex128Strict3D = c16_co_3d
    f8_co__f: onp.ToFloat64Strict3D = f_co_3d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    c16_co__c: onp.ToComplex128Strict3D = c_co_3d  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
