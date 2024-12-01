from collections.abc import Sequence as Seq
from typing import Any, TypeAlias

import numpy as np

import optype.numpy as onp

# setup

_Scalar_b: TypeAlias = np.bool_
_Scalar_j: TypeAlias = np.integer[Any]  # pyright: ignore[reportExplicitAny]
_Scalar_i: TypeAlias = _Scalar_j | _Scalar_b
_Scalar_f: TypeAlias = np.floating[Any] | _Scalar_i  # pyright: ignore[reportExplicitAny]
_Scalar_c: TypeAlias = np.number[Any] | _Scalar_b  # pyright: ignore[reportExplicitAny]
_Scalar_x: TypeAlias = np.generic

_Value_b: TypeAlias = _Scalar_b | bool
_Value_j: TypeAlias = _Scalar_j | int
_Value_i: TypeAlias = _Scalar_i | int
_Value_f: TypeAlias = _Scalar_f | float
_Value_c: TypeAlias = _Scalar_c | complex
_Value_x: TypeAlias = _Scalar_x | complex | bytes | str

_Array0_b: TypeAlias = onp.CanArray0D[_Scalar_b]
_Array0_j: TypeAlias = onp.CanArray0D[_Scalar_j]
_Array0_i: TypeAlias = onp.CanArray0D[_Scalar_i]
_Array0_f: TypeAlias = onp.CanArray0D[_Scalar_f]
_Array0_c: TypeAlias = onp.CanArray0D[_Scalar_c]
_Array0_x: TypeAlias = onp.CanArray0D[_Scalar_x]

_Array1_b: TypeAlias = onp.CanArray1D[_Scalar_b] | Seq[_Array0_b | _Value_b]
_Array1_j: TypeAlias = onp.CanArray1D[_Scalar_j] | Seq[_Array0_j | _Value_j]
_Array1_i: TypeAlias = onp.CanArray1D[_Scalar_i] | Seq[_Array0_i | _Value_i]
_Array1_f: TypeAlias = onp.CanArray1D[_Scalar_f] | Seq[_Array0_f | _Value_f]
_Array1_c: TypeAlias = onp.CanArray1D[_Scalar_c] | Seq[_Array0_c | _Value_c]
_Array1_x: TypeAlias = onp.CanArray1D[_Scalar_x] | Seq[_Array0_x | _Value_x]

_Array2_b: TypeAlias = onp.CanArray2D[_Scalar_b] | Seq[_Array1_b]
_Array2_j: TypeAlias = onp.CanArray2D[_Scalar_j] | Seq[_Array1_j]
_Array2_i: TypeAlias = onp.CanArray2D[_Scalar_i] | Seq[_Array1_i]
_Array2_f: TypeAlias = onp.CanArray2D[_Scalar_f] | Seq[_Array1_f]
_Array2_c: TypeAlias = onp.CanArray2D[_Scalar_c] | Seq[_Array1_c]
_Array2_x: TypeAlias = onp.CanArray2D[_Scalar_x] | Seq[_Array1_x]

_Array3_b: TypeAlias = onp.CanArray3D[_Scalar_b] | Seq[_Array2_b]
_Array3_j: TypeAlias = onp.CanArray3D[_Scalar_j] | Seq[_Array2_j]
_Array3_i: TypeAlias = onp.CanArray3D[_Scalar_i] | Seq[_Array2_i]
_Array3_f: TypeAlias = onp.CanArray3D[_Scalar_f] | Seq[_Array2_f]
_Array3_c: TypeAlias = onp.CanArray3D[_Scalar_c] | Seq[_Array2_c]
_Array3_x: TypeAlias = onp.CanArray3D[_Scalar_x] | Seq[_Array2_x]

b_: _Value_b
j_: _Value_j
i_: _Value_i
f_: _Value_f
c_: _Value_c
x_: _Value_x

b0: _Array0_b | _Value_b
j0: _Array0_j | _Value_j
i0: _Array0_i | _Value_i
f0: _Array0_f | _Value_f
c0: _Array0_c | _Value_c
x0: _Array0_x | _Value_x

b1: _Array1_b
j1: _Array1_j
i1: _Array1_i
f1: _Array1_f
c1: _Array1_c
x1: _Array1_x

b2: _Array2_b
j2: _Array2_j
i2: _Array2_i
f2: _Array2_f
c2: _Array2_c
x2: _Array2_x

b3: _Array3_b
j3: _Array3_j
i3: _Array3_i
f3: _Array3_f
c3: _Array3_c
x3: _Array3_x

# scalar

# scalar :> scalar
b__b_: onp.ToBool = b_
j__j_: onp.ToJustInt = j_
j__b_: onp.ToJustInt = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
j__i_: onp.ToJustInt = i_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
i__i_: onp.ToInt = i_
f__f_: onp.ToFloat = f_
c__c_: onp.ToComplex = c_
x__x_: onp.ToScalar = x_
# scalar :> 0-d (negative)
b__b0: onp.ToBool = b0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
j__j0: onp.ToJustInt = j0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
i__i0: onp.ToInt = i0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
f__f0: onp.ToFloat = f0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
c__c0: onp.ToComplex = c0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
x__x0: onp.ToScalar = x0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

# 1-d :> scalar (negative)
b1_b_: onp.ToBool1D = b_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
j1_j_: onp.ToJustInt1D = j_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
i1_i_: onp.ToInt1D = i_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
f1_f_: onp.ToFloat1D = f_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
c1_c_: onp.ToComplex1D = c_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
x1_x_: onp.ToArray1D = x_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
# 1-d :> 1-d
b1_b1: onp.ToBool1D = b1
j1_j1: onp.ToJustInt1D = j1
i1_i1: onp.ToInt1D = i1
f1_f1: onp.ToFloat1D = f1
c1_c1: onp.ToComplex1D = c1
x1_x1: onp.ToArray1D = x1
# 1-d :> 1-d [strict]
b1s_b1: onp.ToBoolStrict1D = b1
j1s_j1: onp.ToJustIntStrict1D = j1
i1s_i1: onp.ToIntStrict1D = i1
f1s_f1: onp.ToFloatStrict1D = f1
c1s_c1: onp.ToComplexStrict1D = c1
x1s_x1: onp.ToArrayStrict1D = x1

# 2-d :> 1-d (negative)
b2_b1: onp.ToBool2D = b1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
j2_j1: onp.ToJustInt2D = j1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
i2_i1: onp.ToInt2D = i1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
f2_f1: onp.ToFloat2D = f1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
c2_c1: onp.ToComplex2D = c1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
x2_x1: onp.ToArray2D = x1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
# 2-d :> 2-d
b2_b2: onp.ToBool2D = b2
j2_j2: onp.ToJustInt2D = j2
i2_i2: onp.ToInt2D = i2
f2_f2: onp.ToFloat2D = f2
c2_c2: onp.ToComplex2D = c2
x2_x2: onp.ToArray2D = x2
# 2-d :> 2-d [strict]
b2s_b2: onp.ToBoolStrict2D = b2
j2s_j2: onp.ToJustIntStrict2D = j2
i2s_i2: onp.ToIntStrict2D = i2
f2s_f2: onp.ToFloatStrict2D = f2
c2s_c2: onp.ToComplexStrict2D = c2
x2s_x2: onp.ToArrayStrict2D = x2

# 3-d :> 2-d (negative)
b3_b2: onp.ToBool3D = b2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
j3_j2: onp.ToJustInt3D = j2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
i3_i2: onp.ToInt3D = i2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
f3_f2: onp.ToFloat3D = f2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
c3_c2: onp.ToComplex3D = c2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
x3_x2: onp.ToArray3D = x2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
# 3-d :> 3-d
b3_b3: onp.ToBool3D = b3
j3_j3: onp.ToJustInt3D = j3
i3_i3: onp.ToInt3D = i3
f3_f3: onp.ToFloat3D = f3
c3_c3: onp.ToComplex3D = c3
x3_x3: onp.ToArray3D = x3
# 3-d :> 3-d [strict]
b3s_b3: onp.ToBoolStrict3D = b3
j3s_j3: onp.ToJustIntStrict3D = j3
i3s_i3: onp.ToIntStrict3D = i3
f3s_f3: onp.ToFloatStrict3D = f3
c3s_c3: onp.ToComplexStrict3D = c3
x3s_x3: onp.ToArrayStrict3D = x3
