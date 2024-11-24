# ruff: noqa: PYI015
import numpy as np

import optype.numpy as onp

vec_b: onp.Array1D[np.bool] | list[bool | np.bool_]
vec_i: onp.Array1D[np.intp] | list[int | np.intp] | range
vec_f: onp.Array1D[np.longdouble] | list[float | np.longdouble] | range
vec_c: onp.Array1D[np.clongdouble] | list[complex | np.clongdouble] | range

mat_b: onp.Array2D[np.bool] | list[list[bool | np.bool_]]
mat_i: onp.Array2D[np.intp] | list[list[int | np.intp]]
mat_f: onp.Array2D[np.longdouble] | list[list[float | np.longdouble]]
mat_c: onp.Array2D[np.clongdouble] | list[list[complex | np.clongdouble]]

arr_b: onp.ArrayND[np.bool] | list[list[list[bool | np.bool_]]]
arr_i: onp.ArrayND[np.intp] | list[list[list[int | np.intp]]]
arr_f: onp.ArrayND[np.longdouble] | list[list[list[float | np.longdouble]]]
arr_c: onp.ArrayND[np.clongdouble] | list[list[list[complex | np.clongdouble]]]

# scalars

sb0: onp.ToBool = True
sb1: onp.ToBool = np.True_

si0: onp.ToInt = 42
si1: onp.ToInt = np.int_(42)

sf0: onp.ToFloat = 42.0
sf1: onp.ToFloat = np.longdouble(42.0)

sc0: onp.ToComplex = 42 + 0j
sc1: onp.ToComplex = np.clongdouble(42.0 + 1j)

# vectors
vb: onp.ToBool1D = vec_b
vi: onp.ToInt1D = vec_i
vf: onp.ToFloat1D = vec_f
vc: onp.ToComplex1D = vec_c

vsb: onp.ToBool1D = True  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
vsi: onp.ToInt1D = 42  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
vsf: onp.ToFloat1D = 42.0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
vsc: onp.ToComplex1D = 42 + 1j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

vub: onp.ToBool1D = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
vui: onp.ToInt1D = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
vuf: onp.ToFloat1D = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
vuc: onp.ToComplex1D = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

# matrices
mb: onp.ToBool2D = mat_b
mi: onp.ToInt2D = mat_i
mf: onp.ToFloat2D = mat_f
mc: onp.ToComplex2D = mat_c

mvb: onp.ToBool2D = vec_b  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
mvi: onp.ToInt2D = vec_i  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
mvf: onp.ToFloat2D = vec_f  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
mvc: onp.ToComplex2D = vec_c  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

msb: onp.ToBool2D = True  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
msi: onp.ToInt2D = 42  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
msf: onp.ToFloat2D = 42.0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
msc: onp.ToComplex2D = 42 + 0j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

mub: onp.ToBool2D = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
mui: onp.ToInt2D = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
muf: onp.ToFloat2D = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
muc: onp.ToComplex2D = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

# tensors
tb: onp.ToBoolND = arr_b
ti: onp.ToIntND = arr_i
tf: onp.ToFloatND = arr_f
tc: onp.ToComplexND = arr_c

tmb: onp.ToBoolND = mat_b
tmi: onp.ToIntND = mat_i
tmf: onp.ToFloatND = mat_f
tmc: onp.ToComplexND = mat_c

tvb: onp.ToBoolND = vec_b
tvi: onp.ToIntND = vec_i
tvf: onp.ToFloatND = vec_f
tvc: onp.ToComplexND = vec_c

tsb: onp.ToBoolND = True  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
tsi: onp.ToIntND = 42  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
tsf: onp.ToFloatND = 42.0  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
tsc: onp.ToComplexND = 42 + 0j  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

tub: onp.ToBoolND = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
tui: onp.ToIntND = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
tuf: onp.ToFloatND = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
tuc: onp.ToComplexND = "illegal"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
