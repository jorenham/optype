# ruff: noqa: PYI015
import numpy as np

import optype.numpy as onp

vec_b: onp.Array1D[np.bool] | list[bool | np.bool_]
vec_i: onp.Array1D[np.intp] | list[int | np.intp]
vec_f: onp.Array1D[np.longdouble] | list[float | np.longdouble]
vec_c: onp.Array1D[np.clongdouble] | list[complex | np.clongdouble]

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

sc0: onp.ToComplex = 42.0 + 1j
sc1: onp.ToComplex = np.clongdouble(42.0 + 1j)

# vectors

vb: onp.ToBool1D = vec_b
vi: onp.ToInt1D = vec_i
vf: onp.ToFloat1D = vec_f
vc: onp.ToComplex1D = vec_c

# matrices

mb: onp.ToBool2D = mat_b
mi: onp.ToInt2D = mat_i
mf: onp.ToFloat2D = mat_f
mc: onp.ToComplex2D = mat_c

# tensors

tb: onp.ToBoolND = arr_b
ti: onp.ToIntND = arr_i
tf: onp.ToFloatND = arr_f
tc: onp.ToComplexND = arr_c
