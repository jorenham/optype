# Type Aliases

Pre-defined type aliases for common NumPy array and dtype patterns.

## Overview

`optype.numpy` provides convenient type aliases for arrays and dtypes organized by scalar type category. These aliases are more specific than `numpy.typing.ArrayLike` and don't accept bare scalar types (which aligns with NumPy 2+ behavior).

## Array Type Aliases

### `Array`, `ArrayND`, `Array0D-3D`

Generic array type aliases with optional shape and scalar type parameters:

```python
type Array[
    ND: tuple[int, ...] = (int, ...),
    SCT: np.generic = np.generic,
] = np.ndarray[ND, np.dtype[SCT]]

type ArrayND[
    SCT: np.generic = np.generic,
    ND: tuple[int, ...] = (int, ...),
] = np.ndarray[ND, np.dtype[SCT]]

type Array0D[SCT: np.generic = np.generic] = np.ndarray[tuple[()], np.dtype[SCT]]
type Array1D[SCT: np.generic = np.generic] = np.ndarray[tuple[int], np.dtype[SCT]]
type Array2D[SCT: np.generic = np.generic] = np.ndarray[tuple[int, int], np.dtype[SCT]]
type Array3D[SCT: np.generic = np.generic] = np.ndarray[tuple[int, int, int], np.dtype[SCT]]
```

### Other Array Type Aliases

- **`MArray`, `MArray0D-3D`**: Masked arrays (`np.ma.MaskedArray`)
- **`Matrix`**: Matrix type (`np.matrix`)

## Any*Array and Any*DType

Type aliases for arrays coercible to NumPy arrays with specific dtypes:

### Supported Categories

Arrays accepting different scalar types:

| Category         | Array Types                                    | DType Types                                    |
| ---------------- | ---------------------------------------------- | ---------------------------------------------- |
| **Boolean**      | `AnyJustBoolArray`, `AnyBoolArray`, etc.       | `AnyJustBoolDType`, `AnyBoolDType`             |
| **Unsigned Int** | `AnyJustUIntArray`, `AnyJustUInt8Array`, etc.  | `AnyJustUIntDType`, `AnyJustUInt8DType`        |
| **Signed Int**   | `AnyJustIntArray`, `AnyJustInt8Array`, etc.    | `AnyJustIntDType`, `AnyJustInt8DType`          |
| **Floats**       | `AnyJustFloatArray`, `AnyFloat32Array`, etc.   | `AnyJustFloatDType`, `AnyFloat32DType`         |
| **Complex**      | `AnyJustComplexArray`, `AnyComplexArray`       | `AnyJustComplexDType`, `AnyComplexDType`       |
| **Flexible**     | `AnyStrArray`, `AnyBytesArray`, `AnyVoidArray` | `AnyStrDType`, `AnyBytesDType`, `AnyVoidDType` |
| **All**          | `AnyArray`                                     | `AnyDType`                                     |

### Strict Variants

For each category, there are also strict variants (ending with `Strict`) that only accept arrays of that specific dimensionality:

- `AnyBoolArray1D` vs `AnyBoolStrict1D`
- `AnyJustIntArray2D` vs `AnyJustIntStrict2D`

## Key Differences from numpy.typing

Unlike `numpy.typing.ArrayLike`:

- ✅ Arrays of zero dimensions are accepted
- ❌ Bare scalars (e.g., `3.14`, `True`) are **not** accepted
- ✅ More type-safe and precise

### Example

```python
import optype.numpy as onp
```

```python
import numpy as np
import numpy.typing as npt

# numpy.typing accepts bare scalars
arr_np: npt.ArrayLike = 3.14  # ✓ Accepted

# optype.numpy requires array form
arr_op: onp.AnyArray = 3.14   # ✗ Rejected
arr_op: onp.AnyArray = np.array(3.14)  # ✓ Accepted

# Both accept nested sequences
matrix_np: npt.ArrayLike = [[1, 2], [3, 4]]
matrix_op: onp.AnyArray = [[1, 2], [3, 4]]
```

## Type Parameter Information

- **Shape (`ND`)**: `tuple[int, ...]` - Array shape
- **Scalar Type (`SCT`)**: Covariant - NumPy scalar type
- **Data Type (`DT`)**: `np.dtype[SCT]`

## Usage Examples

```python
import numpy as np

# Specific scalar type
floats: onp.AnyFloat64Array = np.array([1.0, 2.0, 3.0])

# Specific dimensionality
matrix: onp.AnyJustIntArray2D = np.array([[1, 2], [3, 4]])

# Generic array
any_arr: onp.AnyArray = np.arange(10)

# Working with dtypes
dtype_f32: onp.AnyFloat32DType = np.dtype(np.float32)
dtype_int: onp.AnyJustIntDType = np.dtype(np.int64)
```

## Platform and Version Specific Notes

!!! info "Integer Type Aliases"

- Since NumPy 2, `np.uint` and `np.int_` are aliases for `np.uintp` and `np.intp`, respectively[^2].
- On unix-based platforms, `np.[u]intc` are aliases for `np.[u]int32`[^3].
- On NumPy 1, `np.uint` and `np.int_` are what in NumPy 2 are now the `np.ulong` and `np.long` types, respectively[^4].

!!! info "Floating Point Type Aliases"

- Depending on the platform, `np.longdouble` is (almost always) an alias for **either** `float128`, `float96`, or (sometimes) `float64`[^5].
- Depending on the platform, `np.clongdouble` is (almost always) an alias for **either** `complex256`, `complex192`, or (sometimes) `complex128`[^6].

!!! info "Other Type Notes"

- Since NumPy 2, `np.bool` is preferred over `np.bool_`, which only exists for backwards compatibility[^7].
- At runtime `np.timedelta64` is a subclass of `np.signedinteger`, but this is currently not reflected in the type annotations[^8].
- The `np.dtypes.StringDType` has no associated numpy scalar type, and its `.type` attribute returns the `str` builtin[^9].

## Related Types

- **[Shape Typing](shape.md)**: For precise shape type annotations
- **[DType](dtype.md)**: For dtype-specific utilities
- **[Scalar](scalar.md)**: For NumPy scalar type annotations
- **[Array-likes](array-likes.md)**: For array-like protocol objects

[^2]: Since NumPy 2, `np.uint` and `np.int_` are aliases for `np.uintp` and `np.intp`, respectively.

[^3]: On unix-based platforms `np.[u]intc` are aliases for `np.[u]int32`.

[^4]: On NumPy 1 `np.uint` and `np.int_` are what in NumPy 2 are now the `np.ulong` and `np.long` types, respectively.

[^5]: Depending on the platform, `np.longdouble` is (almost always) an alias for **either** `float128`, `float96`, or (sometimes) `float64`.

[^6]: Depending on the platform, `np.clongdouble` is (almost always) an alias for **either** `complex256`, `complex192`, or (sometimes) `complex128`.

[^7]: Since NumPy 2, `np.bool` is preferred over `np.bool_`, which only exists for backwards compatibility.

[^8]: At runtime `np.timedelta64` is a subclass of `np.signedinteger`, but this is currently not reflected in the type annotations.

[^9]: The `np.dtypes.StringDType` has no associated numpy scalar type, and its `.type` attribute returns the `str` builtin.
