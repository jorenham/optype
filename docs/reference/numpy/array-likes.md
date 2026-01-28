# Array-likes

Protocols and type aliases for objects that can be converted to NumPy arrays.

## Overview

Array-like type aliases describe objects that NumPy functions can convert to arrays through the `__array__` protocol or direct coercion. Unlike `numpy.typing.ArrayLike`, these don't accept bare scalar types and require the `__len__` method.

## Naming Convention

Each type alias follows the pattern: `To{ScalarType}{Dimensionality}`

- **Scalar Type**: `Bool`, `Int`, `Float32`, `Float64`, `Complex128`, etc.
- **Dimensionality**: `1D`, `2D`, `3D`, or `ND` for n-dimensional

### Prefix: `ToJust*`

The `ToJust*` variants are more restrictive, accepting only exact types. They were added in optype 0.8.0.

## Scalar-Like Aliases

For bare scalar values (no array conversion required):

| Type            | Accepts                              |
| --------------- | ------------------------------------ |
| `ToFalse`       | `False` or `0` → `np.bool_`          |
| `ToJustFalse`   | Only `False` → `np.bool_`            |
| `ToTrue`        | `True` or `1` → `np.bool_`           |
| `ToJustTrue`    | Only `True` → `np.bool_`             |
| `ToInt`         | `int` or `bool`                      |
| `ToJustInt`     | Only `int` values                    |
| `ToFloat64`     | `float`, `int`, or `bool`            |
| `ToJustFloat64` | Only `float` values                  |
| `ToComplex`     | `complex`, `float`, `int`, or `bool` |
| `ToJustComplex` | Only `complex` values                |
| `ToScalar`      | Any numeric type or string           |

## Array-Like Aliases

For arrays of specific dimensions and dtypes:

### 1D Array-Likes

```python
ToInt1D      # Accepts arrays/lists of integers
ToFloat1D    # Accepts arrays/lists of floats
ToComplex1D  # Accepts arrays/lists of complex numbers
```

### 2D Array-Likes

```python
ToInt2D      # Accepts 2D arrays/lists
ToFloat2D    # Accepts 2D arrays/lists
ToComplex2D  # Accepts 2D arrays/lists
```

### 3D Array-Likes

```python
ToInt3D      # Accepts 3D arrays/lists
ToFloat3D    # Accepts 3D arrays/lists
ToComplex3D  # Accepts 3D arrays/lists
```

### N-Dimensional

```python
ToIntND      # Accepts arrays of any dimension
ToFloatND    # Accepts arrays of any dimension
ToComplexND  # Accepts arrays of any dimension
```

## Strict Dimensionality Variants

The `Strict` variants (added in optype 0.7.3) enforce exact dimensionality:

```python
ToFloat1D   vs  ToFloatStrict1D
ToFloat2D   vs  ToFloatStrict2D
```

**Key difference**: Strict variants don't overlap, making them suitable for function overloading.

```python
from typing import overload
import optype.numpy as onp

@overload
def process(data: onp.ToFloatStrict1D) -> float: ...
@overload
def process(data: onp.ToFloatStrict2D) -> onp.Array1D: ...

def process(data):
    arr = np.asarray(data)
    return arr.mean(axis=len(arr.shape) - 1)
```

## Supported Dtypes

Array-like aliases are provided for:

- **Booleans**: `Bool`, `JustBool`
- **Integers**: `Int`, `JustInt`, `UInt`, `JustUInt` (with 8/16/32/64 variants)
- **Floats**: `Float`, `Float16`, `Float32`, `Float64`, `JustFloat64`
- **Complex**: `Complex`, `Complex64`, `Complex128`, `JustComplex128`
- **Strings**: `Str`, `Bytes`
- **Generic**: `Scalar`, `Array`

## Usage Examples

```python
import numpy as np
import optype.numpy as onp

def sum_vector(x: onp.ToFloat1D) -> float:
    """Accept arrays or lists of floats."""
    arr = np.asarray(x, dtype=float)
    return float(arr.sum())

def matrix_product(a: onp.ToFloat2D, b: onp.ToFloat2D) -> onp.Array2D:
    """Accept 2D array-likes."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return np.matmul(a_arr, b_arr)

# Function usage
sum_vector([1.0, 2.0, 3.0])          # ✓ List
sum_vector(np.array([1.0, 2.0]))     # ✓ Array
sum_vector((1.0, 2.0))               # ✓ Tuple

matrix_product([[1, 2], [3, 4]], np.eye(2))  # ✓ Mixed types
```

## Differences from numpy.typing.ArrayLike

| Feature           | `numpy.typing.ArrayLike` | `optype.numpy.To*` |
| ----------------- | ------------------------ | ------------------ |
| Bare scalars      | ✓ Accepted               | ✗ Not accepted     |
| Array form        | ✓ Accepted               | ✓ Accepted         |
| 0D arrays         | ✓ Accepted               | ✓ Accepted         |
| Strict dims       | ✗ Not available          | ✓ Available        |
| Runtime checkable | ✓ Yes                    | ✓ Yes              |
| Dtype-specific    | ✗ Generic `ArrayLike`    | ✓ Specific `To*`   |

## Related Types

- **[Aliases](aliases.md)**: Pre-defined array type aliases
- **[Shape Typing](shape.md)**: For precise shape annotations
- **[DType](dtype.md)**: For dtype specifications
- **[Scalar](scalar.md)**: For scalar type annotations
