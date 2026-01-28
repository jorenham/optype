# Shape Typing

Precise array shape type annotations for NumPy.

## Overview

Shape typing enables precise specifications of array dimensionality and shapes. `optype.numpy` provides comprehensive shape typing utilities including array aliases, typeguards, and shape constraints.

## Array Aliases

### Basic Array Types

Pre-defined aliases for common array dimensions:

```python
type Array0D[ST: np.generic = np.generic] = ndarray[tuple[()], dtype[ST]]
type Array1D[ST: np.generic = np.generic] = ndarray[tuple[int], dtype[ST]]
type Array2D[ST: np.generic = np.generic] = ndarray[tuple[int, int], dtype[ST]]
type Array3D[ST: np.generic = np.generic] = ndarray[tuple[int, int, int], dtype[ST]]
type ArrayND[ST: np.generic = np.generic] = ndarray[tuple[int, ...], dtype[ST]]
```

### Detailed Array vs ArrayND

```python
# Array with both shape and scalar parameters
type Array[
    ND: tuple[int, ...] = (int, ...),
    SCT: np.generic = np.generic,
] = ndarray[ND, dtype[SCT]]

# ArrayND with scalar first (matches numpy style)
type ArrayND[
    SCT: np.generic = np.generic,
    ND: tuple[int, ...] = (int, ...),
] = ndarray[ND, dtype[SCT]]
```

### Usage Examples

```python
import numpy as np
import optype.numpy as onp

# 0D scalar arrays
scalar_arr: onp.Array0D[np.int32] = np.array(42)

# 1D vectors
vector: onp.Array1D[np.float64] = np.array([1.0, 2.0, 3.0])

# 2D matrices
matrix: onp.Array2D[np.float32] = np.array([[1, 2], [3, 4]], dtype=np.float32)

# 3D tensors
tensor: onp.Array3D[np.uint8] = np.zeros((10, 20, 30), dtype=np.uint8)

# N-dimensional arrays
nd_array: onp.ArrayND[np.complex128] = np.random.randn(2, 3, 4, 5) + 1j
```

## Array Typeguards

[PEP 742](https://peps.python.org/pep-0742/) typeguards for runtime shape checking:

| Typeguard     | Narrows To    | Shape Type             |
| ------------- | ------------- | ---------------------- |
| `is_array_nd` | `ArrayND[ST]` | `tuple[int, ...]`      |
| `is_array_0d` | `Array0D[ST]` | `tuple[()]`            |
| `is_array_1d` | `Array1D[ST]` | `tuple[int]`           |
| `is_array_2d` | `Array2D[ST]` | `tuple[int, int]`      |
| `is_array_3d` | `Array3D[ST]` | `tuple[int, int, int]` |

### Signature

```python
_T = TypeVar("_T", bound=np.generic, default=Any)
_ToDType = type[_T] | np.dtype[_T] | HasDType[np.dtype[_T]]

def is_array_0d(a, /, dtype: _ToDType[_T] | None = None) -> TypeIs[Array0D[_T]]: ...
def is_array_1d(a, /, dtype: _ToDType[_T] | None = None) -> TypeIs[Array1D[_T]]: ...
def is_array_2d(a, /, dtype: _ToDType[_T] | None = None) -> TypeIs[Array2D[_T]]: ...
def is_array_3d(a, /, dtype: _ToDType[_T] | None = None) -> TypeIs[Array3D[_T]]: ...
def is_array_nd(a, /, dtype: _ToDType[_T] | None = None) -> TypeIs[ArrayND[_T]]: ...
```

### Typeguard Examples

```python
import numpy as np
import optype.numpy as onp

def process_matrix(data: object) -> float:
    """Process only 2D arrays."""
    if onp.is_array_2d(data, dtype=np.float64):
        # Type checker knows data is Array2D[np.float64]
        return data.mean()
    raise TypeError("Expected 2D float array")

def process_vector(data: object) -> int:
    """Process only 1D integer arrays."""
    if onp.is_array_1d(data, dtype=np.integer):
        # Type checker knows data is Array1D[np.integer]
        return len(data)
    raise TypeError("Expected 1D integer array")

# Usage
arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
arr_1d = np.array([1, 2, 3])

process_matrix(arr_2d)  # ✓ Works
process_vector(arr_1d)  # ✓ Works
```

## Shape Aliases

Type aliases for constraining array dimensionality:

### AtLeast{N}D - Minimum Dimensions

Arrays with at least N dimensions:

```python
type AtLeast0D = tuple[int, ...]  # Any shape
type AtLeast1D = tuple[int, *AtLeast0D]  # 1+ dimensions
type AtLeast2D = tuple[int, int] | AtLeast3D  # 2+ dimensions
type AtLeast3D = (
    tuple[int, int, int]
    | tuple[int, int, int, int]
    | tuple[int, int, int, int, int]
    # ...up to 64 dimensions
)
```

### AtMost{N}D - Maximum Dimensions

Arrays with at most N dimensions:

```python
type AtMost0D = tuple[()]  # Scalar only
type AtMost1D = AtMost0D | tuple[int]  # 0D or 1D
type AtMost2D = AtMost1D | tuple[int, int]  # 0D, 1D, or 2D
type AtMost3D = AtMost2D | tuple[int, int, int]  # 0D through 3D
```

### Usage Examples

```python
import numpy as np
import optype.numpy as onp

def flatten_multidim(arr: onp.Array[onp.AtLeast2D]) -> onp.Array1D:
    """Flatten arrays with 2+ dimensions."""
    return arr.flatten()

def ensure_vector(arr: onp.Array[onp.AtMost1D]) -> onp.Array1D:
    """Ensure shape is at most 1D."""
    if arr.ndim == 0:
        return np.array([arr.item()])
    return arr

# Usage
matrix = np.array([[1, 2], [3, 4]])
flattened = flatten_multidim(matrix)  # ✓ 2D → 1D

scalar = np.array(42)
vector = ensure_vector(scalar)  # ✓ 0D → 1D
```

## Gradual Shape Typing

The `AtLeast{N}D` aliases optionally accept `Any` for gradual typing:

```python
# Default: int (strict)
type AtLeast1D = tuple[int, *tuple[int, ...]]

# With Any: gradual (more flexible)
type AtLeast1D[Any] = tuple[Any, *tuple[Any, ...]]
```

**Note**: mypy has a [bug](https://github.com/python/mypy/issues/19109) with gradual shape types for N≥1.

## NumPy Version Considerations

### NumPy < 2.1

Shape type parameter was **invariant**. Avoid `Literal` in shape types:

```python
# ✗ Avoid on numpy<2.1
arr: np.ndarray[tuple[Literal[3], Literal[3]], ...]

# ✓ Use instead
arr: np.ndarray[tuple[int, int], ...]
```

### NumPy >= 2.1

Shape type parameter is **covariant**. `Literal` shapes work:

```python
# ✓ Works on numpy>=2.1
arr: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]
```

**See**: [numpy/numpy#25729](https://github.com/numpy/numpy/issues/25729), [numpy/numpy#26081](https://github.com/numpy/numpy/pull/26081)

## Related Types

- **[Aliases](aliases.md)**: Array type aliases
- **[Array-likes](array-likes.md)**: Array-like type conversions
- **[DType](dtype.md)**: Data type specifications
- **[Scalar](scalar.md)**: Scalar type annotations
