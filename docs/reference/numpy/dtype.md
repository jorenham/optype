# DType

NumPy data type (dtype) utilities and type-safe handling.

## Overview

NumPy's dtype system uses `numpy.dtype[ST]` to specify the scalar type `ST` of an array. The `optype.numpy.DType` alias provides a more convenient, shorter interface with optional type parameters.

## DType Alias

### Definition

```python
type DType[ST: np.generic = np.generic] = np.dtype[ST]
```

### Comparison

| Feature        | `np.dtype`             | `optype.numpy.DType`                |
| -------------- | ---------------------- | ----------------------------------- |
| Name           | snake_case             | CamelCase (idiomatic)               |
| Type parameter | Required               | Optional (defaults to `np.generic`) |
| Verbosity      | `np.dtype[np.float64]` | `DType[float64]` or `DType`         |
| Equivalence    | N/A                    | `DType[ST]` ≡ `np.dtype[ST]`        |

## Usage Examples

```python
import optype.numpy as onp
```

### Basic Usage

```python
import numpy as np

# Without type parameter - matches any dtype
any_dtype: onp.DType = np.dtype(float)

# With specific scalar type
float_dtype: onp.DType[np.float64] = np.dtype(np.float64)
int_dtype: onp.DType[np.int32] = np.dtype(np.int32)
bool_dtype: onp.DType[np.bool_] = np.dtype(bool)
```

### Function Arguments

```python
import numpy as np

def create_array(
    shape: tuple[int, ...],
    dtype: onp.DType[onp.floating[Any]] = np.float64,
) -> onp.Array:
    """Create array with specified dtype."""
    return np.zeros(shape, dtype=dtype)

# Type-safe calls
arr1 = create_array((3, 4))                           # ✓ Uses float64 default
arr2 = create_array((3, 4), dtype=np.float32)         # ✓ Explicit float32
arr3 = create_array((3, 4), dtype=np.dtype(int))      # ✓ Integer dtype
```

### Type-Safe Dtype Operations

```python
import numpy as np

def dtype_kind(dt: onp.DType[onp.generic]) -> str:
    """Get the kind character of a dtype."""
    return dt.kind  # 'f', 'i', 'u', 'b', 'c', etc.

def is_float_dtype(dt: onp.DType) -> bool:
    """Check if dtype is floating-point."""
    return dt.kind == 'f'

# Usage
dt = np.dtype(np.float32)
print(dtype_kind(dt))      # 'f'
print(is_float_dtype(dt))  # True
```

### Working with Array DTypes

```python
import numpy as np

def cast_array(
    arr: onp.Array,
    target_dtype: onp.DType[onp.floating[Any]],
) -> onp.Array:
    """Cast array to target floating-point dtype."""
    return np.asarray(arr, dtype=target_dtype)

def get_array_dtype(arr: onp.Array) -> onp.DType:
    """Extract dtype from array."""
    return arr.dtype

# Usage
arr = np.array([1, 2, 3])
f32 = cast_array(arr, np.float32)
dtype = get_array_dtype(f32)
```

## Type Parameters

### ST (Scalar Type)

- **Type**: `np.generic` (or subtype)
- **Variance**: Covariant
- **Default**: `np.generic` (any numeric type)
- **Purpose**: Specifies the exact scalar type

### Commonly Used Scalar Types

```python
# Float types
np.float16, np.float32, np.float64

# Integer types
np.int8, np.int16, np.int32, np.int64
np.uint8, np.uint16, np.uint32, np.uint64

# Boolean
np.bool_

# Complex
np.complex64, np.complex128

# String/Bytes
np.str_, np.bytes_
```

## DType Properties

All `np.dtype` properties are available on `DType`:

```python
import numpy as np

dt = np.dtype(np.float32)

print(dt.name)          # 'float32'
print(dt.kind)          # 'f' (float)
print(dt.itemsize)      # 4 (bytes)
print(dt.byteorder)     # '=' (native)
print(dt.char)          # 'f' (character code)
print(dt.fields)        # None (not a structured dtype)
```

## Structured DTypes

For structured arrays with named fields:

```python
import numpy as np

# Create a structured dtype
person_dtype = np.dtype([
    ('name', np.str_, 10),
    ('age', np.int32),
    ('height', np.float64),
])

# Type it as a generic DType
dt: onp.DType = person_dtype

# Create array with structured dtype
people = np.array([
    ('Alice', 30, 5.6),
    ('Bob', 25, 6.0),
], dtype=person_dtype)
```

## Related Types

- **[Aliases](aliases.md)**: Array type aliases with dtype parameters
- **[Scalar](scalar.md)**: NumPy scalar type annotations
- **[Array-likes](array-likes.md)**: Type-safe array conversion
- **[Compat](compat.md)**: Cross-version dtype compatibility
