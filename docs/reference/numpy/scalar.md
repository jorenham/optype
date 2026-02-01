# Scalar

NumPy scalar type annotations and utilities.

## Overview

NumPy scalars are 0-dimensional arrays that represent single values with specific data types. The `optype.numpy.Scalar` interface provides a generic, runtime-checkable protocol for working with NumPy scalars with precise type information.

## Scalar Type Alias

### Definition

```python
type Scalar[
    # The Python type that .item() returns
    PT: object,
    # The number of bits (itemsize in bytes)
    NB: int = int,
] = ...
```

**Purpose**: Type-safe representation of NumPy scalars with Python type and size information

### Type Parameters

- **PT (Python Type)**: Covariant - the Python type returned by `.item()`
- **NB (Number of Bits)**: Covariant - the itemsize in bytes (defaults to `int`)

## NumPy Scalar Types

```python
import numpy as np
from typing import Literal

import optype.numpy as onp
```

### Boolean Scalars

```python
from typing import Literal

# 1-byte boolean
value: onp.Scalar[bool, Literal[1]] = np.bool_(True)

# .item() returns Python bool
py_bool: bool = value.item()
```

### Integer Scalars

```python
# Signed integers
int8: onp.Scalar[int, Literal[1]] = np.int8(42)
int16: onp.Scalar[int, Literal[2]] = np.int16(1000)
int32: onp.Scalar[int, Literal[4]] = np.int32(1000000)
int64: onp.Scalar[int, Literal[8]] = np.int64(9223372036854775807)

# Unsigned integers
uint8: onp.Scalar[int, Literal[1]] = np.uint8(255)
uint16: onp.Scalar[int, Literal[2]] = np.uint16(65535)
uint32: onp.Scalar[int, Literal[4]] = np.uint32(4294967295)
uint64: onp.Scalar[int, Literal[8]] = np.uint64(18446744073709551615)
```

### Floating-Point Scalars

```python
# 16-bit float (2 bytes)
f16: onp.Scalar[float, Literal[2]] = np.float16(3.14)

# 32-bit float (4 bytes)
f32: onp.Scalar[float, Literal[4]] = np.float32(3.14159)

# 64-bit float (8 bytes) - Python's native float
f64: onp.Scalar[float, Literal[8]] = np.float64(3.141592653589793)
```

### Complex Scalars

```python
# 64-bit complex (8 bytes)
c64: onp.Scalar[complex, Literal[8]] = np.complex64(1 + 2j)

# 128-bit complex (16 bytes)
c128: onp.Scalar[complex, Literal[16]] = np.complex128(1 + 2j)
```

### String and Bytes Scalars

```python
# String scalar (variable length)
str_scalar: onp.Scalar[str] = np.str_("hello")

# Bytes scalar (variable length)
bytes_scalar: onp.Scalar[bytes] = np.bytes_(b"hello")
```

## Scalar Protocol

### CanScalar

The generic scalar protocol (runtime-checkable):

```python
def accept_scalar(x: onp.Scalar) -> None:
    """Accept any NumPy scalar."""
    print(f"Itemsize: {x.itemsize}")
    print(f"Value: {x.item()}")

# Usage
accept_scalar(np.int32(42))
accept_scalar(np.float64(3.14))
accept_scalar(np.complex128(1+2j))
```

## Practical Examples

### Function with Scalar Type Specification

```python
def double_value(x: onp.Scalar[int, Literal[8]]) -> onp.Scalar[int, Literal[8]]:
    """Double a 64-bit integer."""
    return np.int64(x.item() * 2)

# Usage
result = double_value(np.int64(21))  # Type-safe
print(result.item())  # 42
```

### Working with Mixed Scalar Types

```python
def to_python_type(scalar: onp.Scalar) -> object:
    """Convert any NumPy scalar to Python type."""
    return scalar.item()

# Works with any scalar
values = [
    np.int32(42),
    np.float64(3.14),
    np.complex128(1+2j),
    np.bool_(True),
]

python_values = [to_python_type(v) for v in values]
```

### Array Element Access as Scalar

```python
def get_pixel(
    image: onp.Array[tuple[int, int, Literal[3]], np.uint8], x: int, y: int,
) -> onp.Scalar[int, Literal[1]]:
    """Get a pixel value as a scalar."""
    return image[y, x]

# The returned element is a NumPy scalar
rgb_value = np.array([(255, 0, 128)], dtype=np.uint8)[0, 0]
```

## Scalar Operations

### Properties

All NumPy scalars have useful properties:

```python
scalar = np.int32(42)

# Size information
print(scalar.itemsize)      # 4 (bytes)
print(scalar.dtype)         # int32
print(scalar.nbytes)        # 4

# Type information
print(scalar.dtype.kind)    # 'i' (integer)
print(scalar.dtype.name)    # 'int32'
```

### Methods

Common scalar methods:

```python
scalar = np.float64(3.14159)

# Convert to Python type
py_float = scalar.item()  # Returns Python float

# Type conversions
as_int = scalar.astype(np.int32)
as_complex = scalar.astype(np.complex128)

# String representation
print(scalar)            # '3.14159'
print(repr(scalar))      # 'np.float64(3.14159)'
```

## Best Practices

### Use Specific Scalar Types in Type Hints

```python
# ✓ Specific and informative
def process(value: onp.Scalar[int, Literal[8]]) -> None:
    pass

# ✗ Vague
def process(value: np.generic) -> None:
    pass
```

### Default NB Parameter

When the number of bits doesn't matter:

```python
# These are equivalent:
onp.Scalar[float]           # NB defaults to int (any size)
onp.Scalar[float, int]
```

### Runtime Checking

```python
def is_scalar(x) -> bool:
    """Check if value is a NumPy scalar."""
    return isinstance(x, np.generic)

print(is_scalar(np.int32(42)))  # True
print(is_scalar(42))            # False
print(is_scalar([42]))          # False
```

## Differences from np.generic

| Feature             | `np.generic` | `Scalar[PT, NB]`         |
| ------------------- | ------------ | ------------------------ |
| Runtime checkable   | ✓ Yes        | ✓ Yes (0D arrays)        |
| Type parameters     | ✗ No         | ✓ Python type + itemsize |
| Python type info    | ✗ No         | ✓ Yes (via PT)           |
| Itemsize info       | ✗ No         | ✓ Yes (via NB)           |
| Backward compatible | ✓ Yes        | ✓ Yes                    |

## Related Types

- **[DType](dtype.md)**: Data type specifications
- **[Aliases](aliases.md)**: Array type aliases
- **[Compat](compat.md)**: Version compatibility utilities
