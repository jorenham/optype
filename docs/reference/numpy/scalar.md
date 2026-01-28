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

### Boolean Scalars

```python
import numpy as np
from optype.numpy import Scalar
from typing import Literal

# 1-byte boolean
value: Scalar[bool, Literal[1]] = np.bool_(True)

# .item() returns Python bool
py_bool: bool = value.item()
```

### Integer Scalars

```python
import numpy as np
from optype.numpy import Scalar
from typing import Literal

# Signed integers
int8: Scalar[int, Literal[1]] = np.int8(42)
int16: Scalar[int, Literal[2]] = np.int16(1000)
int32: Scalar[int, Literal[4]] = np.int32(1000000)
int64: Scalar[int, Literal[8]] = np.int64(9223372036854775807)

# Unsigned integers
uint8: Scalar[int, Literal[1]] = np.uint8(255)
uint16: Scalar[int, Literal[2]] = np.uint16(65535)
uint32: Scalar[int, Literal[4]] = np.uint32(4294967295)
uint64: Scalar[int, Literal[8]] = np.uint64(18446744073709551615)
```

### Floating-Point Scalars

```python
import numpy as np
from optype.numpy import Scalar
from typing import Literal

# 16-bit float (2 bytes)
f16: Scalar[float, Literal[2]] = np.float16(3.14)

# 32-bit float (4 bytes)
f32: Scalar[float, Literal[4]] = np.float32(3.14159)

# 64-bit float (8 bytes) - Python's native float
f64: Scalar[float, Literal[8]] = np.float64(3.141592653589793)
```

### Complex Scalars

```python
import numpy as np
from optype.numpy import Scalar
from typing import Literal

# 64-bit complex (8 bytes)
c64: Scalar[complex, Literal[8]] = np.complex64(1 + 2j)

# 128-bit complex (16 bytes)
c128: Scalar[complex, Literal[16]] = np.complex128(1 + 2j)
```

### String and Bytes Scalars

```python
import numpy as np
from optype.numpy import Scalar

# String scalar (variable length)
str_scalar: Scalar[str] = np.str_("hello")

# Bytes scalar (variable length)
bytes_scalar: Scalar[bytes] = np.bytes_(b"hello")
```

## Scalar Protocol

### CanScalar

The generic scalar protocol (runtime-checkable):

```python
from optype.numpy import Scalar

def accept_scalar(x: Scalar) -> None:
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
import numpy as np
from optype.numpy import Scalar
from typing import Literal

def double_value(x: Scalar[int, Literal[8]]) -> Scalar[int, Literal[8]]:
    """Double a 64-bit integer."""
    return np.int64(x.item() * 2)

# Usage
result = double_value(np.int64(21))  # Type-safe
print(result.item())  # 42
```

### Working with Mixed Scalar Types

```python
import numpy as np
from optype.numpy import Scalar

def to_python_type(scalar: Scalar) -> object:
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
import numpy as np
from optype.numpy import Scalar, Array
from typing import Literal

def get_pixel(
    image: Array[tuple[int, int, Literal[3]], np.uint8],
    x: int,
    y: int,
) -> Scalar[int, Literal[1]]:
    """Get a pixel value as a scalar."""
    return image[y, x]

# The returned element is a NumPy scalar
rgb_value = np.array([(255, 0, 128)], dtype=np.uint8)[0, 0]
```

## Scalar Operations

### Properties

All NumPy scalars have useful properties:

```python
import numpy as np

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
import numpy as np

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
from optype.numpy import Scalar
from typing import Literal

# ✓ Specific and informative
def process(value: Scalar[int, Literal[8]]) -> None:
    pass

# ✗ Vague
def process(value: np.generic) -> None:
    pass
```

### Default NB Parameter

When the number of bits doesn't matter:

```python
from optype.numpy import Scalar

# These are equivalent:
Scalar[float]           # NB defaults to int (any size)
Scalar[float, int]
```

### Runtime Checking

```python
import numpy as np
from optype.numpy import Scalar

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
