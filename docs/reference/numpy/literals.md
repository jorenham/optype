# Literals

Literal types for NumPy constants and enumerations.

## Overview

The `optype.numpy` module provides `Literal` type aliases for common NumPy string constants and enumerations. These enable type-safe handling of NumPy function parameters that accept string values.

## Available Literal Types

### Byte Order

```python
type ByteOrder = ByteOrderChar | ByteOrderName | Literal['L', 'B', 'N', 'I', 'S']

# Character form (single-character representations)
type ByteOrderChar = Literal['<', '>', '=', '|']

# Name form (descriptive names)
type ByteOrderName = Literal['little', 'big', 'native', 'ignore', 'swap']
```

**Usage**: Specifying array byte order in dtype creation

```python
arr = np.array([1, 2, 3], dtype='<i4')  # Little-endian int32
```

### Casting Modes

```python
type Casting = CastingUnsafe | CastingSafe

type CastingUnsafe = Literal['unsafe']
type CastingSafe = Literal['no', 'equiv', 'safe', 'same_kind']
```

**Usage**: Controlling type casting in NumPy operations

```python
np.asarray(data, dtype=float, casting='safe')  # Only safe casts
```

### Convolution Modes

```python
type ConvolveMode = Literal['full', 'same', 'valid']
```

**Usage**: Specifying output shape in convolution-like operations

```python
np.convolve(x, h, mode='same')  # Output same size as input
```

### Device Type

```python
type Device = Literal['cpu']
```

**Usage**: Device specification (placeholder for future GPU support)

```python
arr = np.array([1, 2, 3], device='cpu')
```

### Index Mode

```python
type IndexMode = Literal['raise', 'wrap', 'clip']
```

**Usage**: Handling out-of-bounds indices

```python
np.take([1, 2, 3], [5], mode='wrap')  # Wraps index to valid range
```

### Array Order

```python
# C-order or Fortran-order
type OrderCF = Literal['C', 'F']

# Any, C, or Fortran-order
type OrderACF = Literal['A', 'C', 'F']

# Keep-order, Any, C, or Fortran-order
type OrderKACF = Literal['K', 'A', 'C', 'F']
```

**Usage**: Specifying memory layout

```python
np.reshape(arr, (3, 4), order='C')  # C-contiguous
np.reshape(arr, (3, 4), order='F')  # Fortran-contiguous
```

### Partition Kind

```python
type PartitionKind = Literal['introselect']
```

**Usage**: Algorithm selection for partitioning

```python
np.partition(arr, kth=5, kind='introselect')
```

### Sort Kind

```python
type SortKind = Literal[
    'Q', 'quick', 'quicksort',
    'M', 'merge', 'mergesort',
    'H', 'heap', 'heapsort',
    'S', 'stable', 'stable'
]
```

**Usage**: Selecting sorting algorithm

```python
np.sort(arr, kind='mergesort')  # Guaranteed stable sort
```

### Sort Side

```python
type SortSide = Literal['left', 'right']
```

**Usage**: Breaking ties in searchsorted

```python
np.searchsorted(arr, value, side='right')
```

## Type-Safe Function Signatures

Using these literals makes function signatures more precise:

```python
import numpy as np
import optype.numpy as onp
from typing import Literal

def create_array(shape: tuple[int, ...], byteorder: onp.ByteOrder = '=') -> np.ndarray:
    """Create array with specified byte order."""
    return np.zeros(shape).astype(f'{byteorder}f8')

def convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve returning same-sized output."""
    return np.convolve(x, h, mode='same')

def sort_stable(arr: np.ndarray, kind: onp.SortKind = 'stable') -> np.ndarray:
    """Sort using specified algorithm."""
    return np.sort(arr, kind=kind)
```

## Common Patterns

### Parameter with Multiple Literal Options

```python
def reshape_array(
    arr: np.ndarray,
    shape: tuple[int, ...],
    order: Literal['C', 'F'] = 'C',
) -> np.ndarray:
    """Reshape with C or Fortran order."""
    return arr.reshape(shape, order=order)
```

### Mode Selection

```python
def index_with_mode(
    arr: np.ndarray,
    indices: np.ndarray,
    mode: Literal['raise', 'wrap', 'clip'] = 'raise',
) -> np.ndarray:
    """Index with different boundary handling."""
    return np.take(arr, indices, mode=mode)
```

### Casting Control

```python
def safe_cast(
    arr: np.ndarray,
    dtype: np.dtype,
    casting: Literal['safe', 'same_kind'] = 'safe',
) -> np.ndarray:
    """Cast with safety guarantees."""
    return np.asarray(arr, dtype=dtype, casting=casting)
```

## Related Types

- **[Array-likes](array-likes.md)**: Array type specifications
- **[DType](dtype.md)**: Data type annotations
- **[Scalar](scalar.md)**: NumPy scalar types
