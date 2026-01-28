# Compatibility (compat)

Cross-version NumPy compatibility utilities.

## Overview

The `optype.numpy.compat` submodule provides compatibility utilities for working across different NumPy versions. It ensures that code works consistently with NumPy >= 1.25, supporting both NumPy 1.x and NumPy 2.x versions.

## Supported NumPy Versions

- **Minimum**: NumPy 1.25
- **Current**: NumPy 1.25+ and NumPy 2.2+

This follows:

- [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html): NumPy deprecation policy
- [SPEC 0](https://scientific-python.org/specs/spec-0000/): Scientific Python version requirements

## Key Features

### numpy-typing-compat Integration

The `compat` module uses the [`numpy-typing-compat`](https://github.com/jorenham/numpy-typing-compat) package to provide:

1. **Type parameter defaults**: NumPy >= 2.2 style type parameters with defaults
2. **Version-agnostic imports**: Safe imports across all supported versions
3. **Abstract scalar types**: Numeric scalar types with proper type parameters

### Abstract Numeric Scalar Types

The module provides abstract scalar type categories:

| Category           | Abstract Type        | Concrete Examples                     |
| ------------------ | -------------------- | ------------------------------------- |
| **Unsigned ints**  | `np.unsignedinteger` | `uint8`, `uint16`, `uint32`, `uint64` |
| **Signed ints**    | `np.signedinteger`   | `int8`, `int16`, `int32`, `int64`     |
| **Real floats**    | `np.floating`        | `float16`, `float32`, `float64`       |
| **Complex floats** | `np.complexfloating` | `complex64`, `complex128`             |
| **Numbers**        | `np.number`          | All numeric types                     |
| **Generic**        | `np.generic`         | All numpy scalar types                |

## Installation

The `compat` module is automatically available when installing optype with NumPy support:

```bash
pip install "optype[numpy]"
```

Or via conda:

```bash
conda install conda-forge::optype-numpy
```

## Usage Examples

```python
import numpy as np
import optype.numpy.compat as npc

# Type checking with abstract categories
def process_integers(arr: np.ndarray[..., np.dtype[npc.signedinteger]]):
    """Accept any signed integer array."""
    return arr.astype(np.float64)

def process_floats(arr: np.ndarray[..., np.dtype[npc.floating]]):
    """Accept any floating-point array."""
    return arr * 2.0

# Works across all supported NumPy versions
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1.0, 2.0], dtype=np.float32)

process_integers(arr_int)    # ✓ Works
process_floats(arr_float)    # ✓ Works
```

## Version-Specific Differences

### NumPy < 2.2

- Type parameters may not have defaults
- Some abstract types may be less specific
- `numpy-typing-compat` provides shims for consistency

### NumPy >= 2.2

- Type parameters have defaults (PEP 696 style)
- More precise type parameter specification
- Full support for modern type annotation features

## Best Practices

1. **Always use `optype.numpy.compat` for cross-version code**:
   ```python
   # ✓ Correct - works across versions
   import optype.numpy.compat as npc

   # ✗ Avoid direct numpy.typing for version-specific code
   import numpy.typing as npt
   ```

2. **Rely on abstract types when you don't care about specific size**:
   ```python
   def normalize(arr: np.ndarray[..., np.dtype[npc.floating]]):
       """Works with float16, float32, float64, etc."""
       return arr / arr.sum()
   ```

3. **Use concrete types when size matters**:
   ```python
   def to_numpy_random_seed(seed: int) -> np.uint64:
       """NumPy random functions expect uint64."""
       return np.uint64(seed)
   ```

## Related Modules

- **[Scalar](scalar.md)**: NumPy scalar type annotations
- **[DType](dtype.md)**: Data type specifications
- **[Aliases](aliases.md)**: Type aliases for arrays
- **[Array-likes](array-likes.md)**: Array-like type aliases
