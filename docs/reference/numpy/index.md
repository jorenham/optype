# optype.numpy

NumPy type annotations and protocols.

## Overview

`optype.numpy` provides extensive typing support for NumPy, including:

- **Shape typing**: Precise array shape annotations
- **Array-like protocols**: Protocols for array-like objects
- **DType utilities**: Type-safe data type handling
- **Scalar types**: NumPy scalar type annotations
- **UFunc protocols**: Universal function typing
- **Compatibility layer**: Cross-version NumPy compatibility

## Installation

To use `optype.numpy`, install optype with the NumPy extra:

```shell
pip install "optype[numpy]"
```

Or with conda:

```shell
conda install conda-forge::optype-numpy
```

This ensures compatible versions of NumPy and numpy-typing-compat are installed.

## Key Features

### Shape Typing

Define precise array shapes:

```python
import optype.numpy as onp

def normalize(arr: onp.Array2D[onp.floating]) -> onp.Array2D[float]:
    """Normalize a 2D floating-point array."""
    return arr / arr.sum()
```

### Array Protocols

Work with array-like objects:

```python
import optype.numpy as onp

def to_array[ST: np.generic](
    array_like: onp.CanArray[tuple[int, ...], ST]
) -> onp.Array[tuple[int, ...], ST]:
    return np.asarray(array_like)
```

### Type-Safe DTypes

Handle data types precisely:

```python
import optype.numpy as onp

def create_float_array(
    shape: tuple[int, ...],
    dtype: onp.DType[onp.floating]
) -> onp.Array[tuple[int, ...], onp.floating]:
    return np.zeros(shape, dtype=dtype)
```

## Sections

- [Shape Typing](shape.md): Array shape annotations
- [Array-likes](array-likes.md): Array-like protocols
- [Literals](literals.md): Literal types for NumPy
- [Compatibility](compat.md): Cross-version compatibility
- [Random](random.md): Random number generation
- [DType](dtype.md): Data type objects
- [Scalar](scalar.md): Scalar types
- [UFunc](ufunc.md): Universal functions
- [Type Aliases](aliases.md): Common type aliases
- [Low-level](low-level.md): Low-level interfaces

See the [full README](https://github.com/jorenham/optype/blob/master/README.md#optypenumpy) for complete documentation.
