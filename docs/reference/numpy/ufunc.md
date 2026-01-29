# UFunc

Universal function (ufunc) type annotations for NumPy.

## Overview

Universal functions (ufuncs) are the core of NumPy's vectorized operations. They operate element-wise on arrays and support broadcasting. The `optype.numpy.UFunc` protocol provides comprehensive type annotations for ufuncs.

## Why optype.numpy.UFunc?

The built-in `np.ufunc` lacks type parameters, making it impossible to:

- Specify callable signatures precisely
- Annotate literal attributes (`nin`, `nout`, `identity`)
- Distinguish between regular and generalized ufuncs

`optype.numpy.UFunc` solves this with a runtime-checkable generic protocol.

## UFunc Type Signature

```python
type UFunc[
    # Callable signature
    Fn: CanCall = CanCall,
    # Number of inputs
    Nin: int = int,
    # Number of outputs
    Nout: int = int,
    # Signature (None for element-wise, string for generalized)
    Sig: str | None = str | None,
    # Identity element
    Id: complex | bytes | str | None = float | None,
] = ...
```

## Type Parameters

### Fn (Function)

Callable signature of the ufunc:

```python
import numpy as np
from optype.numpy import UFunc
from typing import Callable

# Binary operation returning same type
add: UFunc[Callable[[float, float], float], ...]

# Unary operation
sin: UFunc[Callable[[float], float], ...]
```

### Nin (Number of Inputs)

Number of input arguments:

```python
# Unary ufuncs
sin: UFunc[..., Literal[1], ...]  # 1 input

# Binary ufuncs
add: UFunc[..., Literal[2], ...]  # 2 inputs
```

### Nout (Number of Outputs)

Number of output values:

```python
# Single output
add: UFunc[..., ..., Literal[1], ...]

# Multiple outputs (like divmod)
divmod: UFunc[..., Literal[2], Literal[2], ...]
```

### Sig (Signature)

- **`None`**: Element-wise ufunc (default)
- **`str`**: Generalized ufunc signature

```python
# Element-wise ufunc
add: UFunc[..., ..., ..., None, ...]

# Generalized ufunc (gufunc)
matmul: UFunc[..., ..., ..., "(m,n),(n,p)->(m,p)", ...]
```

### Id (Identity Element)

The identity value for reduction operations:

```python
# Ufuncs with identity
add: UFunc[..., ..., ..., ..., Literal[0]]  # 0 is identity for addition
multiply: UFunc[..., ..., ..., ..., Literal[1]]  # 1 for multiplication

# No identity
maximum: UFunc[..., ..., ..., ..., None]
```

## Common Ufunc Examples

### Unary Ufuncs

```python
import numpy as np
from optype.numpy import UFunc
from typing import Literal, Callable

# Trigonometric
sin: UFunc[
    Callable[[float], float],
    Literal[1],  # 1 input
    Literal[1],  # 1 output
    None,        # element-wise
    None,        # no identity
]

# Absolute value
abs: UFunc[
    Callable[[float], float],
    Literal[1],
    Literal[1],
    None,
    None,
]
```

### Binary Ufuncs

```python
import numpy as np
from optype.numpy import UFunc
from typing import Literal, Callable

# Addition
add: UFunc[
    Callable[[float, float], float],
    Literal[2],  # 2 inputs
    Literal[1],  # 1 output
    None,        # element-wise
    Literal[0],  # identity = 0
]

# Multiplication
multiply: UFunc[
    Callable[[float, float], float],
    Literal[2],
    Literal[1],
    None,
    Literal[1],  # identity = 1
]

# Maximum (no identity)
maximum: UFunc[
    Callable[[float, float], float],
    Literal[2],
    Literal[1],
    None,
    None,  # no identity
]
```

### Generalized Ufuncs (gufuncs)

```python
import numpy as np
from optype.numpy import UFunc
from typing import Literal

# Matrix multiplication
matmul: UFunc[
    ...,
    Literal[2],
    Literal[1],
    "(m,n),(n,p)->(m,p)",  # signature
    None,
]

# Outer product
outer: UFunc[
    ...,
    Literal[2],
    Literal[1],
    "(i),(j)->(i,j)",
    None,
]
```

## Ufunc Methods

### Direct Call

```python
import numpy as np

result = np.add(1, 2)  # Calls ufunc directly
```

### `.reduce()`

Applies ufunc along an axis:

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
total = np.add.reduce(arr)  # 10 (1+2+3+4)
```

### `.accumulate()`

Intermediate results of reduction:

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
cumsum = np.add.accumulate(arr)  # [1, 3, 6, 10]
```

### `.outer()`

Outer product of two arrays:

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20])
result = np.multiply.outer(a, b)
# [[10, 20],
#  [20, 40],
#  [30, 60]]
```

### `.at()`

In-place operation at specified indices:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
np.add.at(arr, [0, 2, 4], 10)
# arr is now [11, 2, 13, 4, 15]
```

### `.reduceat()`

Reduce at specified slices:

```python
import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5])
result = np.add.reduceat(arr, [0, 3, 5])
# [3, 7, 5]  # sum([0:3]), sum([3:5]), sum([5:])
```

## Custom Ufuncs

### Using np.frompyfunc

```python
import numpy as np
from optype.numpy import UFunc

def my_func(x: float, y: float) -> float:
    """Custom binary function."""
    return x ** 2 + y ** 2

# Create ufunc
ufunc: UFunc = np.frompyfunc(my_func, nin=2, nout=1)

# Use it
arr1 = np.array([1.0, 2.0, 3.0])
arr2 = np.array([4.0, 5.0, 6.0])
result = ufunc(arr1, arr2)
```

### Duck-Typed Ufunc

```python
from optype.numpy import UFunc
from typing import Callable

class CustomUFunc(UFunc[Callable[[float], float], int, int, None, None]):
    """Custom ufunc-like class."""
    
    nin = 1
    nout = 1
    signature = None
    identity = None
    
    def __call__(self, x: float) -> float:
        return x * 2

uf = CustomUFunc()
result = uf(3.14)  # 6.28
```

## Important Notes

### Method Annotation Limitation

NumPy's ufunc methods (`.at`, `.reduce`, `.reduceat`, `.accumulate`, `.outer`) are incorrectly annotated as `None` attributes in numpy's stubs, even though they're callable methods at runtime. This prevents `optype.numpy.UFunc` from properly typing them.

**Workaround**: Cast to specific callable when needed:

```python
import numpy as np
from typing import cast, Callable

reduce_fn = cast(Callable, np.add.reduce)
result = reduce_fn([1, 2, 3, 4])
```

### Identity Requirements

The `identity` attribute is only valid when:

- `Nin == Literal[2]`
- `Nout == Literal[1]`
- `Sig == None` (not a gufunc)

## Runtime Checking

```python
import numpy as np
from optype.numpy import UFunc

def apply_ufunc(func: UFunc, arr: np.ndarray) -> np.ndarray:
    """Apply any ufunc to an array."""
    if isinstance(func, np.ufunc):
        return func(arr)
    raise TypeError("Not a ufunc")

# Usage
result = apply_ufunc(np.sin, np.array([0, np.pi/2, np.pi]))
```

## Common NumPy Ufuncs

### Mathematical

- **Arithmetic**: `add`, `subtract`, `multiply`, `divide`, `power`
- **Trigonometric**: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`
- **Hyperbolic**: `sinh`, `cosh`, `tanh`
- **Exponential**: `exp`, `log`, `log10`, `log2`
- **Rounding**: `floor`, `ceil`, `trunc`, `round`

### Comparison

- `greater`, `greater_equal`, `less`, `less_equal`
- `equal`, `not_equal`
- `maximum`, `minimum`

### Logical

- `logical_and`, `logical_or`, `logical_not`, `logical_xor`

### Bitwise

- `bitwise_and`, `bitwise_or`, `bitwise_xor`, `invert`
- `left_shift`, `right_shift`

## Related Types

- **[Low-level](low-level.md)**: `CanArrayUFunc` protocol
- **[Aliases](aliases.md)**: Array type aliases
- **[Array-likes](array-likes.md)**: Array-like types
- **[Scalar](scalar.md)**: Scalar type annotations

## References

- [NumPy UFunc Documentation](https://numpy.org/doc/stable/reference/ufuncs.html)
- [NumPy Generalized UFuncs](https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html)
- [NEP 13: UFunc Overrides](https://numpy.org/neps/nep-0013-ufunc-overrides.html)
