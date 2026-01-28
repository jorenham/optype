# Low-level Interfaces

Low-level NumPy protocols and interfaces for advanced array operations.

## Overview

NumPy provides several special methods and attributes that enable objects to integrate deeply with NumPy's ecosystem. These protocols are used for advanced array operations, universal functions, and custom array types.

## Array Protocol Methods

### CanArray

Implements the `__array__` protocol for converting objects to arrays:

```python
class CanArray[
    ND: tuple[int, ...] = ...,
    ST: np.generic = ...,
]:
    def __array__(
        self,
        dtype: DType[RT] | None = None,
    ) -> Array[ND, RT]: ...
```

**Purpose**: Enable conversion to NumPy array via `np.asarray()` or `np.array()`

**Usage**:

```python
import numpy as np
from optype import CanArray
from optype.numpy import Array

class MyMatrix(CanArray[tuple[int, int], np.float64]):
    def __init__(self, data):
        self._data = np.asarray(data, dtype=float)
    
    def __array__(self, dtype=None):
        if dtype is None:
            return self._data
        return np.asarray(self._data, dtype=dtype)

# Works with NumPy functions
m = MyMatrix([[1, 2], [3, 4]])
arr = np.asarray(m)  # Calls __array__()
```

### CanArrayUFunc

Implements the `__array_ufunc__` protocol (NEP 13):

```python
class CanArrayUFunc[
    U: UFunc = ...,
    R: object = ...,
]:
    def __array_ufunc__(
        self,
        ufunc: U,
        method: LiteralString,
        *args: object,
        **kwargs: object,
    ) -> R: ...
```

**Purpose**: Override universal function behavior on custom types

**Usage**:

```python
import numpy as np
from optype import CanArrayUFunc

class Quantity(CanArrayUFunc[np.ufunc, "Quantity"]):
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            # Handle element-wise operations
            values = [x.value if isinstance(x, Quantity) else x
                     for x in inputs]
            result = ufunc(*values, **kwargs)
            return Quantity(result, self.unit)
        return NotImplemented

# Use with ufuncs
q1 = Quantity(3.0, 'm')
q2 = Quantity(4.0, 'm')
result = np.add(q1, q2)  # Calls __array_ufunc__
```

### CanArrayFunction

Implements the `__array_function__` protocol (NEP 18):

```python
class CanArrayFunction[
    F: CanCall[..., object] = ...,
    R: object = ...,
]:
    def __array_function__(
        self,
        func: F,
        types: CanIterSelf[type[CanArrayFunction]],
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> R: ...
```

**Purpose**: Intercept calls to NumPy functions for custom array types

**Usage**:

```python
import numpy as np
from optype import CanArrayFunction

class MaskedArray(CanArrayFunction[object, object]):
    def __init__(self, data, mask):
        self.data = np.asarray(data)
        self.mask = np.asarray(mask, dtype=bool)
    
    def __array_function__(self, func, types, args, kwargs):
        if func is np.mean:
            # Custom mean for masked data
            return self.data[~self.mask].mean()
        return NotImplemented

# Override NumPy functions
ma = MaskedArray([1, 2, 3], [False, True, False])
mean = np.mean(ma)  # Uses custom implementation
```

### CanArrayFinalize

Implements the `__array_finalize__` protocol:

```python
class CanArrayFinalize[T: object = ...]:
    def __array_finalize__(self, obj: T) -> None: ...
```

**Purpose**: Called after array creation during subclassing

**Usage**:

```python
import numpy as np

class SpecialArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.special_attr = None
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.special_attr = getattr(obj, 'special_attr', None)
```

### CanArrayWrap

Implements the `__array_wrap__` protocol:

```python
class CanArrayWrap:
    def __array_wrap__(
        self,
        array: Array[ND, ST],
        context: tuple | None = None,
        return_scalar: bool = False,
    ) -> Self | Array[ND, ST]: ...
```

**Purpose**: Control the output type of ufuncs and array operations

## Array Interface Attributes

### HasArrayInterface

Provides the `__array_interface__` attribute:

```python
class HasArrayInterface[V: Mapping[str, object] = ...]:
    __array_interface__: V
```

**Purpose**: Expose memory layout and data buffer information

**Interface structure**:

```python
{
    'shape': (rows, cols),
    'typestr': '<f8',  # dtype string
    'data': (data_ptr, False),
    'strides': (stride1, stride2),
    'version': 3,
}
```

### HasArrayPriority

Provides the `__array_priority__` attribute:

```python
class HasArrayPriority:
    __array_priority__: float
```

**Purpose**: Control which object's method takes precedence in operations

**Convention**:

- Python scalars: 0
- Arrays: 0
- Matrix: 15
- Quantity/custom: 20+

### HasDType

Provides the `dtype` attribute:

```python
class HasDType[DT: DType = ...]:
    dtype: DT
```

**Purpose**: Expose the data type of an object's elements

## Integration Examples

### Complete Custom Array Type

```python
import numpy as np
from optype import CanArray, CanArrayUFunc, HasDType
from optype.numpy import Array, DType

class CustomArray(CanArray[tuple[int, ...], np.generic] &
                  CanArrayUFunc[np.ufunc, "CustomArray"] &
                  HasDType[np.dtype[np.float64]]):
    def __init__(self, data):
        self._array = np.asarray(data, dtype=np.float64)
    
    @property
    def dtype(self) -> np.dtype[np.float64]:
        return self._array.dtype
    
    def __array__(self, dtype=None):
        if dtype is None:
            return self._array
        return np.asarray(self._array, dtype=dtype)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            args = [x._array if isinstance(x, CustomArray) else x
                   for x in inputs]
            result = ufunc(*args, **kwargs)
            return CustomArray(result)
        return NotImplemented

# Usage
c1 = CustomArray([1, 2, 3])
c2 = CustomArray([4, 5, 6])
result = np.add(c1, c2)  # Type-safe!
```

## References

- [NumPy Enhancement Proposals (NEP 13, NEP 18, NEP 29)](https://numpy.org/neps/)
- [NumPy C-API Array Interface Protocol](https://numpy.org/doc/stable/reference/arrays.interface.html)
- [NumPy Subclassing ndarray](https://numpy.org/doc/stable/user/basics.subclassing.html)

## Related Types

- **[UFunc](ufunc.md)**: Universal function type annotations
- **[Aliases](aliases.md)**: Array type aliases
- **[Array-likes](array-likes.md)**: Array-like protocols
- **[DType](dtype.md)**: Data type specifications
