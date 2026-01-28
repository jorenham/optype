# Descriptors

Protocols for descriptor objects implementing the descriptor protocol.

## Overview

The descriptor protocol is a powerful Python mechanism that allows objects to customize attribute access. Descriptors are objects that define `__get__`, `__set__`, and/or `__delete__` methods to intercept attribute access on classes and instances.

`optype` provides protocols for implementing descriptors with proper type safety and variance handling.

| Operation              | Method         | Protocol                |
| ---------------------- | -------------- | ----------------------- |
| `T().d` or `T.d`       | `__get__`      | `CanGet[-T, +V, +VT=V]` |
| `T().d` (returns Self) | `__get__`      | `CanGetSelf[-T, +V]`    |
| `T().k = v`            | `__set__`      | `CanSet[-T, -V]`        |
| `del T().k`            | `__delete__`   | `CanDelete[-T]`         |
| `class T: d = _`       | `__set_name__` | `CanSetName[-T]`        |

## Protocol Details

### CanGet[-T, +V, +VT=V]

The standard descriptor protocol for getting attributes. It handles both instance and class attribute access:

- Instance access: `obj.descriptor -> V`
- Class access: `Class.descriptor -> VT`

```python
from optype import CanGet

class Descriptor(CanGet[object, int, str]):
    def __get__(self, obj: object | None, objtype: type | None = None) -> int | str:
        if obj is None:
            return "class attribute"
        return 42
```

### CanGetSelf[-T, +V]

For descriptors that return `Self` (typing.Self) on class access:

```python
from optype import CanGetSelf
from typing import Self

class ClassMethod(CanGetSelf[object, int]):
    def __get__(self, obj: object | None, objtype: type | None = None) -> int | Self:
        if obj is None:
            return self
        return 42
```

### CanSet[-T, -V]

For data descriptors that allow setting attribute values:

```python
from optype import CanSet

class DataDescriptor(CanSet[object, int]):
    def __set__(self, obj: object, value: int) -> None:
        print(f"Setting to {value}")
```

### CanDelete[-T]

For descriptors that support deletion:

```python
from optype import CanDelete

class DeletableDescriptor(CanDelete[object]):
    def __delete__(self, obj: object) -> None:
        print("Deleted")
```

### CanSetName[-T]

Called when the descriptor is assigned to a class attribute:

```python
from optype import CanSetName

class NameAware(CanSetName[object]):
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
```

## Examples

### Property-like Descriptor

```python
from optype import CanGet, CanSet

class Property(CanGet[object, int], CanSet[object, int]):
    def __init__(self, fget=None, fset=None):
        self.fget = fget
        self.fset = fset
        self.values = {}
    
    def __get__(self, obj: object | None, objtype: type | None = None) -> int:
        if obj is None:
            return self
        if self.fget:
            return self.fget(obj)
        return self.values.get(id(obj), 0)
    
    def __set__(self, obj: object, value: int) -> None:
        if self.fset:
            self.fset(obj, value)
        else:
            self.values[id(obj)] = value
    
    def setter(self, fset):
        return Property(self.fget, fset)


class Circle:
    def __init__(self, radius: int):
        self._radius = radius
    
    @Property
    def radius(self) -> int:
        return self._radius
    
    @radius.setter
    def radius(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value
```

### Lazy Attribute Loader

```python
from optype import CanGet, CanSetName

class Lazy(CanGet[object, object], CanSetName[object]):
    def __init__(self, func):
        self.func = func
        self.name = None
    
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
    
    def __get__(self, obj: object | None, objtype: type | None = None) -> object:
        if obj is None:
            return self
        # Compute value on first access and cache it
        value = self.func(obj)
        setattr(obj, self.name, value)
        return value


class DataModel:
    @Lazy
    def expensive_data(self) -> str:
        print("Computing expensive data...")
        return "result"


obj = DataModel()
print(obj.expensive_data)  # Computing expensive data... (first time)
print(obj.expensive_data)  # (no computation, cached)
```

### Method Descriptor

```python
from optype import CanGet, CanGetSelf
from typing import Self

class Method(CanGetSelf[object, None]):
    def __init__(self, func):
        self.func = func
    
    def __get__(self, obj: object | None, objtype: type | None = None) -> Self:
        if obj is None:
            return self
        # Bind the function to the instance
        from functools import partial
        return partial(self.func, obj)


class MyClass:
    def __init__(self, value: int):
        self.value = value
    
    @Method
    def double(self) -> int:
        return self.value * 2
```

### Typed Validator Descriptor

```python
from optype import CanSet, CanGet, CanDelete

class ValidatedInt(CanGet[object, int], CanSet[object, int], CanDelete[object]):
    def __init__(self, min_val: int = 0, max_val: int = 100):
        self.min_val = min_val
        self.max_val = max_val
        self.values = {}
    
    def __get__(self, obj: object | None, objtype: type | None = None) -> int:
        if obj is None:
            return 0
        return self.values.get(id(obj), self.min_val)
    
    def __set__(self, obj: object, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}")
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(f"Value {value} not in range [{self.min_val}, {self.max_val}]")
        self.values[id(obj)] = value
    
    def __delete__(self, obj: object) -> None:
        self.values.pop(id(obj), None)


class Person:
    age = ValidatedInt(0, 150)
    
    def __init__(self, age: int):
        self.age = age
```

## Type Variance

- **Owner type (`-T`)**: Contravariant - descriptor accepts owner types or supertypes
- **Value (`+V`)**: Covariant - descriptor returns this type or subtypes
- **Value Type (`+VT`)**: Covariant - class attribute type

## See Also

- [Python Descriptor Documentation](https://docs.python.org/3/howto/descriptor.html)
- [Attributes](attributes.md): For general attribute protocols

## Related Protocols

- **[Attributes](attributes.md)**: For attribute access operations
- **[Containers](containers.md)**: For item access operations
