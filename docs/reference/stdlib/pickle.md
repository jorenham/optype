# optype.pickle

Protocols for pickle serialization.

## Overview

Protocols for customizing object serialization and deserialization with the [`pickle`](https://docs.python.org/3/library/pickle.html) standard library.

## Protocols

| Protocol          | Methods                                                               | Purpose                       |
| ----------------- | --------------------------------------------------------------------- | ----------------------------- |
| `CanReduce`       | `__reduce__() -> str \| tuple[object, ...]`                           | Basic pickle support          |
| `CanReduceEx`     | `__reduce_ex__(protocol: int) -> str \| tuple[object, ...]`           | Protocol-aware pickling       |
| `CanGetstate`     | `__getstate__() -> object`                                            | Custom state serialization    |
| `CanSetstate`     | `__setstate__(state: object) -> None`                                 | Custom state restoration      |
| `CanGetnewargs`   | `__getnewargs__() -> tuple[object, ...]`                              | Arguments for `__new__`       |
| `CanGetnewargsEx` | `__getnewargs_ex__() -> tuple[tuple[object, ...], dict[str, object]]` | Args and kwargs for `__new__` |

## Usage Examples

### Basic Custom Pickling

```python
import pickle
from optype.pickle import CanReduce

class CustomObject(CanReduce):
    def __init__(self, value: int) -> None:
        self.value = value
    
    def __reduce__(self) -> tuple[type[CustomObject], tuple[int]]:
        \"\"\"Define how to pickle this object.\"\"\"
        return (self.__class__, (self.value,))

# Pickle and unpickle
obj = CustomObject(42)
pickled = pickle.dumps(obj)
restored = pickle.loads(pickled)
assert restored.value == 42
```

### State Management

```python
from optype.pickle import CanGetstate, CanSetstate
import pickle

class CachedData(CanGetstate, CanSetstate):
    def __init__(self, data: list[int]) -> None:
        self.data = data
        self._cache: dict[str, int] = {}
    
    def __getstate__(self) -> dict[str, object]:
        \"\"\"Exclude cache from pickle.\"\"\"
        state = self.__dict__.copy()
        # Don't pickle the cache
        del state['_cache']
        return state
    
    def __setstate__(self, state: dict[str, object]) -> None:
        \"\"\"Restore state and rebuild cache.\"\"\"
        self.__dict__.update(state)
        # Rebuild empty cache
        self._cache = {}

# Cache is not pickled
obj = CachedData([1, 2, 3])
obj._cache['sum'] = 6
pickled = pickle.dumps(obj)
restored = pickle.loads(pickled)
assert restored.data == [1, 2, 3]
assert restored._cache == {}  # Cache was reset
```

### Protocol-Aware Pickling

```python
from optype.pickle import CanReduceEx
import pickle

class VersionedObject(CanReduceEx):
    VERSION = 2
    
    def __init__(self, data: str) -> None:
        self.data = data
    
    def __reduce_ex__(self, protocol: int) -> tuple[type, tuple[str]]:
        \"\"\"Use different serialization based on protocol.\"\"\"
        if protocol >= 4:
            # Use efficient protocol 4+ features
            return (self.__class__, (self.data,))
        else:
            # Fall back to simpler representation
            return (self.__class__, (self.data.encode('utf-8').decode(),))

obj = VersionedObject("data")
# Use protocol 5 (most recent)
pickled = pickle.dumps(obj, protocol=5)
restored = pickle.loads(pickled)
```

### Immutable Object Pickling

```python
from optype.pickle import CanGetnewargs
import pickle

class Point(CanGetnewargs):
    \"\"\"Immutable point with custom pickling.\"\"\"
    
    def __init__(self, x: float, y: float) -> None:
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'y', y)
    
    def __getnewargs__(self) -> tuple[float, float]:
        \"\"\"Return arguments for __new__.\"\"\"
        return (self.x, self.y)
    
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Point is immutable")

point = Point(3.0, 4.0)
restored = pickle.loads(pickle.dumps(point))
assert (restored.x, restored.y) == (3.0, 4.0)
```

### Advanced Constructor Arguments

```python
from optype.pickle import CanGetnewargsEx
import pickle

class ConfigurableObject(CanGetnewargsEx):
    def __init__(self, value: int, *, debug: bool = False, cache_size: int = 100) -> None:
        self.value = value
        self.debug = debug
        self.cache_size = cache_size
    
    def __getnewargs_ex__(self) -> tuple[tuple[int], dict[str, object]]:
        \"\"\"Return args and kwargs for __new__.\"\"\"
        args = (self.value,)
        kwargs = {'debug': self.debug, 'cache_size': self.cache_size}
        return (args, kwargs)

obj = ConfigurableObject(42, debug=True, cache_size=200)
restored = pickle.loads(pickle.dumps(obj))
assert restored.value == 42
assert restored.debug is True
assert restored.cache_size == 200
```

### Singleton Pattern with Pickle

```python
from optype.pickle import CanReduce
import pickle

class Singleton(CanReduce):
    _instance: 'Singleton | None' = None
    
    def __new__(cls) -> 'Singleton':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __reduce__(self) -> tuple[type[Singleton], tuple[()]]:
        \"\"\"Ensure singleton is maintained after unpickling.\"\"\"
        return (self.__class__, ())

# Same instance is restored
singleton1 = Singleton()
pickled = pickle.dumps(singleton1)
singleton2 = pickle.loads(pickled)
assert singleton1 is singleton2
```

### Security: Restricted Unpickling

```python
from optype.pickle import CanReduce
import pickle

class SafeObject(CanReduce):
    ALLOWED_ATTRS = {'value', 'name'}
    
    def __init__(self, value: int, name: str) -> None:
        self.value = value
        self.name = name
    
    def __reduce__(self) -> tuple[type, tuple[int, str]]:
        return (self.__class__, (self.value, self.name))
    
    def __setstate__(self, state: dict[str, object]) -> None:
        \"\"\"Only restore allowed attributes.\"\"\"
        for key, val in state.items():
            if key in self.ALLOWED_ATTRS:
                setattr(self, key, val)
            else:
                raise ValueError(f"Attribute {key} not allowed")
```

## Protocol Combinations

Common combinations for different use cases:

### Stateful Objects

```python
class MyClass(CanGetstate, CanSetstate):
    # Custom state serialization
    pass
```

### Immutable Objects

```python
class ImmutableClass(CanGetnewargs):
    # Serialize constructor arguments
    pass
```

### Complex Objects

```python
class ComplexClass(CanReduceEx, CanGetstate):
    # Protocol-aware with custom state
    pass
```

## Pickle Protocol Versions

Different Python versions support different pickle protocols:

- **Protocol 0**: ASCII-only, compatible with very old Python
- **Protocol 1**: Binary format from Python 1.x
- **Protocol 2**: Python 2.3+ with efficient pickling of new-style classes
- **Protocol 3**: Python 3.0+ with explicit bytes support
- **Protocol 4**: Python 3.4+ with large object support
- **Protocol 5**: Python 3.8+ with out-of-band data

Use `CanReduceEx` to handle different protocols appropriately.

## Type Parameters

These protocols don't have type parameters but work with:

- **State types**: Any object returned by `__getstate__`
- **Argument types**: Tuples passed to `__new__` or `__init__`
- **Return types**: Reconstruction tuples from `__reduce__`

## Important Notes

!!! warning "Security"
Never unpickle data from untrusted sources. Pickle can execute arbitrary code during unpickling. Use `json` for untrusted data.

!!! tip "Performance"
Use protocol 5 (or the latest available) for best performance with large objects.

!!! note "Compatibility"
`CanReduce` is the most basic protocol. Implement `CanReduceEx` for protocol-specific optimizations.

## Related Protocols

- **[Copy](copy.md)**: Object copying protocols
- **[JSON](json.md)**: Safe serialization alternative
- **[Dataclasses](dataclasses.md)**: Automatic serialization support
