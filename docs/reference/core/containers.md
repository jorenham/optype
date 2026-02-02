# Containers

Protocols for container operations like indexing, length, membership testing, and iteration.

## Overview

Container protocols in `optype` describe the operations that allow objects to behave like sequences, mappings, and other container types. These include length checking, item access, modification, deletion, and membership testing.

| Operation             | Function         | Protocol                   |
| --------------------- | ---------------- | -------------------------- |
| `len(_)`              | `do_len`         | `CanLen`                   |
| `_.__length_hint__()` | `do_length_hint` | `CanLengthHint`            |
| `_[k]`                | `do_getitem`     | `CanGetitem[-K, +V]`       |
| `_.__missing__()`     | `do_missing`     | `CanMissing[-K, +D]`       |
| `_[k] = v`            | `do_setitem`     | `CanSetitem[-K, -V]`       |
| `del _[k]`            | `do_delitem`     | `CanDelitem[-K]`           |
| `k in _`              | `do_contains`    | `CanContains[-K = object]` |
| `reversed(_)`         | `do_reversed`    | `CanReversed[+R]`          |

## Composite Protocols

`optype` provides convenient combinations of related protocols:

- **`CanGetMissing[K, V, D=V]`**: Combines `CanGetitem` and `CanMissing` for mapping-like behavior with fallback values
- **`CanSequence[K: CanIndex | slice, V]`**: Combines `CanLen` and `CanGetitem`, providing a more flexible alternative to `collections.abc.Sequence[V]`

## Examples

```python
import optype as op
```

### Getting Container Length

```python
def get_size(container: op.CanLen) -> int:
    """Get the size of any container."""
    return len(container)


print(get_size([1, 2, 3]))      # 3
print(get_size("hello"))        # 5
print(get_size({1, 2, 3}))      # 3
```

### Indexing Operations

```python
class CanGetSetitem[K, V](op.CanGetitem[K, V], op.CanSetitem[K, V], Protocol[K, V]): ...

def swap_first_last[K, V](container: CanGetSetitem, first_key: K, last_key: K) -> None:
    """Swap the first and last items in a container."""
    first_val = container[first_key]
    container[first_key] = container[last_key]
    container[last_key] = first_val


data = {"a": 1, "z": 26}
swap_first_last(data, "a", "z")
print(data)  # {"a": 26, "z": 1}
```

### Membership Testing

```python
def has_admin[K](permissions: op.CanContains[K], role: K) -> bool:
    """Check if a role is in the permissions."""
    return role in permissions


print(has_admin(["admin", "user"], "admin"))     # True
print(has_admin(["user", "guest"], "admin"))     # False
```

### Reversing Collections

```python
from collections.abc import Iterator

def reverse_items[V](container: op.CanReversed[Iterator[V]]) -> list[V]:
    """Reverse a container's items."""
    return list(reversed(container))


print(reverse_items([1, 2, 3, 4]))  # [4, 3, 2, 1]
```

### Mapping with Fallback (Missing Keys)

```python
class CaseInsensitiveDict(op.CanGetMissing[str, str, str]):
    def __init__(self, data: dict[str, str]):
        self._data = {k.lower(): v for k, v in data.items()}
    
    def __getitem__(self, key: str) -> str:
        return self._data[key.lower()]
    
    def __missing__(self, key: str) -> str:
        return f"<Key {key} not found>"


d = CaseInsensitiveDict({"Name": "Alice", "Age": "30"})
print(d["NAME"])        # Alice
print(d["missing"])     # <Key missing not found>
```

### Custom Sequence

```python
from collections.abc import Iterator

class RangeSequence(op.CanSequence[int, int]):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
    
    def __len__(self) -> int:
        return max(0, self.end - self.start)
    
    def __getitem__(self, key: int | slice) -> int | list[int]:
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self.start + i for i in indices]
        return self.start + key


seq = RangeSequence(10, 15)
print(len(seq))      # 5
print(seq[2])        # 12
print(seq[1:4])      # [11, 12, 13]
```

## Type Variance

- **Key (`-K`)**: Contravariant - container accepts keys of this type or supertypes
- **Value (`+V`)**: Covariant - container returns values of this type or subtypes
- **Result (`+R`)**: Covariant - operation returns this type or subtypes

## Related Protocols

- **[Iteration](iteration.md)**: For `__iter__` and `__next__` methods
- **[Attributes](attributes.md)**: For accessing object attributes
