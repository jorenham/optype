# Iteration

Protocols for iteration operations (`iter()`, `next()`, `for` loops).

## Overview

The iteration protocol allows objects to be used in `for` loops and other iteration contexts. Unlike `collections.abc.Iterable` and `collections.abc.Iterator`, the `optype` protocols provide better type safety and performance characteristics.

### Advantages over collections.abc

- **`CanIter[R]`** vs `Iterable[V]`: `CanIter` binds to the return type of `iter(_)`, allowing you to specify the exact iterator type returned, while `Iterable` only specifies the yielded value type.
- **`CanNext[V]`**: Does not require `__iter__` to be implemented, unlike `collections.abc.Iterator` which unnecessarily requires both methods. This improves `isinstance()` performance significantly.

| Function  | Protocol                       |
| --------- | ------------------------------ |
| `next(_)` | `CanNext[+V]`                  |
| `iter(_)` | `CanIter[+R: CanNext[object]]` |

## Protocols

### CanNext[+V]

Protocol for types that can be iterated over using `next()`. Implements `__next__` method.

```python
from optype import CanNext

def get_next_item(iterator: CanNext[int]) -> int:
    """Get the next item from an iterator."""
    return next(iterator)
```

**Method:** `__next__(self) -> V`

### CanIter[+R: CanNext[object]]

Protocol for types that can be converted to an iterator using `iter()`. Implements `__iter__` method.

```python
from optype import CanIter, CanNext

def iterate_items[V](iterable: CanIter[CanNext[V]]) -> list[V]:
    """Collect all items from an iterable."""
    return list(iter(iterable))
```

**Method:** `__iter__(self) -> R`

### CanIterSelf[V]

Convenience protocol for iterators that return themselves from `__iter__()`. Equivalent to `collections.abc.Iterator[V]` but without the ABC overhead.

## Examples

### Custom Iterator

```python
from optype import CanNext

class CountUp(CanNext[int]):
    def __init__(self, max: int):
        self.current = 0
        self.max = max
    
    def __next__(self) -> int:
        if self.current < self.max:
            self.current += 1
            return self.current
        raise StopIteration


counter = CountUp(3)
print(next(counter))  # 1
print(next(counter))  # 2
print(next(counter))  # 3
```

### Custom Iterable

```python
from optype import CanIter, CanNext

class Range(CanIter[CanNext[int]]):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
    
    def __iter__(self) -> "RangeIterator":
        return RangeIterator(self.start, self.end)


class RangeIterator(CanNext[int]):
    def __init__(self, start: int, end: int):
        self.current = start
        self.end = end
    
    def __next__(self) -> int:
        if self.current < self.end:
            result = self.current
            self.current += 1
            return result
        raise StopIteration


for i in Range(1, 4):
    print(i)  # 1, 2, 3
```

### Self-returning Iterator

```python
from optype import CanIterSelf

class Fibonacci(CanIterSelf[int]):
    def __init__(self, max_count: int):
        self.a = 0
        self.b = 1
        self.count = 0
        self.max_count = max_count
    
    def __iter__(self) -> "Fibonacci":
        return self
    
    def __next__(self) -> int:
        if self.count < self.max_count:
            result = self.a
            self.a, self.b = self.b, self.a + self.b
            self.count += 1
            return result
        raise StopIteration


fib = Fibonacci(5)
print(list(fib))  # [0, 1, 1, 2, 3]
```

### Generic Iteration Handler

```python
from typing import TypeVar
from optype import CanIter, CanNext

T = TypeVar("T")

def collect_all[V](iterable: CanIter[CanNext[V]]) -> list[V]:
    """Collect all items from any iterable."""
    result = []
    for item in iterable:
        result.append(item)
    return result


class SimpleList(CanIter[CanNext[int]]):
    def __init__(self, items: list[int]):
        self.items = items
    
    def __iter__(self):
        return SimpleListIterator(self.items)


class SimpleListIterator(CanNext[int]):
    def __init__(self, items: list[int]):
        self.items = items
        self.index = 0
    
    def __next__(self) -> int:
        if self.index < len(self.items):
            result = self.items[self.index]
            self.index += 1
            return result
        raise StopIteration


data = collect_all(SimpleList([1, 2, 3, 4, 5]))
print(data)  # [1, 2, 3, 4, 5]
```

### Infinite Iterator with Limit

```python
from optype import CanNext

class InfiniteCounter(CanNext[int]):
    def __init__(self):
        self.current = 0
    
    def __next__(self) -> int:
        result = self.current
        self.current += 1
        return result


def take[T](iterator: CanNext[T], n: int) -> list[T]:
    """Take the first n items from an iterator."""
    result = []
    for _ in range(n):
        try:
            result.append(next(iterator))
        except StopIteration:
            break
    return result


counter = InfiniteCounter()
print(take(counter, 5))  # [0, 1, 2, 3, 4]
print(take(counter, 3))  # [5, 6, 7]
```

## Type Parameters

- **Value (`+V`)**: Covariant - the type yielded by iteration
- **Result (`+R`)**: Covariant - the iterator type returned by `__iter__`

## Performance Characteristics

The `optype` iteration protocols have better performance than `collections.abc` because:

- `isinstance()` checks on protocols only need to check required methods
- `CanNext` only requires `__next__`, avoiding redundant method checks
- No ABC metaclass overhead from `collections.abc.ABC`

## Related Protocols

- **[Async Iteration](async-iteration.md)**: For `async for` loops
- **[Containers](containers.md)**: For length and indexing operations
