# Rich Relations

Protocols for Python's comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`).

## Overview

The "rich" comparison special methods often return a `bool`. However, instances of any type can be returned (e.g., a numpy array). This is why the corresponding `optype.Can*` interfaces accept a second type argument for the return type, that defaults to `bool` when omitted.

The first type parameter matches the passed method argument, i.e., the right-hand side operand, denoted here as `x`.

## Operators & Protocols

| Expression | Reflected | Function | Type     | Method   | Protocol                        |
| ---------- | --------- | -------- | -------- | -------- | ------------------------------- |
| `_ == x`   | `x == _`  | `do_eq`  | `DoesEq` | `__eq__` | `CanEq[-T = object, +R = bool]` |
| `_ != x`   | `x != _`  | `do_ne`  | `DoesNe` | `__ne__` | `CanNe[-T = object, +R = bool]` |
| `_ < x`    | `x > _`   | `do_lt`  | `DoesLt` | `__lt__` | `CanLt[-T, +R = bool]`          |
| `_ <= x`   | `x >= _`  | `do_le`  | `DoesLe` | `__le__` | `CanLe[-T, +R = bool]`          |
| `_ > x`    | `x < _`   | `do_gt`  | `DoesGt` | `__gt__` | `CanGt[-T, +R = bool]`          |
| `_ >= x`   | `x <= _`  | `do_ge`  | `DoesGe` | `__ge__` | `CanGe[-T, +R = bool]`          |

## Type Parameters

### First Parameter: `-T`

The contravariant type parameter `T` represents the type of the right-hand side operand (the argument passed to the comparison method).

For equality operations (`CanEq` and `CanNe`), this defaults to `object`, allowing comparison with any type.

For ordering operations (`CanLt`, `CanLe`, `CanGt`, `CanGe`), this parameter has no default and must be specified.

### Second Parameter: `+R`

The covariant type parameter `R` represents the return type of the comparison operation. It defaults to `bool` for all comparison protocols.

This allows for non-boolean return types, which is particularly useful for:

- NumPy arrays that return element-wise comparison results
- Custom types that implement comparison semantics
- Symbolic mathematics libraries

## Examples

### Basic Comparisons

```python
import optype as op

def is_positive(x: op.CanGt[int]) -> bool:
    """Check if a value is positive."""
    return x > 0

is_positive(42)   # True
is_positive(-5)   # False
```

### Non-Boolean Returns

```python
import numpy as np
import optype as op

def element_wise_greater(
    arr: op.CanGt[int, np.ndarray[tuple[int, ...], np.dtype[np.bool_]]]
) -> np.ndarray[tuple[int, ...], np.dtype[np.bool_]]:
    """Return element-wise comparison with zero."""
    return arr > 0

result = element_wise_greater(np.array([1, -2, 3]))
# Returns array([True, False, True])
```

### Combining Comparison Protocols

```python
from typing import Protocol
import optype as op

class Comparable[T](op.CanLt[T], op.CanGt[T], Protocol): ...

def clamp[T](value: Comparable[T], min_val: T, max_val: T) -> T:
    """Clamp value between min and max."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value

clamp(5, 0, 10)    # 5
clamp(-5, 0, 10)   # 0
clamp(15, 0, 10)   # 10
```

### Generic Comparison Functions

```python
import optype as op

def are_equal[T, R](a: op.CanEq[T, R], b: T) -> R:
    """Compare two values for equality."""
    return a == b

are_equal(42, 42)          # True (bool)
are_equal("hello", "hi")   # False (bool)
```

### Ordering Protocol

```python
from typing import Protocol
import optype as op

class Orderable[T](
    op.CanLt[T],
    op.CanLe[T],
    op.CanGt[T],
    op.CanGe[T],
    Protocol,
): ...

def is_between[T](value: Orderable[T], low: T, high: T) -> bool:
    """Check if value is between low and high (inclusive)."""
    return low <= value <= high

is_between(5, 0, 10)   # True
is_between(15, 0, 10)  # False
```
