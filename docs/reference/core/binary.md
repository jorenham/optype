# Binary Operations

Protocols for Python's binary operators (`+`, `-`, `*`, `/`, `@`, `%`, `**`, `<<`, `>>`, `&`, `^`, `|`).

## Overview

In the [Python docs](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex), these are referred to as "arithmetic operations". But the operands aren't limited to numeric types, and because the operations aren't required to be commutative, might be non-deterministic, and could have side-effects. Classifying them "arithmetic" is, at the very least, a bit of a stretch.

Each binary operation has three protocol variants:

- `Can*[T, R]` - Standard form
- `Can*Self[T]` - Returns `typing.Self`
- `Can*Same[T?, R?]` - Accepts `Self | T`, returns `Self | R`

| Operator               | Protocols                                                                         |
| ---------------------- | --------------------------------------------------------------------------------- |
| `_ + x`                | `CanAdd[-T, +R = T]`<br>`CanAddSelf[-T]`<br>`CanAddSame[-T?, +R?]`                |
| `_ - x`                | `CanSub[-T, +R = T]`<br>`CanSubSelf[-T]`<br>`CanSubSame[-T?, +R?]`                |
| `_ * x`                | `CanMul[-T, +R = T]`<br>`CanMulSelf[-T]`<br>`CanMulSame[-T?, +R?]`                |
| `_ @ x`                | `CanMatmul[-T, +R = T]`<br>`CanMatmulSelf[-T]`<br>`CanMatmulSame[-T?, +R?]`       |
| `_ / x`                | `CanTruediv[-T, +R = T]`<br>`CanTruedivSelf[-T]`<br>`CanTruedivSame[-T?, +R?]`    |
| `_ // x`               | `CanFloordiv[-T, +R = T]`<br>`CanFloordivSelf[-T]`<br>`CanFloordivSame[-T?, +R?]` |
| `_ % x`                | `CanMod[-T, +R = T]`<br>`CanModSelf[-T]`<br>`CanModSame[-T?, +R?]`                |
| `divmod(_, x)`         | `CanDivmod[-T, +R]`                                                               |
| `_ ** x` / `pow(_, x)` | `CanPow2[-T, +R = T]`<br>`CanPowSelf[-T]`<br>`CanPowSame[-T?, +R?]`               |
| `pow(_, x, m)`         | `CanPow3[-T, -M, +R = int]`                                                       |
| `_ << x`               | `CanLshift[-T, +R = T]`<br>`CanLshiftSelf[-T]`<br>`CanLshiftSame[-T?, +R?]`       |
| `_ >> x`               | `CanRshift[-T, +R = T]`<br>`CanRshiftSelf[-T]`<br>`CanRshiftSame[-T?, +R?]`       |
| `_ & x`                | `CanAnd[-T, +R = T]`<br>`CanAndSelf[-T]`<br>`CanAndSame[-T?, +R?]`                |
| `_ ^ x`                | `CanXor[-T, +R = T]`<br>`CanXorSelf[-T]`<br>`CanXorSame[-T?, +R?]`                |
| `_ \| x`               | `CanOr[-T, +R = T]`<br>`CanOrSelf[-T]`<br>`CanOrSame[-T?, +R?]`                   |

## Protocol Variants Explained

### Standard Form: `Can*[-T, +R = T]`

The standard protocol accepts an operand of type `T` and returns type `R` (defaulting to `T`).

Example:

```python
import optype as op

def add_numbers(x: op.CanAdd[int, float]) -> float:
    return x + 5.0  # Returns float
```

### Self Form: `Can*Self[-T]`

Returns `typing.Self` for fluent interfaces. Method signature: `(self, rhs: T, /) -> Self`.

Example:

```python
import optype as op

class Builder(op.CanAddSelf[str]):
    def __init__(self, value: str = ""):
        self.value = value

    def __add__(self, other: str) -> "Builder":
        return Builder(self.value + other)
```

### Same Form: `Can*Same[-T?, +R?]`

Accepts `Self | T` and returns `Self | R`. Both `T` and `R` default to `typing.Never`.

To illustrate:

- `CanAddSelf[T]` implements `__add__` as `(self, rhs: T, /) -> Self`
- `CanAddSame[T, R]` implements it as `(self, rhs: Self | T, /) -> Self | R`
- `CanAddSame` (without `T` and `R`) as `(self, rhs: Self, /) -> Self`

Example:

```python
import optype as op

def combine(x: op.CanAddSame[int, float]) -> ...:
    # x can be added to itself OR to an int
    # and returns either itself OR a float
    return x + x  # Returns Self
    # OR
    return x + 5  # Returns Self | float
```

## Special Cases

### pow() with Optional Third Argument

!!! tip "pow() Special Cases"
Because `pow()` can take an optional third argument, `optype` provides:

    - `CanPow2[-T, +R = T]` for `pow(x, y)` 
    - `CanPow3[-T, -M, +R = int]` for `pow(x, y, m)`
    - `CanPow[-T, -M, +R, +RM]` as intersection type for both

    The full `CanPow` type is defined as:
    ```python
    type CanPow[-T, -M, +R, +RM] = CanPow2[T, R] & CanPow3[T, M, RM]
    ```

## Examples

### Basic Usage

```python
import optype as op

def double(x: op.CanMul[int, int]) -> int:
    """Double a value by multiplying by 2."""
    return x * 2

double(21)  # OK: returns 42
```

### Generic Return Types

```python
import optype as op

def add_one[R](x: op.CanAdd[int, R]) -> R:
    """Add 1 to any value that supports it."""
    return x + 1

add_one(41)     # -> int (42)
add_one(2.5)    # -> float (3.5)
add_one([1, 2]) # -> list[int] ([1, 2, 1])
```

### Combining Multiple Operations

```python
from typing import Protocol
import optype as op

class CanAddMul(
    op.CanAdd[int, float],
    op.CanMul[int, float],
    Protocol,
): ...

def calculate(x: CanAddMul) -> float:
    return (x + 1) * 2
```
