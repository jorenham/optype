# Getting Started

This guide will walk you through the core concepts of `optype` and show you how to use it effectively in your projects.

## The Problem with Generic Type Variables

Let's say you're writing a `twice(x)` function that evaluates `2 * x`. Implementing it is trivial:

```python
def twice(x):
    return 2 * x
```

But what about the type annotations? At first glance, you might think:

```python
def twice[T](x: T) -> T:
    return 2 * x
```

However, this has several problems:

1. **Type safety**: Calling `twice(None)` will raise, but type-checkers will accept it
2. **Type transformation**: `twice(True) == 2` changes the type from `bool` to `int`
3. **Type transformation**: `twice((1, 2)) == (1, 2, 1, 2)` changes from a 2-tuple to a 4-tuple
4. **Limited to known types**: It doesn't account for custom types with `__rmul__` methods```

## The optype Solution

`optype` provides protocols for special methods. For multiplication, we can use `CanRMul[T, R]`:

- `T` is the type of the left operand (in `2 * x`, this is `Literal[2]`)
- `R` is the return type of `__rmul__`

```python
import optype as op
```

=== "Python 3.12+"

    ```python
    from typing import Literal

    type Two = Literal[2]
    type RMul2[R] = op.CanRMul[Two, R]


    def twice[R](x: RMul2[R]) -> R:
        return 2 * x
    ```

=== "Python 3.11"

    ```python
    from typing import Literal, TypeAlias, TypeVar

    R = TypeVar("R")
    Two: TypeAlias = Literal[2]
    RMul2: TypeAlias = op.CanRMul[Two, R]


    def twice(x: RMul2[R]) -> R:
        return 2 * x
    ```

Now the type checker correctly understands:

```python
twice(2)           # -> int
twice(3.14)        # -> float
twice('I')         # -> str (because 'I' * 2 == 'II')
twice(True)        # -> int (because 2 * True == 2)
twice((42, True))  # -> tuple[int, bool, int, bool]
```

## Working with Custom Types

`optype` protocols work seamlessly with custom types:

```python
from typing import Literal

type Two = Literal[2]


class MyNumber:
    def __init__(self, value: int):
        self.value = value

    def __rmul__(self, other: Two) -> str:
        return f"{other} * {self.value}"


def twice[R](x: op.CanRMul[Two, R]) -> R:  
    return 2 * x  


result = twice(MyNumber(42))  # -> str
print(result)  # "2 * 42"
```

## The Five Flavors of optype

`optype` provides five categories of types:

### 1. `Just[T]` - Exact Type Matching

The invariant `Just[T]` type accepts only instances of `T` itself,
rejecting strict subtypes.

```python
def assert_int(x: op.Just[int]) -> int:
    assert type(x) is int
    return x


assert_int(42)     # ✓ OK
assert_int(False)  # ✗ Error: bool is a strict subtype of int
```

Use cases:

- Reject `bool` when you only want `int`
- Annotate sentinel objects: `_DEFAULT: op.JustObject = object()`
- Avoid unwanted type promotions

!!! warning "Important: Use `Just[T]` only for inputs, never outputs"

`Just[T]` should **only** be used in input positions (function parameters, constructor arguments, etc.).

    ```python
    # ✓ Correct: Just in input position
    def process(x: op.Just[int]) -> int:
        return x * 2

    # ✗ Wrong: Just in return position
    def get_value() -> op.Just[int]:  # Don't do this!
        return 42
    ```

### 2. `Can*` - What Can Be Done

Protocols describing what operations are **can** be used.
Each `Can*` protocol implements a single special "dunder" method.

```python
_: op.CanAbs[int] = 42         # abs(42) -> int  
_: CanAdd[str, str] = "hi"     # "hi" + "hi" -> str  
_: CanGetitem[int, int] = [1]  # [1][0] -> int
```

### 3. `Has*` - What Attributes Exist

Protocols for special attributes.

```python
def get_name(obj: op.HasName) -> str:
    return obj.__name__


get_name(str)           # ✔️ 
get_name(lambda: None)  # ✔️  
get_name(None)          # ❌
```

### 4. `Does*` - Operator Types

Types for operators themselves (not operands).

```python
# DoesAbs is the type of abs()
my_abs: op.DoesAbs = abs
```

### 5. `do_*` - Typed Operator Implementations

Correctly-typed operator implementations.

```python
# do_abs is a typed version of abs()
result = op.do_abs(-5)  # -> int
```

## Common Patterns

### Accepting Multiple Operations

```python
from typing import Protocol

class CanAddSub(op.CanAdd[int, float], op.CanSub[int, float], Protocol): ...

def process(x: CanAddSub) -> float:
    """Accept types that support both addition and subtraction."""
    return (x + 1) - 1
```

### Combining Protocols

Use intersection types to require multiple capabilities:

```python
from typing import Protocol
import optype as op

class CanAddAndMulIntFloat(op.CanAdd[int, float], op.CanMul[int, float], Protocol):
    pass


def process(x: CanAddAndMulIntFloat) -> float:
    return (x + 1) * 2
```

Some type checkers may support the `&` operator for more concise intersections.

### Union Types

Use union types (`|`) for alternative capabilities:

```python
import optype as op


def to_number(x: op.CanInt | op.CanFloat) -> int | float:
    if isinstance(x, CanInt):
        return int(x)
    return float(x)
```

### Generic Functions

Use type parameters for flexible generic functions:

```python
import optype as op


def add_one[T, R](x: op.CanAdd[T, R], one: T) -> R:
    return x + one
```

### Generic Container Operations

```python
def first_item[T](container: op.CanSequence[int, T]) -> T | None:
    """Get first item from any indexable container."""
    if len(container) == 0:
        return None
    return container[0]


first_item([1, 2, 3])      # -> int | None
```

## Working with NumPy

If you have NumPy installed, `optype.numpy` provides extensive typing support:

```python
import numpy as np
import optype.numpy as onp


def normalize[T: np.inexact](arr: onp.Array2D[T]) -> onp.Array2D[T]:  
    """Normalize a 2D array of real or complex numbers."""  
    return arr / np.linalg.norm(arr)
```

See the [NumPy reference](reference/numpy/index.md) for complete documentation.

## Next Steps

- Explore the [Core Types Reference](reference/core/just.md) for detailed API documentation
- Check out the [Standard Library Modules](reference/stdlib/copy.md) for more specialized protocols
- Learn about [NumPy typing](reference/numpy/index.md) for array operations
