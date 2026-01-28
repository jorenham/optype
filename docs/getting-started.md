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
from typing import TypeVar

T = TypeVar("T")

def twice(x: T) -> T:
    return 2 * x
```

However, this has several problems:

1. **Type transformation**: `twice(True) == 2` changes the type from `bool` to `int`
2. **Type transformation**: `twice((1, 2)) == (1, 2, 1, 2)` changes from a 2-tuple to a 4-tuple
3. **Limited to known types**: It doesn't account for custom types with `__rmul__` methods

## The optype Solution

`optype` provides protocols for special methods. For multiplication, we can use `CanRMul[T, R]`:

- `T` is the type of the left operand (in `2 * x`, this is `Literal[2]`)
- `R` is the return type

=== "Python 3.12+"

    ```python
    from typing import Literal
    from optype import CanRMul

    type Two = Literal[2]
    type RMul2[R] = CanRMul[Two, R]


    def twice[R](x: RMul2[R]) -> R:
        return 2 * x
    ```

=== "Python 3.11"

    ```python
    from typing import Literal, TypeAlias, TypeVar
    from optype import CanRMul

    R = TypeVar("R")
    Two: TypeAlias = Literal[2]
    RMul2: TypeAlias = CanRMul[Two, R]


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

## Runtime Type Checking

All `optype` protocols are runtime-checkable. This means you can use `isinstance()` to check if an object supports an operation:

```python
from optype import CanRMul

if isinstance(x, CanRMul):
    result = 2 * x
```

This is particularly useful for handling different types:

=== "Python 3.12+"

    ```python
    from typing import Literal
    from optype import CanRMul, CanMul

    type Two = Literal[2]
    type Mul2[R] = CanMul[Two, R]
    type RMul2[R] = CanRMul[Two, R]
    type CMul2[R] = Mul2[R] | RMul2[R]


    def twice2[R](x: CMul2[R]) -> R:
        if isinstance(x, CanRMul):
            return 2 * x
        else:
            return x * 2
    ```

=== "Python 3.11"

    ```python
    from typing import Literal, TypeAlias, TypeVar
    from optype import CanRMul, CanMul

    R = TypeVar("R")
    Two: TypeAlias = Literal[2]
    Mul2: TypeAlias = CanMul[Two, R]
    RMul2: TypeAlias = CanRMul[Two, R]
    CMul2: TypeAlias = Mul2[R] | RMul2[R]


    def twice2(x: CMul2[R]) -> R:
        if isinstance(x, CanRMul):
            return 2 * x
        else:
            return x * 2
    ```

## Working with Custom Types

`optype` protocols work seamlessly with custom types:

```python
from typing import Literal
from optype import CanRMul

type Two = Literal[2]


class MyNumber:
    def __init__(self, value: int):
        self.value = value

    def __rmul__(self, other: Two) -> str:
        return f"{other} * {self.value}"


def twice[R](x: CanRMul[Two, R]) -> R:
    return 2 * x


result = twice(MyNumber(42))  # -> str
print(result)  # "2 * 42"
```

## The Five Flavors of optype

`optype` provides five categories of types:

### 1. `Just[T]` - Exact Type Matching

`Just[T]` accepts only instances of `T` itself, rejecting strict subtypes.

```python
import optype as op


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

### 2. `Can*` - What Can Be Done

Protocols describing what operations are supported. Each `Can*` protocol implements a single special method.

```python
import optype as op

# These all work:
isinstance(42, op.CanAbs)       # True (has __abs__)
isinstance("hi", op.CanAdd)     # True (has __add__)
isinstance([1], op.CanGetitem)  # True (has __getitem__)
```

### 3. `Has*` - What Attributes Exist

Protocols for special attributes.

```python
import optype as op


def get_name(obj: op.HasName) -> str:
    return obj.__name__


get_name(str)           # "str"
get_name(lambda: None)  # "<lambda>"
```

### 4. `Does*` - Operator Types

Types for operators themselves (not operands).

```python
import optype as op

# DoesAbs is the type of abs()
my_abs: op.DoesAbs = abs
```

### 5. `do_*` - Typed Operator Implementations

Correctly-typed operator implementations.

```python
import optype as op

# do_abs is a typed version of abs()
result = op.do_abs(-5)  # -> int
```

## Common Patterns

### Accepting Multiple Operations

```python
from optype import CanAdd, CanSub


def process(x: CanAdd[int, float] & CanSub[int, float]) -> float:
    """Accept types that support both addition and subtraction."""
    return (x + 1) - 1
```

### Generic Container Operations

```python
from optype import CanGetitem, CanLen


def first_item[T](container: CanGetitem[int, T] & CanLen) -> T | None:
    """Get first item from any indexable container."""
    if len(container) == 0:
        return None
    return container[0]


first_item([1, 2, 3])      # -> int | None
first_item("hello")        # -> str | None
first_item((1.0, 2.0))     # -> float | None
```

### Type-Safe Iteration

```python
from optype import CanIter, CanNext


def consume[T](it: CanIter[CanNext[T]]) -> list[T]:
    """Consume an iterable into a list."""
    return [item for item in it]
```

## Working with NumPy

If you have NumPy installed, `optype.numpy` provides extensive typing support:

```python
import numpy as np
import optype.numpy as onp


def normalize(arr: onp.Array2D[onp.floating]) -> onp.Array2D[np.float64]:
    """Normalize a 2D array of floats."""
    return arr / np.linalg.norm(arr)
```

See the [NumPy reference](reference/numpy/index.md) for complete documentation.

## Best Practices

### 1. Use Precise Types

Instead of accepting `Any`, use specific protocols:

```python
# ❌ Too generic
def process(x: Any) -> Any: ...

# ✓ Precise
from optype import CanFloat

def process(x: CanFloat) -> float:
    return float(x)
```

### 2. Combine Protocols

Use intersection types for multiple requirements:

```python
from optype import CanIter, CanLen


def process_sequence[T](
    seq: CanIter[T] & CanLen
) -> tuple[int, list[T]]:
    return len(seq), list(seq)
```

### 3. Runtime Validation

Use `isinstance()` for runtime checks:

```python
from optype import CanAdd


def smart_add(x: object, y: object) -> object:
    if isinstance(x, CanAdd):
        return x + y
    raise TypeError(f"{type(x)} doesn't support addition")
```

### 4. Document Type Parameters

Make it clear what your type parameters mean:

```python
from optype import CanMul


def scale[T, R](
    value: CanMul[T, R],  # The value to scale
    factor: T,             # The scaling factor
) -> R:                    # The scaled result
    return value * factor
```

## Next Steps

- Explore the [Core Types Reference](reference/core/just.md) for detailed API documentation
- Check out the [Standard Library Modules](reference/stdlib/copy.md) for more specialized protocols
- Learn about [NumPy typing](reference/numpy/index.md) for array operations
