# optype.typing

Type alias utilities and common type patterns.

## Overview

Utility type aliases and common type patterns for type annotations, including `Any*` type aliases, `Empty*` types, and `Literal` utilities.

## Type Alias Categories

### Any* Type Aliases

Broad type unions for common categories:

| Type Alias       | Equivalent To                               | Use Case                  |
| ---------------- | ------------------------------------------- | ------------------------- |
| `AnyInt`         | `int \| np.integer`                         | Any integer type          |
| `AnyFloat`       | `int \| float \| np.floating \| np.integer` | Any numeric float-like    |
| `AnyComplex`     | `AnyFloat \| complex \| np.complexfloating` | Any numeric type          |
| `AnyIterable[V]` | `Iterable[V] \| Iterator[V]`                | Any iterable or iterator  |
| `AnyLiteral`     | `int \| str \| bytes \| bool \| None`       | Literal-compatible values |

### Empty Types

Type constructors for empty collections:

| Type              | Description  | Example                |
| ----------------- | ------------ | ---------------------- |
| `EmptyTuple`      | `tuple[()]`  | Empty tuple literal    |
| `EmptyDict[K, V]` | `dict[K, V]` | Empty dict constructor |
| `EmptySet[T]`     | `set[T]`     | Empty set constructor  |

### Literal Types

| Type           | Description            |
| -------------- | ---------------------- |
| `LiteralBool`  | `Literal[False, True]` |
| `LiteralFalse` | `Literal[False]`       |
| `LiteralTrue`  | `Literal[True]`        |

## Usage Examples

### Numeric Type Flexibility

```python
from optype.typing import AnyInt, AnyFloat, AnyComplex
import numpy as np

def to_int(value: AnyInt) -> int:
    \"\"\"Convert any integer-like value to Python int.\"\"\"
    return int(value)

def to_float(value: AnyFloat) -> float:
    \"\"\"Convert any float-like value to Python float.\"\"\"
    return float(value)

def to_complex(value: AnyComplex) -> complex:
    \"\"\"Convert any numeric value to Python complex.\"\"\"
    return complex(value)

# Works with Python and NumPy types
to_int(42)                    # 42
to_int(np.int32(42))          # 42
to_float(3.14)                # 3.14
to_float(np.float64(3.14))    # 3.14
to_complex(1 + 2j)            # (1+2j)
to_complex(np.complex128(1 + 2j))  # (1+2j)
```

### Iterable Handling

```python
from optype.typing import AnyIterable

def process_items(items: AnyIterable[int]) -> list[int]:
    \"\"\"Process any iterable or iterator of integers.\"\"\"
    return [x * 2 for x in items]

# Works with iterables
process_items([1, 2, 3])           # [2, 4, 6]
process_items((1, 2, 3))           # [2, 4, 6]
process_items({1, 2, 3})           # [2, 4, 6]

# Works with iterators
process_items(iter([1, 2, 3]))     # [2, 4, 6]
process_items(range(1, 4))         # [2, 4, 6]
```

### Literal Value Validation

```python
from optype.typing import AnyLiteral
from typing import Literal

def make_literal(value: AnyLiteral) -> AnyLiteral:
    \"\"\"Process values that can be used in Literal types.\"\"\"
    return value

# Valid literal values
make_literal(42)         # int
make_literal("text")     # str
make_literal(b"bytes")   # bytes
make_literal(True)       # bool
make_literal(None)       # None

# Can be used in Literal types
MyLiteral = Literal[42, "text", b"bytes", True, None]
```

### Empty Collection Constructors

```python
from optype.typing import EmptyTuple, EmptyDict, EmptySet

def create_empty_tuple() -> EmptyTuple:
    \"\"\"Create an empty tuple.\"\"\"
    return ()

def create_empty_dict() -> EmptyDict[str, int]:
    \"\"\"Create an empty dict with specific types.\"\"\"
    return {}

def create_empty_set() -> EmptySet[int]:
    \"\"\"Create an empty set with specific element type.\"\"\"
    return set()

# Type checkers know these are empty
empty_tuple = create_empty_tuple()  # tuple[()]
empty_dict = create_empty_dict()    # dict[str, int] (empty)
empty_set = create_empty_set()      # set[int] (empty)
```

### Boolean Literal Types

```python
from optype.typing import LiteralBool, LiteralTrue, LiteralFalse

def process_bool(value: LiteralBool) -> str:
    \"\"\"Process a boolean literal.\"\"\"
    return "yes" if value else "no"

def assert_true(value: LiteralTrue) -> None:
    \"\"\"Only accepts True.\"\"\"
    assert value is True

def assert_false(value: LiteralFalse) -> None:
    \"\"\"Only accepts False.\"\"\"
    assert value is False

# Type-safe boolean handling
process_bool(True)   # "yes"
process_bool(False)  # "no"
assert_true(True)    # OK
assert_false(False)  # OK
```

### Generic Collection Builder

```python
from optype.typing import AnyIterable, EmptyDict
from collections.abc import Callable
from typing import TypeVar

K = TypeVar('K')
V = TypeVar('V')

def group_by(
    items: AnyIterable[V],
    key_func: Callable[[V], K],
) -> dict[K, list[V]]:
    \"\"\"Group items by key function.\"\"\"
    result: dict[K, list[V]] = {}
    for item in items:
        key = key_func(item)
        if key not in result:
            result[key] = []
        result[key].append(item)
    return result

# Usage with any iterable
group_by([1, 2, 3, 4, 5], lambda x: x % 2)
# {1: [1, 3, 5], 0: [2, 4]}

group_by("hello", lambda c: c.lower())
# {'h': ['h'], 'e': ['e'], 'l': ['l', 'l'], 'o': ['o']}
```

### Type-Safe Numeric Operations

```python
from optype.typing import AnyInt, AnyFloat, AnyComplex
import numpy as np

def add_numbers(a: AnyComplex, b: AnyComplex) -> complex:
    \"\"\"Add any two numeric values.\"\"\"
    return complex(a) + complex(b)

def multiply_ints(a: AnyInt, b: AnyInt) -> int:
    \"\"\"Multiply any two integer values.\"\"\"
    return int(a) * int(b)

def divide_floats(a: AnyFloat, b: AnyFloat) -> float:
    \"\"\"Divide any two float-like values.\"\"\"
    return float(a) / float(b)

# Works with mixed types
add_numbers(1, 2j)                        # (1+2j)
add_numbers(np.int32(1), np.float64(2))   # (3+0j)
multiply_ints(3, np.int64(4))             # 12
divide_floats(10, np.float32(2.5))        # 4.0
```

### Iterator Consumption

```python
from optype.typing import AnyIterable
from typing import TypeVar

T = TypeVar('T')

def consume(iterable: AnyIterable[T], n: int | None = None) -> None:
    \"\"\"Consume n items from iterable or all if n is None.\"\"\"
    if n is None:
        # Consume all
        for _ in iterable:
            pass
    else:
        # Consume n items
        for _, _ in zip(range(n), iterable):
            pass

# Works with any iterable or iterator
consume([1, 2, 3])           # Consume all
consume(iter([1, 2, 3]), 2)  # Consume first 2
consume(range(100), 10)      # Consume first 10
```

## Type Compatibility

### NumPy Integration

The `Any*` numeric types integrate seamlessly with NumPy:

```python
from optype.typing import AnyInt, AnyFloat
import numpy as np
from numpy.typing import NDArray

def process_array(arr: NDArray[AnyFloat]) -> NDArray[np.floating]:
    \"\"\"Process array with any float-like dtype.\"\"\"
    return np.asarray(arr, dtype=np.float64)

# Works with various dtypes
process_array(np.array([1, 2, 3]))           # int -> float
process_array(np.array([1.0, 2.0, 3.0]))     # float
process_array(np.array([1, 2, 3], dtype=np.float32))  # float32
```

### Literal Type Hierarchies

```python
from optype.typing import LiteralBool, LiteralTrue, LiteralFalse

# Type hierarchy
LiteralTrue <: LiteralBool
LiteralFalse <: LiteralBool
LiteralBool ≡ Literal[False, True]
```

## Common Patterns

### Optional Iterable Parameter

```python
from optype.typing import AnyIterable

def process(items: AnyIterable[int] | None = None) -> list[int]:
    \"\"\"Process items or return empty list.\"\"\"
    if items is None:
        return []
    return list(items)
```

### Numeric Accumulator

```python
from optype.typing import AnyComplex

def sum_all(*values: AnyComplex) -> complex:
    \"\"\"Sum any numeric values.\"\"\"
    return sum(complex(v) for v in values)
```

### Type-Safe Factory

```python
from optype.typing import EmptyDict

def make_counter() -> dict[str, int]:
    \"\"\"Create a counter dictionary.\"\"\"
    result: EmptyDict[str, int] = {}
    return result
```

## Related Types

- **[NumPy Scalar](../numpy/scalar.md)**: NumPy scalar types
- **[String](string.md)**: String literal types
- **[Inspect](inspect.md)**: Type introspection utilities
