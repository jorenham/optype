# Just - Exact Type Matching

## Overview

`Just` is an invariant type "wrapper" where `Just[T]` only accepts instances of `T` and rejects instances of any strict subtypes of `T`.

## Available Types

| Type          | Accepts         |
| ------------- | --------------- |
| `Just[T]`     | `T`             |
| `JustInt`     | `int`           |
| `JustFloat`   | `float`         |
| `JustComplex` | `complex`       |
| `JustBytes`   | `bytes`         |
| `JustObject`  | `object`        |
| `JustDate`    | `datetime.date` |

## Key Behavior

!!! note "Important"
`Literal[""]` and `LiteralString` are **not** strict `str` subtypes and are therefore assignable to `Just[str]`. However, instances of `class S(str): ...` are **not** assignable to `Just[str]`.

!!! tip "Runtime Checkable"
The `Just{Bytes,Int,Float,Complex,Date,Object}` protocols are runtime-checkable:

    ```python
    isinstance(42, JustInt)      # True
    isinstance(True, JustInt)    # False (bool is a subtype of int)
    ```

## Use Cases

### 1. Rejecting bool when accepting int

Since `bool` is a strict subtype of `int` in Python, using `int` as a type hint will accept `bool` values. Use `JustInt` to prevent this:

```python
import optype as op


def assert_int(x: op.Just[int]) -> int:
    assert type(x) is int
    return x


assert_int(42)     # ✓ OK
assert_int(False)  # ✗ Type error: bool is rejected
```

### 2. Annotating Sentinel Objects

Sentinel values are often created as `object()` instances. Use `JustObject` to ensure only the specific sentinel is accepted:

```python
import optype as op

_DEFAULT = object()


def intmap(
    value: int,
    mapping: dict[int, int] | op.JustObject = _DEFAULT,
    /,
) -> int:
    # Same as: if type(mapping) is object
    if isinstance(mapping, op.JustObject):
        return value
    
    return mapping[value]


intmap(1)              # ✓ OK - uses default
intmap(1, {1: 42})     # ✓ OK - uses mapping
intmap(1, "invalid")   # ✗ Type error: str is rejected
```

### 3. Avoiding Type Promotions

Python's type system has special [type promotion rules](https://typing.readthedocs.io/en/latest/spec/special-types.html#special-cases-for-float-and-complex) for `float` and `complex`. Use `JustFloat` and `JustComplex` to avoid unwanted promotions:

```python
import optype as op


def precise_float(x: op.JustFloat) -> op.JustFloat:
    """Only accepts actual floats, not complex numbers."""
    return x * 2.0


precise_float(3.14)  # ✓ OK
precise_float(1+2j)  # ✗ Type error: complex is rejected
```

### 4. Date vs DateTime Distinction

Since `datetime.datetime` is a subtype of `datetime.date`, use `JustDate` when you specifically need dates without times:

```python
import optype as op
from datetime import date, datetime


def format_date(d: op.JustDate) -> str:
    """Format a date (not a datetime)."""
    return d.strftime("%Y-%m-%d")


format_date(date(2024, 1, 1))      # ✓ OK
format_date(datetime.now())        # ✗ Type error: datetime is rejected
```

## Implementation

The `Just` types use metaclasses to provide runtime type checking while maintaining full type-checker compatibility. This allows for both static and runtime type safety.

## Type Variance

All `Just` types are **invariant**. This means:

- `Just[int]` is not assignable to `Just[object]`
- `Just[bool]` is not assignable to `Just[int]`
- `Just[T]` is only compatible with `Just[T]` itself

## See Also

- [Type Conversion](conversion.md) - For converting between types
- [optype.typing](../stdlib/typing.md) - For additional type utilities
