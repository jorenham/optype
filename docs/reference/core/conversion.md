# Builtin Type Conversion

Protocols for Python's builtin type conversion functions.

## Overview

The return type of these special methods is **invariant**. Python will raise an error if some other (sub)type is returned. This is why these `optype` interfaces don't accept generic type arguments.

| Builtin         | Function     | Protocol                   |
| --------------- | ------------ | -------------------------- |
| `complex(_)`    | `do_complex` | `CanComplex`               |
| `float(_)`      | `do_float`   | `CanFloat`                 |
| `int(_)`        | `do_int`     | `CanInt`                   |
| `bool(_)`       | `do_bool`    | `CanBool[+R: bool = bool]` |
| `bytes(_)`      | `do_bytes`   | `CanBytes`                 |
| `str(_)`        | `do_str`     | `CanStr`                   |
| `repr(_)`       | `do_repr`    | `CanRepr`                  |
| `format(_, x)`  | `do_format`  | `CanFormat`                |
| `hash(_)`       | `do_hash`    | `CanHash`                  |
| `_.__index__()` | `do_index`   | `CanIndex`                 |

## Protocols

```python
import optype as op
```

### CanComplex

Protocol for types that can be converted to `complex`.

```python
def to_complex(x: op.CanComplex) -> complex:
    return complex(x)
```

**Method:** `__complex__(self) -> complex`

### CanFloat

Protocol for types that can be converted to `float`.

```python
def to_float(x: op.CanFloat) -> float:
    return float(x)
```

**Method:** `__float__(self) -> float`

### CanInt

Protocol for types that can be converted to `int`.

```python
def to_int(x: op.CanInt) -> int:
    return int(x)
```

**Method:** `__int__(self) -> int`

### CanBool

Protocol for types that can be converted to `bool`.

```python
def to_bool(x: op.CanBool) -> bool:
    return bool(x)
```

**Method:** `__bool__(self) -> bool`

!!! note
`CanBool` is covariant in its return type: `CanBool[+R: bool = bool]`.
While most implementations return `bool`, the protocol allows for subtypes.

### CanBytes

Protocol for types that can be converted to `bytes`.

```python
def to_bytes(x: op.CanBytes) -> bytes:
    return bytes(x)
```

**Method:** `__bytes__(self) -> bytes`

### CanStr

Protocol for types that can be converted to `str`.

```python
def to_str(x: op.CanStr) -> str:
    return str(x)
```

**Method:** `__str__(self) -> str`

### CanRepr

Protocol for types that have a string representation via `repr()`.

```python
def get_repr(x: op.CanRepr) -> str:
    return repr(x)
```

**Method:** `__repr__(self) -> str`

### CanFormat

Protocol for types that can be formatted with `format()`.

```python
def format_value(x: op.CanFormat, fmt: str = "") -> str:
    return format(x, fmt)
```

**Method:** `__format__(self, format_spec: str) -> str`

### CanHash

Protocol for hashable types.

```python
def get_hash(x: op.CanHash) -> int:
    return hash(x)
```

**Method:** `__hash__(self) -> int`

### CanIndex

Protocol for types that implement `__index__()`.

```python
def as_index(x: op.CanIndex) -> int:
    return x.__index__()
```

**Method:** `__index__(self) -> int`

**See:** [Python documentation](https://docs.python.org/3/reference/datamodel.html#object.__index__)

## Operator Functions

### do_complex

Typed implementation of `complex()`.

```python
x: op.CanComplex = 3.14
result = op.do_complex(x)  # -> complex
```

**Type:** `DoesComplex`

### do_float

Typed implementation of `float()`.

```python
x: op.CanFloat = 42
result = op.do_float(x)  # -> float
```

**Type:** `DoesFloat`

### do_int

Typed implementation of `int()`.

```python
x: op.CanInt = 3.14
result = op.do_int(x)  # -> int
```

**Type:** `DoesInt`

### Other do_ functions

Similarly: `do_bool`, `do_bytes`, `do_str`, `do_repr`, `do_format`, `do_hash`, `do_index`

## Examples

### Custom Type with Multiple Conversions

```python
class CustomNumber(op.CanInt, op.CanFloat, op.CanStr):
    def __init__(self, value: float):
        self.value = value
    
    def __int__(self) -> int:
        return int(self.value)
    
    def __float__(self) -> float:
        return self.value
    
    def __str__(self) -> str:
        return f"CustomNumber({self.value})"


num = CustomNumber(3.14)
print(int(num))    # 3
print(float(num))  # 3.14
print(str(num))    # "CustomNumber(3.14)"
```

### Type-Safe Conversion Function

```python
def safe_to_number(x: op.CanInt | op.CanFloat) -> int | float:
    """Convert to int if possible, otherwise to float."""
    if isinstance(x, op.CanInt):
        return int(x)
    return float(x)
```

## See Also

- [Just](just.md) - Exact type matching
- [Rich Relations](relations.md) - Comparison operators
