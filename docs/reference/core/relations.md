# Rich Relations

Protocols for Python's comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`).

## Overview

The "rich" comparison special methods often return `bool`, but can return instances of any type (e.g., NumPy arrays). This is why the corresponding `optype.Can*` interfaces accept a second type argument for the return type, which defaults to `bool` when omitted.

The first type parameter matches the passed method argument (the right-hand side operand).

| Operator | Reflected | Protocol                        |
| -------- | --------- | ------------------------------- |
| `_ == x` | `x == _`  | `CanEq[-T = object, +R = bool]` |
| `_ != x` | `x != _`  | `CanNe[-T = object, +R = bool]` |
| `_ < x`  | `x > _`   | `CanLt[-T, +R = bool]`          |
| `_ <= x` | `x >= _`  | `CanLe[-T, +R = bool]`          |
| `_ > x`  | `x < _`   | `CanGt[-T, +R = bool]`          |
| `_ >= x` | `x <= _`  | `CanGe[-T, +R = bool]`          |

## Examples

```python
from optype import CanLt, CanGt


def clamp[T](value: CanLt[T] & CanGt[T], min_val: T, max_val: T) -> T:
    """Clamp value between min and max."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value
```

See the [full README](https://github.com/jorenham/optype/blob/master/README.md#rich-relations) for complete documentation.
