# Callables

Protocols for callable objects.

## Overview

The `CanCall` protocol describes objects that can be called (invoked) as functions. Unlike `collections.abc.Callable`, `CanCall` is purely runtime-checkable and doesn't rely on esoteric implementation tricks.

`optype` also provides `do_call` as a correctly-typed wrapper around calling objects, giving you an operator for the call operation.

| Operation            | Function  | Protocol             |
| -------------------- | --------- | -------------------- |
| `_(*args, **kwargs)` | `do_call` | `CanCall[**Tss, +R]` |

## Type Parameters

`CanCall` uses `ParamSpec` (**Tss) to capture the parameter signature and return type (`+R`):

- `**Tss`: Captures the parameter specification of the callable
- `+R`: The return type of the call

## Examples

```python
import optype as op
```

### Basic Callable Protocol

```python
def invoke(func: op.CanCall[..., int]) -> int:
    """Invoke any callable that returns an int."""
    return func()


result = invoke(lambda: 42)
print(result)  # 42
```

### Generic Callable Function

```python
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def repeat_call(func: op.CanCall[P, R], *args: P.args, **kwargs: P.kwargs) -> tuple[R, R]:
    """Call a function twice with the same arguments."""
    first = func(*args, **kwargs)
    second = func(*args, **kwargs)
    return (first, second)


def greet(name: str) -> str:
    return f"Hello, {name}!"


result = repeat_call(greet, "Alice")
print(result)  # ("Hello, Alice!", "Hello, Alice!")
```

### Custom Callable Class

```python
class Multiplier(op.CanCall[[int, int], int]):
    def __init__(self, factor: int):
        self.factor = factor
    
    def __call__(self, x: int, y: int) -> int:
        return (x + y) * self.factor


mult = Multiplier(10)
print(mult(2, 3))  # 50
```

### Function Composition

```python
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")

def compose(f: op.CanCall[P, R], g: op.CanCall[[R], S]) -> op.CanCall[P, S]:
    """Compose two functions: (g ∘ f)(x) = g(f(x))"""
    def composed(*args: P.args, **kwargs: P.kwargs) -> S:
        return g(f(*args, **kwargs))
    return composed


def add_ten(x: int) -> int:
    return x + 10

def double(x: int) -> int:
    return x * 2

# Compose: double(add_ten(x))
double_after_add = compose(add_ten, double)
print(double_after_add(5))  # (5 + 10) * 2 = 30
```

## Type Checking Notes

Some type checkers may accept `collections.abc.Callable` in more places than `optype.CanCall` due to the lack of co/contravariance specification for `ParamSpec`. If you encounter such limitations, consider opening an issue for discussion.

## Related Protocols

- **[Awaitables](awaitables.md)**: For async callable objects that return awaitables
