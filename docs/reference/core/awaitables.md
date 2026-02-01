# Awaitables

Protocols for awaitable objects that can be used with the `await` keyword.

## Overview

The `CanAwait[R]` protocol describes objects that can be awaited. It corresponds to the `__await__` special method and provides an alternative to `collections.abc.Awaitable[R]`.

Unlike `collections.abc.Awaitable`, `CanAwait` is a pure interface protocol without being an abstract base class, making it suitable for use in type stubs (`.pyi` files).

| Operation | Protocol       |
| --------- | -------------- |
| `await _` | `CanAwait[+R]` |

## Examples

```python
import optype as op
```

### Basic Awaitable

```python
from typing import Coroutine

async def wait_for_result(awaitable: op.CanAwait[int]) -> int:
    """Wait for an awaitable to return an integer."""
    return await awaitable


async def main() -> None:
    async def get_number() -> int:
        return 42
    
    result = await wait_for_result(get_number())
    print(result)  # 42
```

### Custom Awaitable Class

```python
class DelayedValue(op.CanAwait[str]):
    def __init__(self, value: str, delay: float):
        self.value = value
        self.delay = delay
    
    def __await__(self):
        import asyncio
        yield from asyncio.sleep(self.delay).__await__()
        return self.value


async def example() -> None:
    result = await DelayedValue("Hello!", 0.5)
    print(result)  # "Hello!"
```

### Coroutine Integration

```python
import asyncio

def process_awaitable(coro: op.CanAwait[str]) -> asyncio.Task[str]:
    """Wrap an awaitable in a task."""
    return asyncio.create_task(coro)


async def fetch_data() -> str:
    await asyncio.sleep(0.1)
    return "data"


async def main() -> None:
    task = process_awaitable(fetch_data())
    result = await task
    print(result)  # "data"
```

## Related Protocols

- **[Async Iteration](async-iteration.md)**: For iterating over async iterables
- **[Callables](callables.md)**: For callable objects including coroutines
