# Async Iteration

Protocols for asynchronous iteration (`async for`).

## Overview

Just like the synchronous iteration protocols, the async iteration protocols in `optype` provide a complete alternative to `collections.abc.AsyncIterator` and `collections.abc.AsyncIterable`.

The protocols correspond to the `__aiter__` and `__anext__` special methods used in `async for` loops.

| Function   | Protocol                 |
| ---------- | ------------------------ |
| `anext(_)` | `CanANext[+V]`           |
| `aiter(_)` | `CanAIter[+R: CanANext]` |

## Variants

Additionally, there is `CanAIterSelf[R]` which expects both:

- `__aiter__() -> Self`
- `__anext__() -> V`

## Notes

Technically, `__anext__` can return any type, and `anext()` will pass it along. While you *could* return a non-awaitable from `__anext__`, this is not recommended for typical use cases.

## Examples

```python
import optype as op
```

```python
async def consume(async_iter: op.CanAIter[object]) -> None:
    """Consume an async iterable."""
    while True:
        try:
            item = await anext(async_iter)
            print(item)
        except StopAsyncIteration:
            break
```

```python
class AsyncCounter(op.CanAIterSelf[int]):
    def __init__(self, max: int):
        self.current = 0
        self.max = max
    
    def __aiter__(self) -> Self:
        return self
    
    async def __anext__(self) -> int:
        if self.current < self.max:
            self.current += 1
            return self.current
        raise StopAsyncIteration
```
