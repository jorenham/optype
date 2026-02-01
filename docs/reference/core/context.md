# Context Managers

Protocols for context managers used with `with` and `async with` statements.

## Overview

Context managers in Python allow for proper resource management and cleanup. The `optype` library provides protocols for both synchronous (`with`) and asynchronous (`async with`) context managers through dedicated `Can*` and `CanAsync*` protocols.

## Synchronous Context Managers

Protocols for the `with` statement.

| Operation      | Protocol                                             |
| -------------- | ---------------------------------------------------- |
| `__enter__`    | `CanEnter[+C]` or `CanEnterSelf`                     |
| `__exit__`     | `CanExit[+R = None]`                                 |
| `with _ as c:` | `CanWith[+C, +R = None]` or `CanWithSelf[+R = None]` |

### Aliases

- **`CanEnterSelf`**: Shorthand for `CanEnter[Self]` - when `__enter__` returns self
- **`CanWithSelf[+R = None]`**: Shorthand for `CanWith[Self, +R]` - context manager that returns self

## Asynchronous Context Managers

Protocols for the `async with` statement.

| Operation            | Protocol                                                       |
| -------------------- | -------------------------------------------------------------- |
| `__aenter__`         | `CanAEnter[+C]` or `CanAEnterSelf`                             |
| `__aexit__`          | `CanAExit[+R = None]`                                          |
| `async with _ as c:` | `CanAsyncWith[+C, +R = None]` or `CanAsyncWithSelf[+R = None]` |

### Aliases

- **`CanAEnterSelf`**: Shorthand for `CanAEnter[Self]` - when `__aenter__` returns self
- **`CanAsyncWithSelf[+R = None]`**: Shorthand for `CanAsyncWith[Self, +R]` - async context manager that returns self

## Examples

```python
import optype as op
```

### File-like Context Manager

```python
from typing import TextIO

def read_file_content(f: op.CanWith[TextIO, None]) -> str:
    """Read content from any context manager that yields a file."""
    with f as file:
        return file.read()
```

### Custom Resource Manager

```python
class DatabaseConnection(op.CanEnter[object], op.CanExit[None]):
    def __init__(self, url: str):
        self.url = url
        self.connected = False
    
    def __enter__(self) -> object:
        print(f"Connecting to {self.url}")
        self.connected = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        print("Disconnecting")
        self.connected = False


with DatabaseConnection("postgres://localhost") as db:
    print(f"Connected: {db.connected}")  # True
```

### Self-returning Context Manager

```python
class Lock(op.CanEnterSelf):
    def __init__(self):
        self.acquired = False
    
    def __enter__(self) -> "Lock":
        self.acquired = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.acquired = False


lock = Lock()
with lock as l:
    print(f"Lock held: {l.acquired}")  # True
print(f"Lock held: {lock.acquired}")  # False
```

### Async Context Manager

```python
import asyncio

class AsyncConnection(op.CanAsyncWith[object, None]):
    def __init__(self, name: str):
        self.name = name
    
    async def __aenter__(self) -> object:
        await asyncio.sleep(0.1)
        print(f"Opened {self.name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await asyncio.sleep(0.1)
        print(f"Closed {self.name}")


async def main():
    async with AsyncConnection("db") as conn:
        print("Using connection")


asyncio.run(main())
```

### Generic Context Manager Handler

```python
from typing import TypeVar, Generic

T = TypeVar("T")
R = TypeVar("R")

class ContextHandler(Generic[T, R]):
    def execute[S](self, ctx: op.CanWith[T, R], func: (T) -> S) -> S:
        """Execute a function within a context."""
        with ctx as resource:
            return func(resource)


handler = ContextHandler()

def process_file(f) -> int:
    return len(f.read())

# Usage with file-like objects
from io import StringIO
content = handler.execute(StringIO("hello"), process_file)
print(content)  # 5
```

## Exception Handling

Both `__exit__` and `__aexit__` receive exception information if an exception occurs within the context:

```python
class ErrorHandler(op.CanWith[None, bool]):
    def __enter__(self) -> None:
        print("Entering")
        return None
    
    def __exit__(self, exc_type: type | None, exc_val: Exception | None,
                 exc_tb: object | None) -> bool:
        if exc_type is not None:
            print(f"Caught exception: {exc_type.__name__}")
            return True  # Suppress the exception
        print("Exiting normally")
        return False


with ErrorHandler():
    raise ValueError("test")
print("No exception propagated")
```

## Type Parameters

- **`+C`**: The type yielded by `__enter__` (covariant)
- **`+R`**: The return type of `__exit__` (covariant, defaults to `None`)

## Related Protocols

- **[Awaitables](awaitables.md)**: For `await` expressions used in async context managers
