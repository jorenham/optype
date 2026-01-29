# optype.io

Protocols for Python's `io` module and file I/O operations.

## Overview

Runtime-accessible protocols and type aliases for I/O operations, providing consistent naming and better accessibility than `_typeshed` analogues.

## Protocols

| Protocol                      | Method Signature     | Replaces                              |
| ----------------------------- | -------------------- | ------------------------------------- |
| `CanFSPath[+T: str \| bytes]` | `__fspath__() -> T`  | `os.PathLike[AnyStr]`                 |
| `CanRead[+T]`                 | `read() -> T`        | -                                     |
| `CanReadN[+T]`                | `read(int) -> T`     | `_typeshed.SupportsRead[+T]`          |
| `CanReadline[+T]`             | `readline() -> T`    | `_typeshed.SupportsNoArgReadline[+T]` |
| `CanReadlineN[+T]`            | `readline(int) -> T` | `_typeshed.SupportsReadline[+T]`      |
| `CanWrite[-T, +RT]`           | `write(T) -> RT`     | `_typeshed.SupportsWrite[-T]`         |
| `CanFlush[+RT]`               | `flush() -> RT`      | `_typeshed.SupportsFlush`             |
| `CanFileno`                   | `fileno() -> int`    | `_typeshed.HasFileno`                 |

## Type Aliases

| Alias                      | Definition          | Replaces                               |
| -------------------------- | ------------------- | -------------------------------------- |
| `ToPath[+T: str \| bytes]` | `T \| CanFSPath[T]` | `_typeshed.StrPath`, `BytesPath`, etc. |
| `ToFileno`                 | `int \| CanFileno`  | `_typeshed.FileDescriptorLike`         |

## Usage Examples

### Path Handling

```python
from pathlib import Path
from optype.io import ToPath, CanFSPath

def read_file(path: ToPath[str]) -> str:
    \"\"\"Accept str or path-like object.\"\"\"
    if isinstance(path, str):
        filepath = path
    else:
        filepath = path.__fspath__()
    
    with open(filepath) as f:
        return f.read()

# Works with both
content = read_file("file.txt")
content = read_file(Path("file.txt"))
```

### Custom Path-Like Object

```python
from optype.io import CanFSPath

class DatabasePath(CanFSPath[str]):
    def __init__(self, table: str, record_id: int):
        self.table = table
        self.record_id = record_id
    
    def __fspath__(self) -> str:
        return f"db://{self.table}/{self.record_id}"

# Can be used where paths are expected
db_path = DatabasePath("users", 123)
path_str = db_path.__fspath__()  # "db://users/123"
```

### File-Like Objects

```python
from optype.io import CanRead, CanWrite, CanFlush

class StringBuffer(CanRead[str], CanWrite[str, int], CanFlush[None]):
    def __init__(self):
        self._buffer = []
    
    def read(self) -> str:
        result = "".join(self._buffer)
        self._buffer.clear()
        return result
    
    def write(self, data: str) -> int:
        self._buffer.append(data)
        return len(data)
    
    def flush(self) -> None:
        pass  # In-memory, nothing to flush

# Usage
buf = StringBuffer()
buf.write("Hello")
buf.write(" World")
print(buf.read())  # "Hello World"
```

### Generic File Reader

```python
from optype.io import CanReadN

def read_chunks(file: CanReadN[bytes], chunk_size: int = 1024) -> list[bytes]:
    \"\"\"Read file in chunks.\"\"\"
    chunks = []
    while True:
        chunk = file.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
    return chunks

# Works with any readable
with open("data.bin", "rb") as f:
    chunks = read_chunks(f)
```

### File Descriptor Operations

```python
from optype.io import ToFileno, CanFileno
import os

def get_file_size(fd: ToFileno) -> int:
    \"\"\"Get size from file descriptor or file-like object.\"\"\"
    if isinstance(fd, int):
        fileno = fd
    else:
        fileno = fd.fileno()
    
    return os.fstat(fileno).st_size

# Works with both
with open("file.txt") as f:
    size = get_file_size(f)        # File object
    size = get_file_size(f.fileno())  # Raw fd
```

## Type Parameters

### Protocols

- **`T` (Path Type)**: Covariant - `str` or `bytes`
- **`T` (Read Type)**: Covariant - what `read()` returns
- **`T` (Write Type)**: Contravariant - what `write()` accepts
- **`RT` (Return Type)**: Covariant - return type of `write()` or `flush()`

### Type Aliases

- **`T` (ToPath)**: Covariant - `str` or `bytes` for path types

## Related Protocols

- **[String](string.md)**: String literal types
- **[JSON](json.md)**: JSON I/O type aliases
- **[Pickle](pickle.md)**: Pickle serialization protocols
