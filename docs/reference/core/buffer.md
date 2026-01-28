# Buffer Types

Protocols for the buffer protocol and memory views.

## Overview

The buffer protocol is a Python mechanism for exposing a byte-oriented data buffer interface. It allows efficient access to the internal data of objects that support it, such as bytes, bytearrays, and arrays.

`optype` provides protocols for implementing the buffer protocol via the `__buffer__` and `__release_buffer__` special methods.

| Operation           | Protocol           |
| ------------------- | ------------------ |
| `memoryview(_)`     | `CanBuffer`        |
| `del memoryview(_)` | `CanReleaseBuffer` |

## Examples

### Implementing a Bufferable Type

```python
from optype import CanBuffer
import struct

class ByteArray(CanBuffer):
    def __init__(self, data: bytes):
        self._data = bytearray(data)
    
    def __buffer__(self, flags: int) -> memoryview:
        """Return a memoryview of the internal buffer."""
        return memoryview(self._data)


# Usage
ba = ByteArray(b"hello")
mv = memoryview(ba)
print(bytes(mv))  # b'hello'
```

### Working with Bufferable Objects

```python
from optype import CanBuffer

def copy_buffer_data(obj: CanBuffer) -> bytes:
    """Copy data from any bufferable object."""
    mv = memoryview(obj)
    return bytes(mv)


# Works with any object supporting __buffer__
data = copy_buffer_data(b"test")
print(data)  # b'test'
```

### Custom Buffer with Release

```python
from optype import CanBuffer, CanReleaseBuffer

class ManagedBuffer(CanBuffer, CanReleaseBuffer):
    def __init__(self, size: int):
        self._buffer = bytearray(size)
        self._locked = False
    
    def __buffer__(self, flags: int) -> memoryview:
        """Return a memoryview."""
        self._locked = True
        return memoryview(self._buffer)
    
    def __release_buffer__(self, mv: memoryview) -> None:
        """Called when the memoryview is deleted."""
        self._locked = False


mb = ManagedBuffer(10)
mv = memoryview(mb)
print(mb._locked)  # True
del mv
print(mb._locked)  # False
```

## Buffer Protocol Reference

For more details on the Python buffer protocol, see the [official documentation](https://docs.python.org/3/reference/datamodel.html#python-buffer-protocol).

## Related Protocols

- **[Containers](containers.md)**: For container protocols like `__len__` and `__getitem__`
