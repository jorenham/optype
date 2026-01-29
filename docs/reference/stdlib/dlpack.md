# optype.dlpack

Protocols for the DLPack array interchange protocol.

## Overview

The [DLPack](https://dmlc.github.io/dlpack/latest/) protocol enables zero-copy data exchange between different array libraries (NumPy, PyTorch, TensorFlow, JAX, CuPy, etc.). `optype.dlpack` provides type-safe protocols for this interchange format.

## Protocols

### CanDLPack

Implements the `__dlpack__` method:

```python
class CanDLPack[+T = int, +D: int = int]:
    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[T, D] | None = None,
        copy: bool | None = None,
    ) -> types.CapsuleType: ...
```

**Purpose**: Export array data as a DLPack capsule

### CanDLPackDevice

Implements the `__dlpack_device__` method:

```python
class CanDLPackDevice[+T = int, +D: int = int]:
    def __dlpack_device__(self) -> tuple[T, D]: ...
```

**Purpose**: Report the device type and device ID of the array

## Enums

### DLDeviceType

Device type codes:

```python
class DLDeviceType(IntEnum):
    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10
    CUDA_HOST = 11
    CUDA_MANAGED = 13
    ONE_API = 14
```

### DLDataTypeCode

Data type codes for DLPack arrays:

```python
class DLDataTypeCode(IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BFLOAT = 4
    COMPLEX = 5
```

## Usage Examples

### Implementing DLPack Protocol

```python
import numpy as np
from optype.dlpack import CanDLPack, CanDLPackDevice, DLDeviceType

class MyArray(CanDLPack[DLDeviceType, int], CanDLPackDevice[DLDeviceType, int]):
    def __init__(self, data: np.ndarray):
        self._array = data
    
    def __dlpack__(self, *, stream=None, **kwargs):
        """Export as DLPack capsule."""
        return self._array.__dlpack__(stream=stream, **kwargs)
    
    def __dlpack_device__(self) -> tuple[DLDeviceType, int]:
        """Report device (CPU, device 0)."""
        return (DLDeviceType.CPU, 0)

# Create array
arr = MyArray(np.array([1, 2, 3]))

# Can be consumed by other libraries
import torch
tensor = torch.from_dlpack(arr)  # Zero-copy!
```

### Cross-Library Array Sharing

```python
import numpy as np
import torch
from optype.dlpack import CanDLPack

def share_array(arr: CanDLPack) -> torch.Tensor:
    """Convert any DLPack array to PyTorch tensor."""
    return torch.from_dlpack(arr)

# NumPy array
np_arr = np.array([[1, 2], [3, 4]])
torch_tensor = share_array(np_arr)

# Works with any DLPack-compatible array
```

### Device Type Checking

```python
from optype.dlpack import CanDLPackDevice, DLDeviceType

def is_on_gpu(arr: CanDLPackDevice) -> bool:
    """Check if array is on GPU."""
    device_type, device_id = arr.__dlpack_device__()
    return device_type in (DLDeviceType.CUDA, DLDeviceType.ROCM, DLDeviceType.METAL)

# Usage with PyTorch
import torch
cpu_tensor = torch.randn(10)
gpu_tensor = torch.randn(10, device='cuda' if torch.cuda.is_available() else 'cpu')

print(is_on_gpu(cpu_tensor))  # False
print(is_on_gpu(gpu_tensor))  # True (if CUDA available)
```

## Type Parameters

- **`T` (Device Type)**: Covariant - device type (typically `int` or `DLDeviceType`)
- **`D` (Device ID)**: Covariant - device identifier (integer)

## Important Notes

### Zero-Copy Semantics

DLPack enables zero-copy data sharing:

- No data duplication
- Shared memory between libraries
- Modifications affect both arrays

```python
import numpy as np
import torch

# Zero-copy sharing
np_arr = np.array([1, 2, 3])
torch_tensor = torch.from_dlpack(np_arr)

# Modifying one affects the other!
torch_tensor[0] = 999
print(np_arr)  # [999, 2, 3]
```

### Stream Synchronization

For async GPU operations, specify the stream:

```python
# With CUDA stream
dlpack_capsule = gpu_array.__dlpack__(stream=cuda_stream_handle)
```

## References

- [DLPack Specification](https://dmlc.github.io/dlpack/latest/)
- [Python Array API Standard](https://data-apis.org/array-api/latest/)
- [NumPy DLPack Support](https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html)

## Related Types

- **[NumPy Low-level](../numpy/low-level.md)**: NumPy array protocols
- **[NumPy Aliases](../numpy/aliases.md)**: NumPy array types
