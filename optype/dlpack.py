"""
Types and interfaces for DLPack, as used the Array API.
https://github.com/dmlc/dlpack
"""

from __future__ import annotations

import enum
import sys
from typing import Any, Protocol


if sys.version_info >= (3, 13):
    from types import CapsuleType
    from typing import TypeVar, runtime_checkable
else:
    from typing_extensions import CapsuleType, TypeVar, runtime_checkable


__all__ = "CanDLPack", "CanDLPackDevice"


def __dir__() -> tuple[str, ...]:
    return __all__


class DLDeviceType(enum.IntEnum):
    # GPU device
    CPU = 1
    # Cuda GPU device
    CUDA = 2
    # Pinned CUDA GPU memory vy `cudaMallocHost`
    CPU_PINNED = 3
    # OpenCL devices
    OPENCL = 4
    # Vulkan buffer for next generation graphics
    VULKAN = 7
    # Metal for Apple GPU
    METAL = 8
    # Verilog simulation buffer
    VPI = 9
    # ROCm GPU's for AMD GPU's
    ROCM = 10
    # CUDA managed/unified memory allocated by `cudaMallocManaged`
    CUDA_MANAGED = 13
    # Unified shared memory allocated on a oneAPI non-partititioned
    # device. Call to oneAPI runtime is required to determine the device
    # type, the USM allocation type and the sycl context it is bound to.
    ONE_API = 14


class DLDataTypeCode(enum.IntEnum):
    # signed integer
    INT = 0
    # unsigned integer
    UINT = 1
    # IEEE floating point
    FLOAT = 2
    # Opaque handle type, reserved for testing purposes.
    OPAQUE_HANDLE = 3
    # bfloat16
    BFLOAT = 4
    # complex number (C/C++/Python layout: compact struct per complex number)
    COMPLEX = 5
    # boolean
    BOOL = 6


_TypeT_co = TypeVar(
    "_TypeT_co",
    covariant=True,
    bound=enum.Enum | int,
    default=enum.Enum | int,
)
_DeviceT_co = TypeVar("_DeviceT_co", covariant=True, bound=int, default=int)


# NOTE: Because `__dlpack__` doesn't mutate the type, and the type parameters bind to
# the *co*variant `tuple`, they should be *co*variant; NOT *contra*variant!
# NOTE NOTE: This shows that PEP 695 claim, i.e. that variance can always be inferred,
# is utter nonsense.


@runtime_checkable
class CanDLPack(Protocol[_TypeT_co, _DeviceT_co]):  # type: ignore[misc] # pyright: ignore[reportInvalidTypeVarUse]
    def __dlpack__(  # type: ignore[no-any-explicit]
        self,
        /,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[_TypeT_co, _DeviceT_co] | None = None,
        # NOTE: This should be `bool | None`, but because of an incorrect annotation in
        # `numpy.ndarray.__dlpack__`, this is not possible at the moment.
        copy: Any | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> CapsuleType: ...


@runtime_checkable
class CanDLPackDevice(Protocol[_TypeT_co, _DeviceT_co]):
    def __dlpack_device__(self, /) -> tuple[_TypeT_co, _DeviceT_co]: ...
