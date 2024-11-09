from typing import Literal, Protocol, TypeAlias, runtime_checkable

from ._device_dask import _DeviceDask


__all__ = ["Device"]


_DeviceNP: TypeAlias = Literal["cpu"]


@runtime_checkable
class _DeviceXPS(Protocol):
    # https://github.com/data-apis/array-api-strict/blob/2.1/array_api_strict/_array_object.py#L46-L61
    _device: Literal["CPU_DEVICE", "device1", "device2"]


@runtime_checkable
class _DeviceTorch(Protocol):
    # https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    # https://github.com/pytorch/pytorch/blob/v2.5.0/torch/_C/__init__.pyi.in#L106-L124
    @property
    def index(self, /) -> int: ...
    @property
    def type(self, /) -> str: ...


@runtime_checkable
class _HasID(Protocol):
    # the common denominator of the cupy and jax device type
    #
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Device.html
    # https://github.com/cupy/cupy/blob/v13.3.0/cupy/cuda/device.pyx#L120
    #
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.Device.html
    # https://github.com/openxla/xla/blob/4d0fe88/xla/python/xla_extension/__init__.pyi#L442-L460
    @property
    def id(self, /) -> int: ...


Device: TypeAlias = _DeviceNP | _HasID | _DeviceXPS | _DeviceTorch | _DeviceDask
