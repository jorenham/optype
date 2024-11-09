import sys
from typing import Literal, TypeAlias

if sys.version_info >= (3, 13):
    from typing import override
else:
    from typing_extensions import override

__all__ = ["_DeviceDask"]

class _dask_device:  # noqa: N801
    @override
    def __repr__(self, /) -> Literal["DASK_DEVICE"]: ...

_DeviceDask: TypeAlias = _dask_device
