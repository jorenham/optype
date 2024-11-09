try:
    from array_api_compat.common._helpers import (  # pyright: ignore[reportMissingTypeStubs]
        _dask_device as _DeviceDask,  # pyright: ignore[reportUnknownVariableType]  # noqa: N812
    )
except ImportError:
    from typing import NoReturn as _DeviceDask


__all__ = ["_DeviceDask"]
