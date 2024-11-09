# mypy: disable-error-code="import-untyped"
# pyright: reportPrivateUsage=false, reportMissingTypeStubs=false
# ruff: noqa: N812

try:
    from array_api_compat.common._helpers import _dask_device as _DeviceDask
except ImportError:
    from typing import NoReturn as _DeviceDask


__all__ = ["_DeviceDask"]
