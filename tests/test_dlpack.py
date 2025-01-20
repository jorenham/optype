import numpy as np

from optype.dlpack import CanDLPackCompat, CanDLPackDevice


def test_ndarray_can_dlpack() -> None:
    x: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0)
    # NOTE: mypy doesn't understand covariance...
    x_dl: CanDLPackCompat = x

    assert isinstance(x, CanDLPackCompat)


def test_ndarray_can_dlpack_device() -> None:
    x: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0)
    x_dl: CanDLPackDevice = x

    assert isinstance(x, CanDLPackDevice)
