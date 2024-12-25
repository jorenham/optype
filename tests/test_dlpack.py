import numpy as np

from optype.dlpack import CanDLPack, CanDLPackDevice


def test_ndarray_can_dlpack():
    x: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0)
    # NOTE: mypy doesn't understand covariance...
    x_dl: CanDLPack = x

    assert isinstance(x, CanDLPack)


def test_ndarray_can_dlpack_device():
    x: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0)
    x_dl: CanDLPackDevice = x

    assert isinstance(x, CanDLPack)
