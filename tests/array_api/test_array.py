import numpy as np

from optype.array_api._array import Array, BaseArray


def test_numpy_ndarray() -> None:
    a_np = np.empty((3, 2), dtype=np.dtypes.Float64DType())

    a_xp0: BaseArray = a_np
    assert isinstance(a_np, BaseArray)

    a_xp: Array = a_np
    assert isinstance(a_np, Array)
