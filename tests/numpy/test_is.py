# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownMemberType=false
import operator
from collections.abc import Callable

import numpy as np
import pytest

import optype.numpy as onp

if np.__version__ >= "2":
    CHARS = "?BbHhIiLlQqNnPpefdgFDGSUVOMm"
else:
    CHARS = "?BbHhIiLlQqPpefdgFDGSUVOMm"  # pyright: ignore[reportConstantRedefinition]

DTYPES = [np.dtype(char) for char in CHARS]
SCTYPES = {dtype.type for dtype in DTYPES if issubclass(dtype.type, np.generic)}
NDARRAY_TYPES = np.ndarray, np.ma.MaskedArray


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_is_dtype(dtype: onp.DType) -> None:
    assert onp.is_dtype(dtype)
    assert not onp.is_dtype(type(dtype))
    assert not onp.is_dtype(dtype.type)
    assert not onp.is_dtype(np.empty((), dtype))
    assert not onp.is_dtype(np.empty((), dtype).item())


@pytest.mark.parametrize("sctype", SCTYPES, ids=operator.attrgetter("__name__"))
def test_is_sctype(sctype: type[np.generic]) -> None:
    assert onp.is_sctype(sctype)
    assert not onp.is_sctype(np.empty((), sctype))
    assert not onp.is_sctype(np.empty((), sctype).item())
    assert not onp.is_sctype(np.dtype(sctype))


@pytest.mark.parametrize(
    "dtype_map",
    [
        np.dtype,
        operator.attrgetter("type"),
        lambda dtype: dtype.type.mro()[1],
        lambda dtype: dtype.type.mro()[-2],
    ],
    ids=["_", "_.type", "_.type.mro()[1]", "_.type.mro()[-2]"],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("ndtype", NDARRAY_TYPES)
def test_is_array(
    ndtype: type[onp.Array],
    dtype: onp.DType,
    dtype_map: Callable[[onp.DType], onp.DType | type[np.generic]],
) -> None:
    arr = [np.empty(tuple(range(ndim)), dtype).view(ndtype) for ndim in range(5)]

    assert onp.is_array_nd(arr[0])
    assert onp.is_array_0d(arr[0])
    assert not onp.is_array_1d(arr[0])
    assert not onp.is_array_2d(arr[0])
    assert not onp.is_array_3d(arr[0])

    assert onp.is_array_nd(arr[1])
    assert not onp.is_array_0d(arr[1])
    assert onp.is_array_1d(arr[1])
    assert not onp.is_array_2d(arr[1])
    assert not onp.is_array_3d(arr[1])

    assert onp.is_array_nd(arr[2])
    assert not onp.is_array_0d(arr[2])
    assert not onp.is_array_1d(arr[2])
    assert onp.is_array_2d(arr[2])
    assert not onp.is_array_3d(arr[2])

    assert onp.is_array_nd(arr[2].view(np.matrix))
    assert not onp.is_array_0d(arr[2].view(np.matrix))
    assert not onp.is_array_1d(arr[2].view(np.matrix))
    assert onp.is_array_2d(arr[2].view(np.matrix))
    assert not onp.is_array_3d(arr[2].view(np.matrix))

    assert onp.is_array_nd(arr[3])
    assert not onp.is_array_0d(arr[3])
    assert not onp.is_array_1d(arr[3])
    assert not onp.is_array_2d(arr[3])
    assert onp.is_array_3d(arr[3])

    assert onp.is_array_nd(arr[4])
    assert not onp.is_array_0d(arr[4])
    assert not onp.is_array_1d(arr[4])
    assert not onp.is_array_2d(arr[4])
    assert not onp.is_array_3d(arr[4])

    dtype_is = dtype_map(dtype)
    assert onp.is_array_0d(arr[0], dtype=dtype_is)
    assert onp.is_array_1d(arr[1], dtype=dtype_is)
    assert onp.is_array_2d(arr[2], dtype=dtype_is)
    assert onp.is_array_3d(arr[3], dtype=dtype_is)
    assert onp.is_array_nd(arr[4], dtype=dtype_is)

    dtype_not = np.dtype("?") if dtype.char != "?" else np.dtype("B")
    # pyrefly: ignore[bad-argument-type]
    assert not onp.is_array_0d(arr[0], dtype=dtype_not)
    # pyrefly: ignore[bad-argument-type]
    assert not onp.is_array_1d(arr[1], dtype=dtype_not)
    # pyrefly: ignore[bad-argument-type]
    assert not onp.is_array_2d(arr[2], dtype=dtype_not)
    # pyrefly: ignore[bad-argument-type]
    assert not onp.is_array_3d(arr[3], dtype=dtype_not)
    # pyrefly: ignore[bad-argument-type]
    assert not onp.is_array_nd(arr[4], dtype=dtype_not)
