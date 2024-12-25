from collections.abc import Callable
from typing import TypeAlias, TypeVar, cast

import numpy as np
import pytest

import optype.numpy as onp


_Shape0D: TypeAlias = tuple[()]
_Shape1D: TypeAlias = tuple[int]
_Shape2D: TypeAlias = tuple[int, int]

_AnyCallable: TypeAlias = Callable[[], object]


# Don't wake up, Neo...
@pytest.mark.filterwarnings("ignore:the matrix .*:PendingDeprecationWarning")
def test_can_array() -> None:
    dt = np.dtypes.Int8DType()

    val_x = dt.type(42)
    val_y: onp.CanArray[_Shape0D, np.dtype[np.int8]] = val_x
    assert isinstance(val_x, onp.CanArray)
    assert not isinstance(val_x.tolist(), onp.CanArray)

    nd0_x = np.empty((), dt)
    nd0_y: onp.CanArray[_Shape0D, np.dtype[np.int8]] = nd0_x
    assert isinstance(nd0_x, onp.CanArray)

    nd1_x = cast("np.ndarray[_Shape1D, np.dtypes.Int8DType]", np.empty((42,), dt))
    nd1_y: onp.CanArray[_Shape1D, np.dtype[np.int8]] = nd1_x
    assert isinstance(nd1_x, onp.CanArray)
    assert not isinstance(nd1_x.tolist(), onp.CanArray)

    nd2_x = cast(
        "np.ndarray[_Shape2D, np.dtypes.Int8DType]",
        cast("object", np.empty((6, 7), dt)),
    )
    nd2_y: onp.CanArray[_Shape2D, np.dtype[np.int8]] = nd2_x
    assert isinstance(nd2_x, onp.CanArray)
    assert not isinstance(nd2_x.tolist(), onp.CanArray)

    mat_x = np.asmatrix([[1, 0], [0, 1]], dt)
    mat_y: onp.CanArray[_Shape2D, np.dtypes.Int8DType] = mat_x
    assert isinstance(mat_x, onp.CanArray)


_T = TypeVar("_T", bound=np.generic)
_Arr0D: TypeAlias = onp.Array[tuple[()], _T]
_Arr1D: TypeAlias = onp.Array[tuple[int], _T]
_Arr2D: TypeAlias = onp.Array[tuple[int, int], _T]


def test_can_array_function() -> None:
    sct: type[np.generic] = np.uint8

    assert not isinstance(sct(42), onp.CanArrayFunction)

    arr_0d: onp.CanArrayFunction[_AnyCallable, _Arr0D[np.uint8]] = np.array(42, sct)
    assert isinstance(arr_0d, onp.CanArrayFunction)

    arr_1d: onp.CanArrayFunction[_AnyCallable, _Arr1D[np.uint8]] = np.array([42], sct)
    assert isinstance(arr_1d, onp.CanArrayFunction)

    arr_2d: onp.CanArrayFunction[_AnyCallable, _Arr2D[np.uint8]] = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.CanArrayFunction)


def test_can_array_finalize() -> None:
    sct: type[np.generic] = np.uint8

    arr_0d: onp.CanArrayFinalize[_Arr0D[np.uint8]] = np.array(42, sct)
    assert isinstance(arr_0d, onp.CanArrayFinalize)
    assert not isinstance(42, onp.CanArrayFinalize)

    arr_1d: onp.CanArrayFinalize[_Arr1D[np.uint8]] = np.array([42], sct)
    assert isinstance(arr_1d, onp.CanArrayFinalize)
    assert not isinstance([42], onp.CanArrayFinalize)

    arr_2d: onp.CanArrayFinalize[_Arr2D[np.uint8]] = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.CanArrayFinalize)


def test_can_array_wrap() -> None:
    sct: type[np.generic] = np.uint8

    arr_0d: onp.CanArrayWrap = np.array(42, sct)
    assert isinstance(arr_0d, onp.CanArrayWrap)
    assert not isinstance(42, onp.CanArrayWrap)

    arr_1d: onp.CanArrayWrap = np.array([42], sct)
    assert isinstance(arr_1d, onp.CanArrayWrap)
    assert not isinstance([42], onp.CanArrayWrap)

    arr_2d: onp.CanArrayWrap = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.CanArrayWrap)


def test_has_array_priority() -> None:
    sct: type[np.generic] = np.uint8

    scalar: onp.HasArrayPriority = sct(42)
    assert isinstance(scalar, onp.HasArrayPriority)

    arr_0d: onp.HasArrayPriority = np.array(42, sct)
    assert isinstance(arr_0d, onp.HasArrayPriority)
    assert not isinstance(42, onp.HasArrayPriority)

    arr_1d: onp.HasArrayPriority = np.array([42], sct)
    assert isinstance(arr_1d, onp.HasArrayPriority)
    assert not isinstance([42], onp.HasArrayPriority)

    arr_2d: onp.HasArrayPriority = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.HasArrayPriority)


def test_has_array_interface() -> None:
    sct: type[np.generic] = np.uint8

    scalar: onp.HasArrayInterface = sct(42)
    assert isinstance(scalar, onp.HasArrayInterface)

    arr_0d: onp.HasArrayInterface = np.array(42, sct)
    assert isinstance(arr_0d, onp.HasArrayInterface)
    assert not isinstance(42, onp.HasArrayInterface)

    arr_1d: onp.HasArrayInterface = np.array([42], sct)
    assert isinstance(arr_1d, onp.HasArrayInterface)
    assert not isinstance([42], onp.HasArrayInterface)

    arr_2d: onp.HasArrayInterface = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.HasArrayInterface)
