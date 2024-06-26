# ruff: noqa: PYI042
from typing import Any, TypeAlias, TypeVar

import numpy as np
import pytest

import optype.numpy as onp


_Shape0D: TypeAlias = tuple[()]
_Shape1D: TypeAlias = tuple[int]
_Shape2D: TypeAlias = tuple[int, int]


# Don't wake up, Neo...
@pytest.mark.filterwarnings('ignore:the matrix .*:PendingDeprecationWarning')
def test_can_array():
    sct: type[np.generic] = np.uint8

    scalar: onp.CanArray[_Shape0D, sct] = sct(42)
    assert isinstance(scalar, onp.CanArray)
    assert not isinstance(42, onp.CanArray)

    arr_0d: onp.CanArray[_Shape0D, sct] = np.array(42, sct)
    assert isinstance(arr_0d, onp.CanArray)

    arr_1d: onp.CanArray[_Shape1D, sct] = np.array([42], sct)
    assert isinstance(arr_1d, onp.CanArray)
    assert not isinstance([42], onp.CanArray)

    arr_2d: onp.CanArray[_Shape2D, sct] = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.CanArray)

    mat: onp.CanArray[_Shape2D, sct] = np.asmatrix(42, sct)
    assert isinstance(mat, onp.CanArray)


def test_some_array():
    sct: type[np.generic] = np.uint8

    py_sc: onp.AnyArray[_Shape0D, sct, int] = 42
    np_sc: onp.AnyArray[_Shape0D, sct, int] = sct(py_sc)
    assert np_sc.shape == ()

    np_0d: onp.AnyArray[_Shape0D, sct, int] = np.array(py_sc, sct)
    assert np_0d.shape == ()

    py_1d: onp.AnyArray[_Shape1D, sct, int] = [42]
    np_1d: onp.AnyArray[_Shape1D, sct, int] = np.array(py_1d, sct)
    assert np_1d.shape == (1,)

    py_2d: onp.AnyArray[_Shape2D, sct, int] = [[42]]
    np_2d: onp.AnyArray[_Shape2D, sct, int] = np.array(py_2d, sct)
    assert np_2d.shape == (1, 1)


_T = TypeVar('_T', bound=np.generic)
_Arr0D: TypeAlias = onp.Array[tuple[()], _T]
_Arr1D: TypeAlias = onp.Array[tuple[int], _T]
_Arr2D: TypeAlias = onp.Array[tuple[int, int], _T]


def test_can_array_function():
    sct: type[np.generic] = np.uint8

    assert not isinstance(sct(42), onp.CanArrayFunction)

    arr_0d: onp.CanArrayFunction[Any, _Arr0D[sct]] = np.array(42, sct)
    assert isinstance(arr_0d, onp.CanArrayFunction)

    arr_1d: onp.CanArrayFunction[Any, _Arr1D[sct]] = np.array([42], sct)
    assert isinstance(arr_1d, onp.CanArrayFunction)

    arr_2d: onp.CanArrayFunction[Any, _Arr2D[sct]] = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.CanArrayFunction)


def test_can_array_finalize():
    sct: type[np.generic] = np.uint8

    arr_0d: onp.CanArrayFinalize[_Arr0D[sct]] = np.array(42, sct)
    assert isinstance(arr_0d, onp.CanArrayFinalize)
    assert not isinstance(42, onp.CanArrayFinalize)

    arr_1d: onp.CanArrayFinalize[_Arr1D[sct]] = np.array([42], sct)
    assert isinstance(arr_1d, onp.CanArrayFinalize)
    assert not isinstance([42], onp.CanArrayFinalize)

    arr_2d: onp.CanArrayFinalize[_Arr2D[sct]] = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.CanArrayFinalize)


def test_can_array_wrap():
    sct: type[np.generic] = np.uint8

    arr_0d: onp.CanArrayWrap = np.array(42, sct)
    assert isinstance(arr_0d, onp.CanArrayWrap)
    assert not isinstance(42, onp.CanArrayWrap)

    arr_1d: onp.CanArrayWrap = np.array([42], sct)
    assert isinstance(arr_1d, onp.CanArrayWrap)
    assert not isinstance([42], onp.CanArrayWrap)

    arr_2d: onp.CanArrayWrap = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.CanArrayWrap)


def test_has_array_priority():
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


def test_has_array_interface():
    sct: type[np.generic] = np.uint8

    scalar: onp.HasArrayInterface[Any] = sct(42)
    assert isinstance(scalar, onp.HasArrayInterface)

    arr_0d: onp.HasArrayInterface[Any] = np.array(42, sct)
    assert isinstance(arr_0d, onp.HasArrayInterface)
    assert not isinstance(42, onp.HasArrayInterface)

    arr_1d: onp.HasArrayInterface[Any] = np.array([42], sct)
    assert isinstance(arr_1d, onp.HasArrayInterface)
    assert not isinstance([42], onp.HasArrayInterface)

    arr_2d: onp.HasArrayInterface[Any] = np.array([[42]], sct)
    assert isinstance(arr_2d, onp.HasArrayInterface)
