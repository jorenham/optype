# ruff: noqa: PYI042
from typing import Literal, TypeAlias

import numpy as np

import optype.numpy as onp


_0D: TypeAlias = tuple[()]
_1D: TypeAlias = tuple[Literal[1]]
_2D: TypeAlias = tuple[Literal[1], Literal[1]]


def test_can_array():
    scalar: onp.CanArray[_0D, np.uint8] = np.uint8(42)
    assert isinstance(scalar, onp.CanArray)

    arr_0d: onp.CanArray[_0D, np.uint8] = np.array(42, dtype=np.uint8)
    assert isinstance(arr_0d, onp.CanArray)

    arr_1d: onp.CanArray[_1D, np.uint8] = np.array([42], dtype=np.uint8)
    assert isinstance(arr_1d, onp.CanArray)

    arr_2d: onp.CanArray[_2D, np.uint8] = np.array([[42]], dtype=np.uint8)
    assert isinstance(arr_2d, onp.CanArray)
