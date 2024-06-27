# ruff: noqa: F841, PYI042

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np


if TYPE_CHECKING:
    import optype.numpy as onp


_0D: TypeAlias = tuple[()]
_1D: TypeAlias = tuple[Literal[1]]
_2D: TypeAlias = tuple[Literal[1], Literal[1]]


def test_any_array():
    type_np: type[np.integer[Any]] = np.int16
    v = 42

    # scalar

    v_py: onp.AnyArray[_0D] = v
    assert np.shape(v_py) == ()

    v_np: onp.AnyArray[_0D] = type_np(v_py)
    assert int(v_np) == v_py
    assert np.shape(v_np) == ()

    # 0d

    x0_np: onp.AnyArray[_0D] = np.array(v_py)
    assert np.shape(x0_np) == ()

    # 1d

    x1_py: onp.AnyArray[_1D] = [v_py]
    assert np.shape(x1_py) == (1,)

    x1_py_np: onp.AnyArray[_1D] = [x0_np]
    assert np.shape(x1_py_np) == (1,)

    x1_np: onp.AnyArray[_1D] = np.array(x1_py)
    assert np.shape(x1_np) == (1,)

    # 2d

    x2_py: onp.AnyArray[_2D] = [x1_py]
    assert np.shape(x2_py) == (1, 1)

    x2_py_np: onp.AnyArray[_2D] = [x1_np]
    assert np.shape(x2_py_np) == (1, 1)

    x2_np: onp.AnyArray[_2D] = np.array(x2_py)
    assert np.shape(x2_np) == (1, 1)
