import sys
from typing import Any, Literal

import numpy as np
import pytest

from optype.numpy import Scalar


if sys.version_info >= (3, 13):
    from typing import assert_type
else:
    from typing_extensions import assert_type


_NP_V2 = np.__version__.startswith('2.')


def test_scalar_from_bool():
    x_py = True
    x_np = np.True_

    s_py: Scalar[bool] = x_py  # pyright: ignore[reportAssignmentType]
    assert not isinstance(x_py, Scalar)

    s_np_any_n: Scalar[Any] = x_np
    s_np_any_1: Scalar[Any, Literal[1]] = x_np

    s_np_n: Scalar[bool] = x_np
    s_np_1: Scalar[bool, Literal[1]] = x_np

    s_np_wrong_n: Scalar[str] = x_np  # pyright: ignore[reportAssignmentType]
    s_np_wrong_1: Scalar[str, Literal[1]] = x_np  # pyright: ignore[reportAssignmentType]

    assert isinstance(x_np, Scalar)

    assert_type(x_np.item(), bool)
    assert_type(s_np_n.item(), bool)


@pytest.mark.parametrize(
    'sctype',
    [
        np.int8, np.uint8,
        np.int16, np.uint16,
        np.int32, np.uint32,
        np.int64, np.uint64,
    ],
)
def test_scalar_from_integer(sctype: type[np.integer[Any]]):
    x_py = 42
    x_np = sctype(x_py)

    s_py: Scalar[int] = x_py  # pyright: ignore[reportAssignmentType]
    assert not isinstance(x_py, Scalar)

    s_np: Scalar[int] = x_np
    s_np_wrong: Scalar[str] = x_np  # pyright: ignore[reportAssignmentType]
    s_np_wrong_contra: Scalar[bool] = x_np  # pyright: ignore[reportAssignmentType]
    assert isinstance(x_np, Scalar)

    y_py = s_np.item()
    assert_type(y_py, int)
    assert_type(y_py, bool)  # pyright: ignore[reportAssertTypeFailure]


@pytest.mark.parametrize('sctype', [np.float16, np.float32, np.float64])
def test_scalar_from_floating(sctype: type[np.floating[Any]]):
    x_py = -1 / 12
    x_np = sctype(x_py)

    s_py: Scalar[float] = x_py  # pyright: ignore[reportAssignmentType]
    assert not isinstance(x_py, Scalar)

    s_np: Scalar[float] = x_np
    s_np_wrong: Scalar[str] = x_np  # pyright: ignore[reportAssignmentType]
    s_np_wrong_contra: Scalar[int] = x_np  # pyright: ignore[reportAssignmentType]
    assert isinstance(x_np, Scalar)

    y_py = s_np.item()
    assert_type(y_py, float)
    assert_type(y_py, int)  # pyright: ignore[reportAssertTypeFailure]


@pytest.mark.parametrize('sctype', [np.complex64, np.complex128])
def test_scalar_from_complex(sctype: type[np.complexfloating[Any, Any]]):
    x_py = 3 - 4j
    x_np = sctype(x_py)

    s_py: Scalar[complex] = x_py  # pyright: ignore[reportAssignmentType]
    assert not isinstance(x_py, Scalar)

    s_np: Scalar[complex] = x_np
    s_np_wrong: Scalar[str] = x_np  # pyright: ignore[reportAssignmentType]
    s_np_wrong_contra: Scalar[float] = x_np  # pyright: ignore[reportAssignmentType]
    assert isinstance(x_np, Scalar)

    y_py = s_np.item()
    assert_type(y_py, complex)
    assert_type(y_py, float)  # pyright: ignore[reportAssertTypeFailure]
