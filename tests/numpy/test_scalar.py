# mypy: disable-error-code="unreachable"
import sys
from typing import Literal

import numpy as np
import pytest

if sys.version_info >= (3, 13):
    from typing import assert_type
else:
    from typing_extensions import assert_type

from optype.numpy import Scalar, _scalar as _sc  # pyright: ignore[reportPrivateUsage]

NP2 = np.__version__.startswith("2.")


def test_from_bool() -> None:
    x_py = True
    x_np = np.True_

    s_py: Scalar[bool] = x_py  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert not isinstance(x_py, Scalar)

    s_np_n: Scalar[bool] = x_np
    s_np_1: Scalar[bool, Literal[1]] = x_np

    s_np_wrong_n: Scalar[str] = x_np  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_np_wrong_1: Scalar[str, Literal[1]] = x_np  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    assert isinstance(x_np, Scalar)

    x_item: bool = x_np.item()
    s_item: bool = s_np_n.item()


@pytest.mark.parametrize(
    "sctype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64],
)
def test_from_integer(sctype: type[_sc.integer]) -> None:
    x_py = 42
    x_np = sctype(x_py)

    s_py: Scalar[int] = x_py  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert not isinstance(x_py, Scalar)

    s_np: Scalar[int] = x_np
    s_np_wrong: Scalar[str] = x_np  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_np_wrong_contra: Scalar[bool] = x_np  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(x_np, Scalar)

    y_py = s_np.item()
    assert_type(y_py, int)
    assert_type(y_py, bool)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]


@pytest.mark.parametrize("sctype", [np.float16, np.float32, np.float64])
def test_from_floating(sctype: type[_sc.floating]) -> None:
    x_py = -1 / 12
    x_np = sctype(x_py)

    s_py: Scalar[float] = x_py  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert not isinstance(x_py, Scalar)

    s_np: Scalar[float] = x_np
    s_np_wrong: Scalar[str] = x_np  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_np_wrong_contra: Scalar[int] = x_np  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(x_np, Scalar)

    y_py = s_np.item()
    assert_type(y_py, float)
    assert_type(y_py, int)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]


@pytest.mark.parametrize("sctype", [np.complex64, np.complex128])
def test_from_complex(sctype: type[_sc.cfloating]) -> None:
    x_py = 3 - 4j
    x_np = sctype(x_py)

    s_py: Scalar[complex] = x_py  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert not isinstance(x_py, Scalar)

    s_np: Scalar[complex] = x_np
    s_np_wrong: Scalar[str] = x_np  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_np_wrong_contra: Scalar[float] = x_np  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert isinstance(x_np, Scalar)

    y_py = s_np.item()
    assert_type(y_py, complex)
    assert_type(y_py, float)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
