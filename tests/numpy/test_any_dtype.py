import types
from typing import Final, Never

import numpy as np
import pytest

from optype.numpy import _dtype_attr

_DTYPES: Final = (
    np.dtype(np.bool_),
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.int64),
    np.dtype(np.intc),
    np.dtype(np.intp),
    np.dtype(np.int_),
    np.dtype(np.longlong),
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.uint64),
    np.dtype(np.uintc),
    np.dtype(np.uintp),
    np.dtype(np.uint),
    np.dtype(np.ulonglong),
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
    np.dtype(np.longdouble),
    np.dtype(np.complex64),
    np.dtype(np.complex128),
    np.dtype(np.clongdouble),
    np.dtype(np.object_),
    np.dtype(np.bytes_),
    np.dtype("c"),
    np.dtype(np.str_),
    np.dtype(np.void),
    np.dtype(np.datetime64),
    np.dtype(np.timedelta64),
    # `StringDType` is too broken too test at the moment
)
_TIME_UNITS: Final = "as", "fs", "ps", "ns", "us", "s", "m", "h", "D", "W", "M", "Y"


def _get_dtype_codes(
    dtype: np.dtype[np.generic],
    module: types.ModuleType = _dtype_attr,
) -> tuple[frozenset[str], frozenset[str]]:
    try:
        strcode = dtype.str[1:]
        literal_name = getattr(module, f"{strcode}_name")
        literal_char = getattr(module, f"{strcode}_char")
    except AttributeError:
        literal_name = getattr(module, f"{dtype.char}_name")
        literal_char = getattr(module, f"{dtype.char}_char")

    names = frozenset(() if literal_name is Never else literal_name.__args__)
    chars = frozenset(() if literal_char is Never else literal_char.__args__)
    return names, chars


def _normalized_dtype_name(dtype: np.dtype[np.generic]) -> str:
    return dtype.name.split("[")[0]  # avoid e.g. "datetime64[ns]"


@pytest.mark.parametrize(
    ("dtype", "names", "chars"),
    [(dtype, *_get_dtype_codes(dtype)) for dtype in _DTYPES],
)
def test_dtype_has_codes(
    dtype: np.dtype[np.generic],
    names: frozenset[str],
    chars: frozenset[str],
) -> None:
    name = dtype.name

    assert name in names, (name, names)
    assert dtype.str in chars, (dtype.str, chars)
    assert dtype.str[1:] in chars, (dtype.str, chars)

    codes = names | chars
    out_names: set[str] = set()
    for code in codes:
        try:
            dtype_ = np.dtype(code)
        except TypeError:
            continue

        out_names.add(_normalized_dtype_name(dtype_))

    assert len(out_names) == 1


@pytest.mark.parametrize(
    ("dtype", "names", "chars"),
    [
        (dtype, *_get_dtype_codes(dtype))
        for dtype in [np.dtype(np.datetime64), np.dtype(np.timedelta64)]
    ],
)
@pytest.mark.parametrize("unit", _TIME_UNITS)
def test_time_units(
    unit: str,
    dtype: np.dtype[np.datetime64] | np.dtype[np.timedelta64],
    names: frozenset[str],
    chars: frozenset[str],
) -> None:
    name = f"{dtype.name}[{unit}]"
    str_ = f"{dtype.str}[{unit}]"

    assert name in names, (name, names)
    assert str_ in chars, (str_, chars)
