# ruff: noqa: I001, PLC2701
from typing import Any

import numpy as np
import pytest

from optype.numpy import _any_scalar, _any_dtype  # pyright: ignore[reportPrivateUsage]
from optype.inspect import get_args


_TEMPORAL = 'TimeDelta64', 'DateTime64'
_FLEXIBLE = 'Str', 'Bytes', 'Void'
_SIMPLE = 'Bool', 'Object'
_NUMERIC_N = (
    'UInt8', 'UInt16', 'UInt32', 'UInt64',
    'Int8', 'Int16', 'Int32', 'Int64',
    'Float16', 'Float32', 'Float64',
    'Complex64', 'Complex128',
)
_NUMERIC_C = (
    'UByte', 'UShort', 'UIntC', 'ULong',
    'Byte', 'Short', 'IntC', 'Long',
    'Half', 'Single', 'Double',
    'CSingle', 'CDouble',
)


def _get_attr_args(obj: Any, name: str) -> tuple[Any, ...]:
    tp = getattr(obj, name)
    return get_args(tp) or (tp, )


def _get_dtype_info(name: str) -> tuple[
    frozenset[type],
    frozenset[str],
    frozenset[str],
]:
    types = _get_attr_args(_any_scalar, f'Any{name}')
    # workaround for `np.dtype(datetime.{datetime,timedelta})` bug
    types = {tp for tp in types if tp.__module__ != 'datetime'}

    names = _get_attr_args(_any_dtype, f'_{name}Name')
    chars = _get_attr_args(_any_dtype, f'_{name}Char')
    return frozenset(types), frozenset(names), frozenset(chars)


@pytest.mark.parametrize(
    'name',
    [*_NUMERIC_N, *_NUMERIC_C, *_SIMPLE, *_TEMPORAL, *_FLEXIBLE],
)
def test_sctypes(name: str):
    dtype_expect = np.dtype(name.lower())
    sctype_expect = dtype_expect.type
    types, names, chars = _get_dtype_info(name)

    assert dtype_expect.type in types

    for arg in types | names | {c for c in chars if c[0] not in '=<>'}:
        dtype: np.dtype[Any] = np.dtype(arg)
        assert (
            dtype == dtype_expect
            # only needed for `np.dtype(ct.c_char)`
            or dtype.type is sctype_expect
        ), f'np.dtype({arg!r}) (= {dtype!r}) != {dtype_expect}'


@pytest.mark.parametrize(
    'name',
    [*_NUMERIC_N, *_SIMPLE, *_TEMPORAL, *_FLEXIBLE],
)
def test_sctype_name(name: str):
    dtype_expect = np.dtype(name.lower())
    _, names, _ = _get_dtype_info(name)

    assert dtype_expect.name in names


@pytest.mark.parametrize(
    'name',
    [*_NUMERIC_C, *_SIMPLE, * _TEMPORAL, * _FLEXIBLE],
)
def test_sctype_char(name: str):
    dtype_expect = np.dtype(name.lower())
    _, _, chars = _get_dtype_info(name)

    assert dtype_expect.char in chars
