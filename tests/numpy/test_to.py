from typing import Annotated, get_args, get_origin

import numpy as np
import pytest
from beartype.door import is_bearable

import optype.numpy as onp

_FAMILIES = [
    ("Bool", np.bool_, np.bool_),
    ("Int64", np.int64, np.int64),
    ("Float16", np.float16, np.float16),
    ("Float32", np.float32, np.float32),
    ("Float64", np.float64, np.float64),
    ("LongDouble", np.longdouble, np.longdouble),
    ("Complex64", np.complex64, np.complex64),
    ("Complex128", np.complex128, np.complex128),
    ("CLongDouble", np.clongdouble, np.clongdouble),
    ("Int", np.integer, (np.integer, np.bool_)),
    ("Float", np.floating, (np.floating, np.integer, np.bool_)),
    ("Complex", np.complexfloating, (np.number, np.bool_)),
]

_NDIMS = [0, 1, 2, 3, None]

_NO_COERCIBLE = {"Int64", "LongDouble", "CLongDouble"}

_VALUES = [
    True,
    1,
    0.5,
    1j,
    "",
    object(),
    np.int8(1),
    np.float16(1),
    np.float64(1),
    np.longdouble(1),
    np.complex128(1 + 1j),
    np.timedelta64(1, "s"),
    np.array([True, False]),
    np.array([1.0, 2.0], dtype=np.float32),
    [[1, 2]],
    np.array([[[1.0]]], dtype=np.float16),
]


def _alias_name(family: str, nd: int | None, just: bool) -> str:
    prefix = "Just" if just else ""
    if nd == 0:
        return f"To{prefix}{family}"
    base = f"{family}_" if family[-1].isdigit() else family
    return f"To{prefix}{base}{'ND' if nd is None else f'{nd}D'}"


_ALL_NAMES = [
    _alias_name(f, nd, j)
    for f, _, _ in _FAMILIES
    for nd in _NDIMS
    for j in (False, True)
    if j or f not in _NO_COERCIBLE
] + [
    f"To{'Just' if j else ''}{f}Strict{nd}D"
    for f, _, _ in _FAMILIES
    for nd in (1, 2, 3)
    for j in (False, True)
    if j or f not in _NO_COERCIBLE
]


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_annotated(name: str) -> None:
    assert get_origin(getattr(onp, name)) is Annotated


@pytest.mark.parametrize("just", [False, True], ids=["co", "just"])
@pytest.mark.parametrize("nd", _NDIMS)
@pytest.mark.parametrize("value", _VALUES, ids=lambda v: repr(v)[:30])
@pytest.mark.parametrize(
    ("family", "just_sct", "co_target"),
    _FAMILIES,
    ids=[f[0] for f in _FAMILIES],
)
def test_to(  # noqa: PLR0913, PLR0917
    family: str,
    just_sct: type[np.generic],
    co_target: type[np.generic] | tuple[type, ...],
    value: object,
    nd: int | None,
    just: bool,
) -> None:
    if not just and family in _NO_COERCIBLE:
        pytest.skip(f"No coercible To{family}")

    alias = getattr(onp, _alias_name(family, nd, just))

    try:
        arr = np.asanyarray(value)
    except (ValueError, TypeError):
        predicate = False
    else:
        target = just_sct if just else co_target
        if nd is not None and arr.ndim != nd:
            predicate = False
        elif just or isinstance(target, tuple):
            predicate = issubclass(arr.dtype.type, target)
        else:
            predicate = bool(np.can_cast(arr.dtype, target))

    plain = get_args(alias)[0]
    assert is_bearable(value, alias) is (is_bearable(value, plain) and predicate)
