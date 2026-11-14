import warnings

import pytest

import optype.test
from optype.test import UncheckedWarning, assert_subtype, assert_type


def test_assert_subtype() -> None:
    assert_subtype[int](True)
    assert_subtype[int](1)
    with pytest.raises(AssertionError):
        assert_subtype[int](1.0)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
    with pytest.raises(AssertionError):
        assert_subtype[int](object())  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


def test_assert_type() -> None:
    x = int("0")
    assert assert_type(x, int) is x
    with pytest.raises(AssertionError):
        assert_type(x, bool)  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure]
    assert_type(x, object)  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure]


def test_unchecked_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optype.test, "die_if_unbearable", None)
    monkeypatch.setattr(optype.test, "_warned", False)
    x = int("0")

    with pytest.warns(UncheckedWarning):
        assert_type(x, int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert_type(x, int)
        assert_subtype[int](x)
