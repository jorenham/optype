# pyright: reportUnusedVariable=false
# ruff: noqa: F841, PLR0914

from typing import Any, TypeAlias, TypeVar

from optype import (
    CanIAdd,
    CanIAddSelf,
    CanIAnd,
    CanIAndSelf,
    CanIMul,
    CanIMulSelf,
    CanIOr,
    CanIOrSelf,
    CanISub,
    CanISubSelf,
    CanIXor,
    CanIXorSelf,
    CanIter,
    CanIterSelf,
)


_T_ReIter = TypeVar('_T_ReIter')
CanReIter: TypeAlias = CanIter[CanIterSelf[_T_ReIter]]


def test_iadd():
    """
    The `builtins.list` type is the only builtin collection that implements
    `__iadd__`.
    """
    some_list: list[int] = [42]

    x_iadd: CanIAdd[CanReIter[int], list[int]] = some_list
    x_iadd_any_in: CanIAdd[Any, list[int]] = some_list
    x_iadd_any_out: CanIAdd[CanReIter[int], Any] = some_list
    x_iadd_wrong_in: CanIAdd[str, list[int]] = some_list  # pyright: ignore[reportAssignmentType]
    x_iadd_wrong_in_val: CanIAdd[CanReIter[str], list[int]] = some_list  # pyright: ignore[reportAssignmentType]
    x_iadd_wrong_out: CanIAdd[CanReIter[int], list[str]] = some_list  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_list, CanIAdd)

    x_iadd_self: CanIAddSelf[CanReIter[int]] = some_list
    x_iadd_self_any: CanIAddSelf[Any] = some_list
    x_iadd_self_wrong: CanIAddSelf[str] = some_list  # pyright: ignore[reportAssignmentType]
    x_iadd_self_wrong_val: CanIAddSelf[CanReIter[str]] = some_list  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_list, CanIAddSelf)


def test_imul():
    """
    The `builtins.list` type is the only builtin collection that implements
    `__imul__`.
    """
    some_list: list[int] = [42]

    x_imul: CanIMul[int, list[int]] = some_list
    x_imul_any_in: CanIMul[Any, list[int]] = some_list
    x_imul_any_out: CanIMul[int, Any] = some_list
    x_imul_wrong_in: CanIMul[str, list[int]] = some_list  # pyright: ignore[reportAssignmentType]
    x_imul_wrong_out: CanIMul[int, list[str]] = some_list  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_list, CanIMul)

    x_imul_self: CanIMulSelf[int] = some_list
    x_imul_self_any: CanIMulSelf[Any] = some_list
    x_imul_self_wrong: CanIMulSelf[str] = some_list  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_list, CanIMulSelf)


def test_isub():
    """
    The `builtins.set` type is the only builtin collection that implements
    `__isub__`.
    """
    some_set: set[int] = {42}

    x_isub: CanISub[set[int], set[int]] = some_set
    x_isub_any_in: CanISub[Any, set[int]] = some_set
    x_isub_any_out: CanISub[set[int], Any] = some_set
    x_isub_wrong_in: CanISub[int, set[int]] = some_set  # pyright: ignore[reportAssignmentType]
    x_isub_wrong_out: CanISub[set[int], set[str]] = some_set  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, CanISub)

    x_isub_self: CanISubSelf[set[int]] = some_set
    x_isub_self_any: CanISubSelf[Any] = some_set
    x_isub_self_wrong: CanISubSelf[str] = some_set  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, CanISubSelf)


def test_iand():
    """
    The `builtins.set` type is the only builtin collection that implements
    `__iand__`.

    Note that the `set.__iand__` and `set.__isub__` method signatures are
    equivalent, namely `(Self, collections.abc.Set[object]) -> Self`.

    But unfortunately, there is currently no *clean* way to avoid code
    duplication between these tests.
    """
    some_set: set[int] = {42}

    x_iand: CanIAnd[set[int], set[int]] = some_set
    x_iand_any_in: CanIAnd[Any, set[int]] = some_set
    x_iand_any_out: CanIAnd[set[int], Any] = some_set
    x_iand_wrong_in: CanIAnd[int, set[int]] = some_set  # pyright: ignore[reportAssignmentType]
    x_iand_wrong_out: CanIAnd[set[int], set[str]] = some_set  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, CanIAnd)

    x_iand_self: CanIAndSelf[set[int]] = some_set
    x_iand_self_any: CanIAndSelf[Any] = some_set
    x_iand_self_wrong: CanIAndSelf[str] = some_set  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, CanIAndSelf)


def test_ixor():
    """
    The `builtins.set` type is the only builtin collection that implements
    `__ixor__`.
    Its method signature is `(Self[~T], collections.abc.Set[~T]) -> Self[~T]`.
    """
    some_set: set[bool] = {False, True}

    x_ixor: CanIXor[set[bool], set[bool]] = some_set
    x_ixor_any_in: CanIXor[Any, set[bool]] = some_set
    x_ixor_any_out: CanIXor[set[bool], Any] = some_set
    x_ixor_wrong_in: CanIXor[set[str], set[bool]] = some_set  # pyright: ignore[reportAssignmentType]
    x_ixor_wrong_out: CanIXor[set[bool], set[str]] = some_set  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, CanIXor)

    x_ixor_self: CanIXorSelf[set[bool]] = some_set
    x_ixor_self_any: CanIXorSelf[Any] = some_set
    x_ixor_self_wrong: CanIXorSelf[set[str]] = some_set  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, CanIXorSelf)


def test_ior():
    """
    Both the `builtins.set` and the `builtins.dict` standard collection types
    implement `__ior__` method.
    """
    some_set: set[float] = {1 / 137}

    x_ior: CanIOr[set[float], set[float]] = some_set
    x_ior_any_in: CanIOr[Any, set[float]] = some_set
    x_ior_any_out: CanIOr[set[float], Any] = some_set
    x_ior_wrong_in: CanIOr[set[complex], set[float]] = some_set  # pyright: ignore[reportAssignmentType]
    x_ior_wrong_out: CanIOr[set[float], set[int]] = some_set  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, CanIOr)

    x_ior_self: CanIOrSelf[set[float]] = some_set
    x_ior_self_any: CanIOrSelf[Any] = some_set
    x_ior_self_wrong: CanIOrSelf[set[str]] = some_set  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_set, CanIOrSelf)

    some_dict: dict[bytes, int] = {b'answer': 0x2a}

    y_ior: CanIOr[dict[bytes, int], dict[bytes, int]] = some_dict
    y_ior_any_in: CanIOr[Any, dict[bytes, int]] = some_dict
    y_ior_any_out: CanIOr[dict[bytes, int], Any] = some_dict
    y_ior_wrong_in: CanIOr[dict[str, int], dict[bytes, int]] = some_dict  # pyright: ignore[reportAssignmentType]
    y_ior_wrong_out: CanIOr[dict[bytes, int], dict[str, int]] = some_dict  # pyright: ignore[reportAssignmentType]

    assert isinstance(some_dict, CanIOr)

    y_ior_self: CanIOrSelf[dict[bytes, int]] = some_dict
    y_ior_self_any: CanIOrSelf[Any] = some_dict
    y_ior_self_wrong: CanIOrSelf[dict[str, int]] = some_dict  # pyright: ignore[reportAssignmentType]
    assert isinstance(some_dict, CanIOrSelf)
