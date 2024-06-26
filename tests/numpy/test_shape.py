# ruff: noqa: F841, PYI042, ERA001
from typing import TYPE_CHECKING, Literal, TypeAlias

import optype.numpy as onp  # noqa: TCH001


if TYPE_CHECKING:
    import sys
    if sys.version_info >= (3, 13):
        from typing import Never
    else:
        from typing_extensions import Never


_0: TypeAlias = Literal[0]
_1: TypeAlias = Literal[1]
_2: TypeAlias = Literal[2]
_3: TypeAlias = Literal[3]
_4: TypeAlias = Literal[4]


def test_at_least_0d():
    s: onp.AtLeast0D = ()

    # bool <: int, and the variadic type params of a tuple are somehow
    # covariant (even though `TypeVarTuple`'s can only be invariant...)
    s_bool: onp.AtLeast0D[bool] = (True,)

    s_int: onp.AtLeast0D = (1,)
    s_lit: onp.AtLeast0D[_1] = (1,)
    s_lit_wrong: onp.AtLeast0D[_0] = (1,)  # pyright: ignore[reportAssignmentType]

    s_int_int: onp.AtLeast0D = 1, 2
    s_lit_lit: onp.AtLeast0D[_1, _2] = 1, 2
    s_int_missing: onp.AtLeast0D[int] = 1, 2  # pyright: ignore[reportAssignmentType]
    s_int_extra: onp.AtLeast0D[int, int, int] = 1, 2  # pyright: ignore[reportAssignmentType]


def test_at_least_1d():
    s_int0: onp.AtLeast1D = ()  # pyright: ignore[reportAssignmentType]

    s_1: onp.AtLeast1D = (1,)
    s_2: onp.AtLeast1D = 1, 2

    s_int1_2: onp.AtLeast1D[int] = 1, 2
    s_int2: onp.AtLeast1D[int, int] = 1, 2

    s_lit2: onp.AtLeast1D[_1, _2] = 1, 2
    s_lit2_wrong1: onp.AtLeast1D[_0, _2] = 1, 2  # pyright: ignore[reportAssignmentType]
    s_lit2_wrong2: onp.AtLeast1D[_1, _0] = 1, 2  # pyright: ignore[reportAssignmentType]

    s_3: onp.AtLeast1D = 1, 2, 3
    s_int3: onp.AtLeast1D[int, int, int] = 1, 2, 3
    s_int3_missing: onp.AtLeast1D[int, int] = 1, 2, 3  # pyright: ignore[reportAssignmentType]
    s_int3_extra: onp.AtLeast1D[int, int, int, int] = 1, 2, 3  # pyright: ignore[reportAssignmentType]

    s_lit3: onp.AtLeast1D[_1, _2, _3] = 1, 2, 3


def test_at_least_2d():
    s_0: onp.AtLeast2D = ()  # pyright: ignore[reportAssignmentType]
    s_1: onp.AtLeast2D = (1,)  # pyright: ignore[reportAssignmentType]

    s_2: onp.AtLeast2D = 1, 2
    s_int1_2: onp.AtLeast2D[int] = 1, 2
    s_int2: onp.AtLeast2D[int, int] = 1, 2
    s_lit2: onp.AtLeast2D[_1, _2] = 1, 2

    s_3: onp.AtLeast2D = 1, 2, 3
    s_int1_3: onp.AtLeast2D[int] = 1, 2, 3
    s_int2_3: onp.AtLeast2D[int, int] = 1, 2, 3
    s_int3: onp.AtLeast2D[int, int, int] = 1, 2, 3

    s_lit3: onp.AtLeast2D[_1, _2, _3] = 1, 2, 3
    s_lit3_wrong1: onp.AtLeast2D[_0, _2, _3] = 1, 2, 3  # pyright: ignore[reportAssignmentType]
    s_lit3_wrong2: onp.AtLeast2D[_1, _0, _3] = 1, 2, 3  # pyright: ignore[reportAssignmentType]
    s_lit3_wrong3: onp.AtLeast2D[_1, _2, _0] = 1, 2, 3  # pyright: ignore[reportAssignmentType]


def test_at_least_3d():
    s_0: onp.AtLeast3D = ()  # pyright: ignore[reportAssignmentType]
    s_1: onp.AtLeast3D = (1,)  # pyright: ignore[reportAssignmentType]
    s_2: onp.AtLeast3D = 1, 2  # pyright: ignore[reportAssignmentType]

    s_3: onp.AtLeast3D = 1, 2, 3
    s_4: onp.AtLeast3D = 1, 2, 3, 4
    s_int1_4: onp.AtLeast3D[int] = 1, 2, 3, 4
    s_int2_4: onp.AtLeast3D[int, int] = 1, 2, 3, 4
    s_int3_4: onp.AtLeast3D[int, int, int] = 1, 2, 3, 4
    s_int4: onp.AtLeast3D[int, int, int, int] = 1, 2, 3, 4

    s_lit4: onp.AtLeast3D[_1, _2, _3, _4] = 1, 2, 3, 4
    s_lit4_wrong1: onp.AtLeast3D[_0, _2, _3, _4] = 1, 2, 3, 4  # pyright: ignore[reportAssignmentType]
    s_lit4_wrong2: onp.AtLeast3D[_1, _0, _3, _4] = 1, 2, 3, 4  # pyright: ignore[reportAssignmentType]
    s_lit4_wrong3: onp.AtLeast3D[_1, _2, _0, _4] = 1, 2, 3, 4  # pyright: ignore[reportAssignmentType]
    s_lit4_wrong4: onp.AtLeast3D[_1, _2, _3, _0] = 1, 2, 3, 4  # pyright: ignore[reportAssignmentType]


def test_at_most_0d():
    s_0: onp.AtMost0D = ()
    s_1: onp.AtMost0D = (1,)  # pyright: ignore[reportAssignmentType]


def test_at_most_1d():
    s_0: onp.AtMost1D = ()
    s_0_int: onp.AtMost1D[int] = ()
    s_0_never: onp.AtMost1D[Never] = ()

    s_1: onp.AtMost1D = (1,)
    s_1_int: onp.AtMost1D[int] = (1,)
    s_1_lit: onp.AtMost1D[_1] = (1,)

    # Behold! The bug that was "never" meant to be found:
    # https://github.com/microsoft/pyright/issues/8237
    # s_1_never: onp.AtMost1D[Never] = (1,)

    s_2: onp.AtMost1D = 1, 2  # pyright: ignore[reportAssignmentType]


def test_at_most_2d():
    s_0: onp.AtMost2D = ()
    s_0_int: onp.AtMost2D[int] = ()
    s_0_int_int: onp.AtMost2D[int, int] = ()
    s_0_never: onp.AtMost2D[Never] = ()
    s_0_never_never: onp.AtMost2D[Never, Never] = ()

    s_1: onp.AtMost2D = (1,)
    s_1_lit: onp.AtMost2D[_1] = (1,)
    s_1_lit_lit: onp.AtMost2D[_1, _2] = (1,)
    s_1_lit_never: onp.AtMost2D[_1, Never] = (1,)

    # Behold! The bug that was "never" meant to be found:
    # https://github.com/microsoft/pyright/issues/8237
    # s_1_never: onp.AtMost2D[Never] = (1,)

    s_2: onp.AtMost2D = 1, 2
    s_2_lit: onp.AtMost2D[_1] = 1, 2
    s_2_lit_lit: onp.AtMost2D[_1, _2] = 1, 2

    s_3: onp.AtMost2D = 1, 2, 3  # pyright: ignore[reportAssignmentType]


def test_at_most_3d():
    s_0: onp.AtMost3D = ()
    s_1: onp.AtMost3D = (1,)
    s_2: onp.AtMost3D = 1, 2
    s_3: onp.AtMost3D = 1, 2, 3

    s_3_lit: onp.AtMost3D[_1] = 1, 2, 3
    s_3_lit_lit: onp.AtMost3D[_1, _2] = 1, 2, 3
    s_3_lit_lit_lit: onp.AtMost3D[_1, _2, _3] = 1, 2, 3

    s_4: onp.AtMost3D = 1, 2, 3, 4  # pyright: ignore[reportAssignmentType]

    s_3_wrong1: onp.AtMost3D[_0, _2, _3] = 1, 2, 3  # pyright: ignore[reportAssignmentType]
    s_3_wrong2: onp.AtMost3D[_1, _0, _3] = 1, 2, 3  # pyright: ignore[reportAssignmentType]
    s_3_wrong3: onp.AtMost3D[_1, _2, _0] = 1, 2, 3  # pyright: ignore[reportAssignmentType]
