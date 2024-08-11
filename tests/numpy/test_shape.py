from typing import TYPE_CHECKING, Literal, TypeAlias


if TYPE_CHECKING:
    import sys

    import optype.numpy as onp
    if sys.version_info >= (3, 13):
        from typing import Never
    else:
        from typing_extensions import Never


Neg1: TypeAlias = Literal[-1]
Pos0: TypeAlias = Literal[0]
Pos1: TypeAlias = Literal[1]
Pos2: TypeAlias = Literal[2]
Pos3: TypeAlias = Literal[3]
Pos4: TypeAlias = Literal[4]


def test_at_least_0d() -> None:
    s0: onp.AtLeast0D = ()
    s0_lit: onp.AtLeast0D[Neg1] = ()

    # bool <: int, and the variadic type params of a tuple are somehow
    # covariant (even though `TypeVarTuple`'s can only be invariant...)
    s1_bool: onp.AtLeast0D[bool] = (True,)

    s1_int: onp.AtLeast0D = (1,)
    s1_lit: onp.AtLeast0D[Pos1] = (1,)
    s1_lit_wrong: onp.AtLeast0D[Pos0] = (1,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s2: onp.AtLeast0D = 1, 2
    s2_lit: onp.AtLeast0D[Pos1 | Pos2] = 1, 2
    s2_int_int: onp.AtLeast0D[int, int] = 1, 2  # type: ignore[type-arg]  # pyright: ignore[reportInvalidTypeForm]


def test_at_least_1d() -> None:
    s0: onp.AtLeast1D = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s0_lit: onp.AtLeast1D[Pos0] = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s0_lit_lit: onp.AtLeast1D[Pos0, Pos1] = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s1: onp.AtLeast1D = (0,)
    s1_lit: onp.AtLeast1D[Pos0] = (0,)
    s1_wrong: onp.AtLeast1D[Pos0] = (-1,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s1_lit_lit: onp.AtLeast1D[Pos0, Pos1] = (0,)

    s2: onp.AtLeast1D = 0, 1
    s2_lit: onp.AtLeast1D[Pos0] = 0, 1
    s2_wrong: onp.AtLeast1D[Pos0] = -1, 1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s2_lit_lit: onp.AtLeast1D[Pos0, Neg1] = 0, -1
    s2_lit_wrong: onp.AtLeast1D[Pos0, Neg1] = 0, 42  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s3_lit_lit: onp.AtLeast1D[Pos0, Neg1] = 0, -1, -1
    s3_lit_lit_lit: onp.AtLeast1D[Pos0, Neg1, Pos2] = 0, -1, -1  # type: ignore[type-arg]  # pyright: ignore[reportInvalidTypeForm]


def test_at_least_2d() -> None:
    s0: onp.AtLeast2D = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s0_lit_lit: onp.AtLeast2D[Pos0, Pos1] = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s1: onp.AtLeast2D = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s1_lit_lit: onp.AtLeast2D[Pos0, Neg1] = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s2: onp.AtLeast2D = 0, 1
    s2_lit: onp.AtLeast2D[Pos0] = 0, 1
    s2_wrong: onp.AtLeast2D[Pos0] = -1, 1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s2_lit_lit: onp.AtLeast2D[Pos0, Pos1] = 0, 1
    s2_lit_wrong: onp.AtLeast2D[Pos0, Pos1] = 0, 42  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s2_lit_lit_lit: onp.AtLeast2D[Pos0, Pos1, Neg1] = 0, 1

    s3: onp.AtLeast2D = 0, 1, 2
    s3_lit_lit: onp.AtLeast2D[Pos0, Pos1] = 0, 1, 2
    s3_lit_lit_lit: onp.AtLeast2D[Pos0, Pos1, Pos2] = 0, 1, 2
    s3_lit_lit_wrong: onp.AtLeast2D[Pos0, Pos1, Pos2] = 0, 1, 42  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_least_3d() -> None:
    s0: onp.AtLeast3D = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s0_lit_lit_lit: onp.AtLeast3D[Pos0, Pos1, Pos2] = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s1: onp.AtLeast3D = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s1_lit_lit_lit: onp.AtLeast3D[Pos0, Pos1, Pos2] = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s2: onp.AtLeast3D = 0, 1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s2_lit_lit_lit: onp.AtLeast3D[Pos0, Pos1, Pos2] = 0, 1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s3: onp.AtLeast3D = 0, 1, 2
    s3_lit_lit_lit: onp.AtLeast3D[Pos0, Pos1, Pos2] = 0, 1, 2
    s3_lit_lit_lit_lit: onp.AtLeast3D[Pos0, Pos1, Pos2, Pos3] = 0, 1, 2
    s3_lit_lit_lit_wrong: onp.AtLeast3D[Pos0, Pos1, Pos2] = 0, 1, 42  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_most_0d() -> None:
    s_0: onp.AtMost0D = ()
    s_1: onp.AtMost0D = (1,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_most_1d() -> None:
    s_0: onp.AtMost1D = ()
    s_0_int: onp.AtMost1D[int] = ()
    s_0_never: onp.AtMost1D[Never] = ()

    s_1: onp.AtMost1D = (1,)
    s_1_int: onp.AtMost1D[int] = (1,)
    s_1_lit: onp.AtMost1D[Pos1] = (1,)

    s_1_never: onp.AtMost1D[Never] = (1,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s_2: onp.AtMost1D = 1, 2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_most_2d() -> None:
    s_0: onp.AtMost2D = ()
    s_0_int: onp.AtMost2D[int] = ()
    s_0_int_int: onp.AtMost2D[int, int] = ()
    s_0_never: onp.AtMost2D[Never] = ()
    s_0_never_never: onp.AtMost2D[Never, Never] = ()

    s_1: onp.AtMost2D = (1,)
    s_1_lit: onp.AtMost2D[Pos1] = (1,)
    s_1_lit_lit: onp.AtMost2D[Pos1, Pos2] = (1,)
    s_1_lit_never: onp.AtMost2D[Pos1, Never] = (1,)
    s_1_never: onp.AtMost2D[Never] = (1,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_1_never_never: onp.AtMost2D[Never, Never] = (1,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s_2: onp.AtMost2D = 1, 2
    s_2_lit: onp.AtMost2D[Pos1] = 1, 2
    s_2_lit_lit: onp.AtMost2D[Pos1, Pos2] = 1, 2
    s_2_lit_never: onp.AtMost2D[Pos1, Never] = 1, 2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_2_never: onp.AtMost2D[Never] = 1, 2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_2_never_never: onp.AtMost2D[Never, Never] = 1, 2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s_3: onp.AtMost2D = 1, 2, 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_most_3d() -> None:
    s_0: onp.AtMost3D = ()
    s_1: onp.AtMost3D = (1,)
    s_2: onp.AtMost3D = 1, 2
    s_3: onp.AtMost3D = 1, 2, 3

    s_3_lit: onp.AtMost3D[Pos1] = 1, 2, 3
    s_3_lit_lit: onp.AtMost3D[Pos1, Pos2] = 1, 2, 3
    s_3_lit_lit_lit: onp.AtMost3D[Pos1, Pos2, Pos3] = 1, 2, 3

    s_4: onp.AtMost3D = 1, 2, 3, 4  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s_3_wrong1: onp.AtMost3D[Pos0, Pos2, Pos3] = 1, 2, 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_3_wrong2: onp.AtMost3D[Pos1, Pos0, Pos3] = 1, 2, 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_3_wrong3: onp.AtMost3D[Pos1, Pos2, Pos0] = 1, 2, 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
