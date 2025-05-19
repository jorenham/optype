from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optype.numpy as onp


def test_at_least_0d() -> None:
    s0: onp.AtLeast0D = ()

    # bool <: int, and the variadic type params of a tuple are somehow
    # covariant (even though `TypeVarTuple`'s can only be invariant...)
    s1_bool: onp.AtLeast0D = (True,)

    s1_int: onp.AtLeast0D = (1,)
    s1_str: onp.AtLeast0D = ("1",)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s2: onp.AtLeast0D = 1, 2
    s2_str1: onp.AtLeast0D = "1", 2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s2_str2: onp.AtLeast0D = 1, "2"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_least_1d() -> None:
    s0: onp.AtLeast1D = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s1: onp.AtLeast1D = (0,)
    s1_str: onp.AtLeast1D = ("1",)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s2: onp.AtLeast1D = 1, 2
    s2_str1: onp.AtLeast1D = "1", 2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s2_str2: onp.AtLeast1D = 1, "2"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s3: onp.AtLeast1D = 1, 2, 3
    s3_str1: onp.AtLeast1D = "1", 2, 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s3_str2: onp.AtLeast1D = 1, "2", 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s3_str3: onp.AtLeast1D = 1, 2, "3"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_least_2d() -> None:
    s0: onp.AtLeast2D = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s1: onp.AtLeast2D = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s2: onp.AtLeast2D = 1, 2
    s2_str1: onp.AtLeast2D = "1", 2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s2_str2: onp.AtLeast2D = 1, "2"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s3: onp.AtLeast2D = 1, 2, 3
    s3_str1: onp.AtLeast2D = "1", 2, 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s3_str2: onp.AtLeast2D = 1, "2", 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s3_str3: onp.AtLeast2D = 1, 2, "3"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_least_3d() -> None:
    s0: onp.AtLeast3D = ()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s1: onp.AtLeast3D = (0,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s2: onp.AtLeast3D = 0, 1  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s3: onp.AtLeast3D = 0, 1, 2
    s3_str: onp.AtLeast3D = 0, 1, "2"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_most_0d() -> None:
    s_0: onp.AtMost0D = ()
    s_1: onp.AtMost0D = (1,)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_most_1d() -> None:
    s_0: onp.AtMost1D = ()

    s_1: onp.AtMost1D = (1,)
    s_1_str: onp.AtMost1D = ("1",)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s_2: onp.AtMost1D = 1, 2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_most_2d() -> None:
    s_0: onp.AtMost2D = ()

    s_1: onp.AtMost2D = (1,)
    s_1_str: onp.AtMost2D = ("1",)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s_2: onp.AtMost2D = 1, 2
    s_2_str1: onp.AtMost2D = "1", 2  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_2_str2: onp.AtMost2D = 1, "2"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s_3: onp.AtMost2D = 1, 2, 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_at_most_3d() -> None:
    s_0: onp.AtMost3D = ()
    s_1: onp.AtMost3D = (1,)
    s_2: onp.AtMost3D = 1, 2
    s_3: onp.AtMost3D = 1, 2, 3

    s_3_str1: onp.AtMost3D = "1", 2, 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_3_str2: onp.AtMost3D = 1, "2", 3  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    s_3_str3: onp.AtMost3D = 1, 2, "3"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    s_4: onp.AtMost3D = 1, 2, 3, 4  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
