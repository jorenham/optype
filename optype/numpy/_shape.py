import sys
from typing import Literal, TypeAlias

if sys.version_info >= (3, 13):
    from typing import TypeAliasType, Unpack
else:
    from typing_extensions import TypeAliasType, Unpack


__all__ = [
    "AtLeast0D", "AtLeast1D", "AtLeast2D", "AtLeast3D",
    "AtMost0D", "AtMost1D", "AtMost2D", "AtMost3D",
    "NDim", "NDim0",
]  # fmt: skip


def __dir__() -> list[str]:
    return __all__


###


# 0D (scalar)

AtLeast0D: TypeAlias = tuple[int, ...]
"""Shape with `ndim >= 0`, i.e. like `(Ns: int = int, ...)`."""
AtMost0D: TypeAlias = tuple[()]
"""Shape with `ndim <= 0`, i.e. *just* the empty tuple `()`."""

# 1D (vector)
AtLeast1D = TypeAliasType("AtLeast1D", tuple[int, Unpack[tuple[int, ...]]])
"""Shape with `ndim >= 1`, i.e. like `(N0: int = int, *AtLeast0D[Ns?])`."""
AtMost1D = TypeAliasType("AtMost1D", tuple[int] | tuple[()])
"""Shape with `ndim <= 1`, i.e. like `(N0: int = int, *AtMost0D) `."""

# 2D (matrix)

AtLeast2D = TypeAliasType("AtLeast2D", tuple[int, int, Unpack[tuple[int, ...]]])
"""Shape with `ndim >= 2`, i.e. like `(N0: int = int, *AtLeast1D[N1?, Ns?])`."""
AtMost2D = TypeAliasType("AtMost2D", tuple[int, int] | tuple[int] | tuple[()])
"""Shape with `ndim <= 2`, i.e. like `(N0: int = int, *AtMost1D[N1])`."""

# 3D (cuboid / tensor)

AtLeast3D = TypeAliasType("AtLeast3D", tuple[int, int, int, Unpack[tuple[int, ...]]])
"""Shape with `ndim >= 3`, i.e. like `(N0: int = int, *AtLeast2D[N1?, N2?, Ns?])`."""
AtMost3D = TypeAliasType(
    "AtMost3D",
    tuple[int, int, int] | tuple[int, int] | tuple[int] | tuple[()],
)
"""Shape with `ndim <= 3`, i.e. like `(N0: int = int, *AtMost2D[N1, N2])`."""

# ND

# NOTE: on `numpy<2` this was at most 32
_NDim0: TypeAlias = Literal[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
]  # fmt: skip
NDim0 = TypeAliasType("NDim0", _NDim0)
NDim = TypeAliasType("NDim", Literal[0, _NDim0])
