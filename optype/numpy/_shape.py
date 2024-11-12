import sys
from typing import TypeAlias


if sys.version_info >= (3, 13):
    from typing import TypeAliasType, TypeVar, Unpack
else:
    from typing_extensions import TypeAliasType, TypeVar, Unpack


__all__ = [
    "AtLeast0D", "AtLeast1D", "AtLeast2D", "AtLeast3D",
    "AtMost0D", "AtMost1D", "AtMost2D", "AtMost3D",
]  # fmt: skip


_N0 = TypeVar("_N0", bound=int, default=int)
_N1 = TypeVar("_N1", bound=int, default=int)
_N2 = TypeVar("_N2", bound=int, default=int)
_Ns = TypeVar("_Ns", bound=int, default=int)


# 0D (scalar)

AtLeast0D: TypeAlias = tuple[_Ns, ...]
"""Shape with `ndim >= 0`, i.e. like `(Ns: int = int, ...)`."""
AtMost0D: TypeAlias = tuple[()]
"""Shape with `ndim <= 0`, i.e. *just* the empty tuple `()`."""

# 1D (vector)
AtLeast1D = TypeAliasType(
    "AtLeast1D",
    tuple[_N0, Unpack[tuple[_Ns, ...]]],
    type_params=(_N0, _Ns),
)
"""Shape with `ndim >= 1`, i.e. like `(N0: int = int, *AtLeast0D[Ns?])`."""
AtMost1D = TypeAliasType("AtMost1D", tuple[_N0] | tuple[()], type_params=(_N0,))
"""Shape with `ndim <= 1`, i.e. like `(N0: int = int, *AtMost0D) `."""


# 2D (matrix)

AtLeast2D = TypeAliasType(
    "AtLeast2D",
    tuple[_N0, _N1, Unpack[tuple[_Ns, ...]]],
    type_params=(_N0, _N1, _Ns),
)
"""Shape with `ndim >= 2`, i.e. like `(N0: int = int, *AtLeast1D[N1?, Ns?])`."""
AtMost2D = TypeAliasType(
    "AtMost2D",
    tuple[_N0, _N1] | tuple[_N0] | tuple[()],
    type_params=(_N0, _N1),
)
"""Shape with `ndim <= 2`, i.e. like `(N0: int = int, *AtMost1D[N1])`."""


# 3D (cuboid / tensor)

AtLeast3D = TypeAliasType(
    "AtLeast3D",
    tuple[_N0, _N1, _N2, Unpack[tuple[_Ns, ...]]],
    type_params=(_N0, _N1, _N2, _Ns),
)
"""Shape with `ndim >= 3`, i.e. like `(N0: int = int, *AtLeast2D[N1?, N2?, Ns?])`."""
AtMost3D = TypeAliasType(
    "AtMost3D",
    tuple[_N0, _N1, _N2] | tuple[_N0, _N1] | tuple[_N0] | tuple[()],
    type_params=(_N0, _N1, _N2),
)
"""Shape with `ndim <= 3`, i.e. like `(N0: int = int, *AtMost2D[N1, N2])`."""
