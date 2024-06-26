import sys
from typing import TypeAlias


if sys.version_info >= (3, 13):
    from typing import TypeVar, TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVar, TypeVarTuple, Unpack


# fmt: off
__all__ = (
    'AtLeast0D', 'AtLeast1D', 'AtLeast2D', 'AtLeast3D',
    'AtMost0D', 'AtMost1D', 'AtMost2D', 'AtMost3D',
)
# fmt: on


# 0D (scalar)

_Ns_0D = TypeVarTuple('_Ns_0D', default=Unpack[tuple[int, ...]])
AtLeast0D: TypeAlias = tuple[Unpack[_Ns_0D]]
"""Shapes with `ndim >= 0`, i.e. like `tuple[*(Ns=int)]`."""
AtMost0D: TypeAlias = tuple[()]
"""Shapes with `ndim <= 0`, i.e. *just* the empty tuple `()`."""


# 1D (vector)

_N0_1D = TypeVar('_N0_1D', bound=int, default=int)
_Ns_1D = TypeVarTuple('_Ns_1D', default=Unpack[tuple[int, ...]])
AtLeast1D: TypeAlias = tuple[_N0_1D, Unpack[_Ns_1D]]
"""Shapes with `ndim >= 1`, i.e. like `tuple[N0=int, *(Ns=int)]`."""
AtMost1D: TypeAlias = tuple[_N0_1D] | AtMost0D
"""Shapes with `ndim <= 1`, i.e. like `(N0: int = int, *AtMost0D) `."""


# 2D (matrix)

_N0_2D = TypeVar('_N0_2D', bound=int, default=int)
_N1_2D = TypeVar('_N1_2D', bound=int, default=int)
_Ns_2D = TypeVarTuple('_Ns_2D', default=Unpack[tuple[int, ...]])
AtLeast2D: TypeAlias = tuple[_N0_2D, _N1_2D, Unpack[_Ns_2D]]
"""Shapes with `ndim >= 2`, i.e. like `tuple[N0=int, N1=N0, *(Ns=int)]`."""
AtMost2D: TypeAlias = tuple[_N0_2D, _N1_2D] | AtMost1D[_N0_2D]
"""
Shapes with `ndim <= 2`, i.e. like `(N0: int = int, *AtMost1D[N1: int = int])`.
"""


# 3D

_N0_3D = TypeVar('_N0_3D', bound=int, default=int)
_N1_3D = TypeVar('_N1_3D', bound=int, default=int)
_N2_3D = TypeVar('_N2_3D', bound=int, default=int)
_Ns_3D = TypeVarTuple('_Ns_3D', default=Unpack[tuple[int, ...]])
AtLeast3D: TypeAlias = tuple[_N0_3D, _N1_3D, _N2_3D, Unpack[_Ns_3D]]
"""
Shapes with `ndim >= 3`, i.e. like `tuple[N0=int, N1=int, N2=int, *(Ns=int)]`
"""
AtMost3D: TypeAlias = tuple[_N0_3D, _N1_3D, _N2_3D] | AtMost2D[_N0_3D, _N1_3D]
"""
Shapes with `ndim <= 3`, i.e. like
`(N0: int = int, *AtMost2D[N1: int = int, N2: int = int])`.
"""
