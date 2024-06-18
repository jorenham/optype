from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, type_check_only

import numpy as np


if sys.version_info >= (3, 13):
    from types import CapsuleType
    from typing import (
        Protocol,
        Self,
        TypeVar,
        final,
        overload,
        override,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        CapsuleType,  # noqa: TCH002
        Protocol,
        Self,  # noqa: TCH002
        TypeVar,
        final,
        overload,
        override,
        runtime_checkable,
    )


if TYPE_CHECKING:
    import numpy.typing as npt
    from numpy._core.multiarray import flagsobj


_L0: TypeAlias = Literal[0]
_L1: TypeAlias = Literal[1]

_DT_Array0D = TypeVar('_DT_Array0D', bound=np.dtype[Any])
_Array0D: TypeAlias = np.ndarray[tuple[()], _DT_Array0D]

# fmt: off
_DTypeKind: TypeAlias = Literal[
    'b', 'i', 'u', 'f', 'c', 'm', 'M', 'O', 'S', 'U', 'V',
]
_DTypeChar: TypeAlias = Literal[
    # boolean
    '?',
    # unsigned integer
    'B', 'H', 'I', 'L', 'Q', 'P', 'N',
    # signed integer
    'b', 'h', 'i', 'l', 'q', 'p', 'n',
    # floating
    'e', 'f', 'd', 'g',
    # complex
    'F', 'D', 'G',
    # temporal
    'M', 'm',
    # flexible
    'U', 'S', 'V',
    # object
    'O',

]
# unlike the docs claim, there >21 different built-in types; let's assume <=32
_DTypeNum: TypeAlias = Literal[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
]
if sys.byteorder == 'little':
    _DTypeStr: TypeAlias = Literal[
        # boolean
        '|b1',
        # unsigned integer
        '|u1', '<u2', '<u4', '<u8', '<u16', '<u32',
        # signed integer
        '|i1', '<i2', '<i4', '<i8', '<i16', '<i32',
        # floating
        '<f2', '<f4', '<f8', '<f10', '<f12', '<f16', '<f32',
        # complex floating
        '<c8', '<c16', '<c20', '<c24', '<c32', '<c64',
        # temporal
        '<M8', '<m8',
        # flexible
        '<U0', '|S0', '|V0',
        # object
        '|O',
    ]
else:
    _DTypeStr: TypeAlias = Literal[
        # boolean
        '|b1',
        # unsigned integer
        '|u1', '>u2', '>u4', '>u8', '>u16', '>u32',
        # signed integer
        '|i1', '>i2', '>i4', '>i8', '>i16', '>i32',
        # floating
        '>f2', '>f4', '>f8', '>f10', '>f12', '>f16', '>f32',
        # complex floating
        '>c8', '>c16', '>c20', '>c24', '>c32', '>c64',
        # temporal
        '>M8', '>m8',
        # flexible
        '>U0', '|S0', '|V0',
        # object
        '|O',
    ]
_DTypeName: TypeAlias = Literal[
    'bool',
    'uint8', 'uint16', 'uint32', 'uint64', 'uint128', 'uint256',
    'int8', 'int16', 'int32', 'int64', 'int128', 'int256',
    'float16', 'float32', 'float64',
    'float80', 'float96', 'float128', 'float256',
    'complex64', 'complex128',
    'complex160', 'complex192', 'complex256', 'complex512',
    'datetime64', 'timedelta64',
    'str', 'bytes', 'void',
    'object',
]
_DTypeItemsize: TypeAlias = Literal[0, 1, 2, 4, 8, 10, 12, 16, 20, 24, 32, 64]
_DTypeByteOrder: TypeAlias = Literal[
    '=',  # native
    '<',  # little-endian
    '>',  # big-endian
    '|',  # not applicable
]
_DTypeByteOrderArg: TypeAlias = _DTypeByteOrder | Literal[
    'S',  # swap
    'L',  # little-indian
    'B',  # big-indian
    'N',  # native
    'I',  # ignore
]
# fmt: on


# ScalarDType

_T_DType = TypeVar('_T_DType', infer_variance=True, bound='Scalar[Any, Any]')
_N_DType = TypeVar('_N_DType', bound=int)


@final
@type_check_only
class ScalarDType(Protocol[_T_DType]):
    """
    Interface for a non-composite `np.dtype`, with a more flexible type
    parameter (`T: Scalar[Any, Any]`) than that of `np.dtype[T: np.generic]`.
    """
    @property
    def type(self, /) -> type[_T_DType]: ...
    @property
    def kind(self, /) -> _DTypeKind: ...
    @property
    def char(self, /) -> _DTypeChar: ...
    @property
    def num(self, /) -> _DTypeNum: ...
    @property
    def str(self, /) -> _DTypeStr: ...
    @property
    def name(self, /) -> _DTypeName: ...
    @property
    def itemsize(self, /) -> _DTypeItemsize: ...  # != Scalar.itemsize
    @property
    def byteorder(self, /) -> _DTypeByteOrder: ...

    @property
    def fields(self, /) -> None: ...
    @property
    def names(self, /) -> None: ...
    @property
    def subdtype(self, /) -> None: ...
    @property
    def shape(self, /) -> tuple[()]: ...
    @property
    def ndim(self, /) -> Literal[0]: ...

    @property
    def hasobject(self, /) -> bool: ...
    @property
    def flags(self, /) -> Literal[0, 8, 63]: ...
    @property
    def isbuiltin(self, /) -> Literal[1]: ...
    @property
    def isnative(self, /) -> Literal[True]: ...
    @property
    def isalignedstruct(self, /) -> Literal[False]: ...

    @property
    def descr(self, /) -> list[tuple[Literal[''], _DTypeStr]]: ...
    @property
    def alignment(self: ScalarDType[Scalar[Any, _N_DType]], /) -> _N_DType: ...
    @property
    def base(self, /) -> Self | np.dtype[Any]: ...
    @property
    def metadata(self, /) -> None: ...

    def newbyteorder(self, order: _DTypeByteOrderArg = ..., /) -> Self: ...

    # I still have no idea what these are supposed to do in `np.dtype`...
    def __gt__(self, rhs: npt.DTypeLike, /) -> bool: ...
    def __ge__(self, rhs: npt.DTypeLike, /) -> bool: ...
    def __lt__(self, rhs: npt.DTypeLike, /) -> bool: ...
    def __le__(self, rhs: npt.DTypeLike, /) -> bool: ...


# Scalar

_T_Scalar = TypeVar('_T_Scalar', infer_variance=True, bound=object)
_N_Scalar = TypeVar('_N_Scalar', infer_variance=True, bound=int, default=Any)
_DT_Scalar = TypeVar('_DT_Scalar', bound=np.dtype[Any])


@runtime_checkable
class Scalar(Protocol[_T_Scalar, _N_Scalar]):
    """
    A lightweight `numpy.generic` interface that's actually generic, and
    doesn't require all that nasty `numpy.typing.NBitBase` stuff.
    """

    def item(self, k: _L0 | tuple[()] | tuple[_L0] = ..., /) -> _T_Scalar: ...
    # unfortunately `| int` is required for compat with `numpy.__init__.pyi`
    @property
    def itemsize(self, /) -> _N_Scalar | int: ...

    @property
    def base(self, /) -> None: ...
    @property
    def data(self, /) -> memoryview: ...
    @property
    def dtype(self, /) -> np.dtype[Self]: ...  # pyright: ignore[reportInvalidTypeArguments]
    @property
    def flags(self, /) -> flagsobj: ...
    @property
    def nbytes(self, /) -> int: ...
    @property
    def ndim(self, /) -> _L0: ...
    @property
    def shape(self, /) -> tuple[()]: ...
    @property
    def size(self, /) -> _L1: ...
    @property
    def strides(self, /) -> tuple[()]: ...

    @property
    def __array_priority__(self, /) -> float: ...  # -1000000.0
    @property
    def __array_interface__(self, /) -> dict[str, Any]: ...  # TypedDict?
    @property
    def __array_struct__(self, /) -> CapsuleType: ...

    @override
    def __hash__(self, /) -> int: ...
    @override
    def __eq__(self, other: object, /) -> np.bool_: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ne__(self, other: object, /) -> np.bool_: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    def __bool__(self, /) -> bool: ...
    def __bytes__(self, /) -> bytes: ...

    if sys.version_info >= (3, 12):
        def __buffer__(self, flags: int, /) -> memoryview: ...

    def __copy__(self, /) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any] | None, /) -> Self: ...

    @overload
    def __array__(self, /) -> _Array0D[np.dtype[Self]]: ...  # pyright: ignore[reportInvalidTypeArguments]
    @overload
    def __array__(self, dtype: _DT_Scalar, /) -> _Array0D[_DT_Scalar]: ...
