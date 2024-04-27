from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    TypedDict,
    overload,
    runtime_checkable,
)

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable

    from ._array import CanArray, SomeArray, SomeTensor


if sys.version_info < (3, 11):
    from typing_extensions import LiteralString, Unpack
else:
    from typing import LiteralString, Unpack


_PyScalar: TypeAlias = bool | int | float | complex | str | bytes

_ND = TypeVar('_ND', bound=tuple[int, ...])
_S = TypeVar('_S', bound=np.generic)

_T = TypeVar('_T')
_OneOrMore: TypeAlias = _T | tuple[_T, Unpack[tuple[_T, ...]]]


class UFuncKwargs(Generic[_ND, _S], TypedDict, total=False):
    where: SomeArray[_ND, np.bool_, bool] | None
    casting: Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe']
    order: Literal['K', 'A', 'C', 'F'] | None
    dtype: np.dtype[_S]
    subok: bool
    signature: _OneOrMore[np.dtype[_S] | LiteralString]


@runtime_checkable
class UFunc(Protocol):
    __name__: Final[LiteralString]
    nin: Final[int]
    nout: Final[int]
    nargs: Final[int]
    ntypes: Final[int]

    @property
    def types(self) -> list[LiteralString]: ...
    identity: Final[_PyScalar | None]
    signature: Final[LiteralString | None]

    @overload
    def __call__(
        self,
        *inputs: SomeArray[Any, Any, Any],
        out: None = ...,
        **kwargs: Unpack[UFuncKwargs[tuple[()], _S]],
    ) -> _OneOrMore[CanArray[Any, _S]]: ...
    @overload
    def __call__(
        self,
        *inputs: SomeArray[Any, Any, Any],
        out: CanArray[_ND, _S],
        **kwargs: Unpack[UFuncKwargs[tuple[()], _S]],
    ) -> _OneOrMore[CanArray[_ND, _S]]: ...

    def at(
        self,
        a: CanArrayUFunc[..., Any],
        indices: SomeTensor[Any, np.bool_ | np.integer[Any], bool | int],
        /,
    ) -> None: ...

    # yea... well... numpy does it too...
    reduce: Callable[..., Any]
    reduceat: Callable[..., Any]
    accumulate: Callable[..., Any]
    outer: Callable[..., Any]


UFuncMethod: TypeAlias = Literal[
    '__call__',
    'reduce',
    'reduceat',
    'accumulate',
    'outer',
    'inner',
]

_Xss = ParamSpec('_Xss')
_Y_co = TypeVar('_Y_co', covariant=True)


class CanArrayUFunc(Protocol[_Xss, _Y_co]):
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: UFuncMethod,
        *inputs: _Xss.args,
        **kwargs: _Xss.kwargs,
    ) -> _Y_co: ...
