from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    TypeAlias,
)

import numpy as np


if sys.version_info >= (3, 13):
    from typing import (
        Protocol,
        TypeVar,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        Protocol,
        TypeVar,
        runtime_checkable,
    )


if TYPE_CHECKING:
    from collections.abc import Callable


_NP_V2: Final[bool] = np.__version__.startswith('2.')


_N_in_co = TypeVar('_N_in_co', bound=int, covariant=True)
_N_arg_co = TypeVar('_N_arg_co', bound=int, covariant=True)
_N_out_co = TypeVar('_N_out_co', bound=int, covariant=True)
_N_tp_co = TypeVar('_N_tp_co', bound=int, covariant=True)
_T_sig_co = TypeVar('_T_sig_co', bound=str | None, covariant=True)
_T_id_co = TypeVar('_T_id_co', bound=object | None, covariant=True)


class _HasUfuncAttrs(
    Protocol[_N_in_co, _N_arg_co, _N_out_co, _N_tp_co, _T_sig_co, _T_id_co],
):
    @property
    def nin(self) -> _N_in_co: ...
    @property
    def nout(self) -> _N_out_co: ...
    @property
    def nargs(self) -> _N_arg_co: ...
    @property
    def ntypes(self) -> _N_tp_co: ...
    @property
    def types(self) -> list[str]: ...
    @property
    def signature(self) -> _T_sig_co: ...
    @property
    def identity(self) -> _T_id_co: ...


@runtime_checkable
class AnyUfunc(_HasUfuncAttrs[Any, Any, Any, Any, Any, Any], Protocol):

    # this horrible mess is required for numpy.typing compat :(
    @property
    def __call__(self) -> Callable[..., Any]: ...
    @property
    def at(self) -> Callable[..., Any] | None: ...
    @property
    def reduce(self) -> Callable[..., Any] | None: ...
    @property
    def reduceat(self) -> Callable[..., Any] | None: ...
    @property
    def accumulate(self) -> Callable[..., Any] | None: ...
    @property
    def outer(self) -> Callable[..., Any] | None: ...


if _NP_V2:
    _UFuncMethod: TypeAlias = Literal[
        '__call__',
        'reduce',
        'reduceat',
        'accumulate',
        'outer',
        'at',
    ]
else:
    _UFuncMethod: TypeAlias = Literal[
        '__call__',
        'reduce',
        'reduceat',
        'accumulate',
        'outer',
        'inner',
    ]


_F_CanArrayUFunc = TypeVar(
    '_F_CanArrayUFunc',
    infer_variance=True,
    bound=AnyUfunc,
    default=Any,
)


@runtime_checkable
class CanArrayUFunc(Protocol[_F_CanArrayUFunc]):
    """
    Interface for ufunc operands.

    See Also:
        - https://numpy.org/devdocs/reference/arrays.classes.html
    """
    def __array_ufunc__(
        self,
        /,
        ufunc: _F_CanArrayUFunc,
        method: _UFuncMethod,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
