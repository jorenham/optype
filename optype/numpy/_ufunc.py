from collections.abc import Callable
from typing import (
    Any,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)


_N_in_co = TypeVar('_N_in_co', bound=int, covariant=True)
_N_arg_co = TypeVar('_N_arg_co', bound=int, covariant=True)
_N_out_co = TypeVar('_N_out_co', bound=int, covariant=True)
_N_tp_co = TypeVar('_N_tp_co', bound=int, covariant=True)
_T_sig_co = TypeVar('_T_sig_co', bound=str | None, covariant=True)
_T_id_co = TypeVar('_T_id_co', bound=object | None, covariant=True)


AnyUfuncMethod: TypeAlias = Literal[
    '__call__',
    'reduce',
    'reduceat',
    'accumulate',
    'outer',
    'inner',
]
AnyCastKind: TypeAlias = Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe']
AnyOrder: TypeAlias = Literal['K', 'A', 'C', 'F']


class HasUfuncAttrs(
    Protocol[_N_in_co, _N_arg_co, _N_out_co, _N_tp_co, _T_sig_co, _T_id_co],
):
    @property
    def __name__(self) -> str: ...
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
class AnyUfunc(HasUfuncAttrs[Any, Any, Any, Any, Any, Any], Protocol):

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


_F_contra = TypeVar('_F_contra', bound=AnyUfunc, contravariant=True)
_Xss = ParamSpec('_Xss')
_Y_co = TypeVar('_Y_co', covariant=True)


@runtime_checkable
class CanArrayUfunc(Protocol[_F_contra, _Xss, _Y_co]):
    """
    Interface for ufunc operands.

    See Also:
        - https://numpy.org/devdocs/reference/arrays.classes.html
    """
    def __array_ufunc__(
        self,
        ufunc: _F_contra,
        method: AnyUfuncMethod,
        *inputs: _Xss.args,
        **kwargs: _Xss.kwargs,
    ) -> _Y_co: ...
