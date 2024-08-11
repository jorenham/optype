from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, TypeAlias as _Type

import numpy as np

import optype.numpy._compat as _x


if sys.version_info >= (3, 13):
    from typing import LiteralString, Protocol, TypeVar, runtime_checkable
else:
    from typing_extensions import (
        LiteralString,
        Protocol,
        TypeVar,
        runtime_checkable,
    )


if TYPE_CHECKING:
    from collections.abc import Callable as CanCall, Mapping
    from types import NotImplementedType

    import optype as opt


__all__ = ['CanArrayFunction', 'CanArrayUFunc', 'UFunc']


_FT_co = TypeVar(
    '_FT_co',
    bound='CanCall[..., Any]',
    covariant=True,
    default=Any,
)
_FT_contra = TypeVar(
    '_FT_contra',
    bound='CanCall[..., Any]',
    contravariant=True,
    default=Any,
)
_NInT_co = TypeVar('_NInT_co', bound=int, covariant=True, default=int)
_NoutT_co = TypeVar('_NoutT_co', bound=int, covariant=True, default=int)
_SigT_co = TypeVar(
    '_SigT_co',
    bound=LiteralString | None,
    covariant=True,
    default=LiteralString | None,
)
_IdT_co = TypeVar(
    '_IdT_co',
    bound=int | float | complex | str | bytes | None,
    covariant=True,
    default=float | None,
)

_AnyArray: _Type = np.ndarray[Any, Any]


@runtime_checkable
class UFunc(Protocol[_FT_co, _NInT_co, _NoutT_co, _SigT_co, _IdT_co]):
    """
    A generic interface for `numpy.ufunc` "universal function" instances,
    e.g. `numpy.exp`, `numpy.add`, `numpy.frexp`, `numpy.divmod`.

    This also includes gufunc's (generalized universion functions), which
    have a specified `signature`, and aren't necessarily element-wise
    functions (which "regular" ufuncs are).
    At the moment, the only gufuncs within numpy are `numpy.matmul`, and
    `numpy.vecdot` (since `numpy>=2`).

    TODO:
        Attempt property overloading (based on type params) of e.g. `nargs`,
        using descriptors.
    """
    @property
    def __call__(self, /) -> _FT_co: ...

    # The number of positional-only parameters, within numpy this is either 1
    # or 2, but might be more for 3rd party ufuncs.
    @property
    def nin(self, /) -> _NInT_co: ...
    # The number of output values, within numpy this is either 1 or 2.
    @property
    def nout(self, /) -> _NoutT_co: ...
    # A string i.f.f. this is a gufunc (generalized ufunc).
    @property
    def signature(self, /) -> _SigT_co: ...

    # If `signature is None and nin == 2 and nout == 1`, this *may* be set to
    # a python scalar s.t. `self(x, identity) == x` for all possible `x`.
    # Within numpy==2.0.0, this is only the case for `multiply` (`1`),
    # `logaddexp` (`-inf`), `logaddexp2` (`-inf`), `logical_and` (`True`),
    # and `bitwise_and` (`-1`).
    # Note that the `complex` return annotation implicitly includes
    # `bool | int | float` (these are its supertypes).
    @property
    def identity(self, /) -> _IdT_co: ...
    # Within numpy this is always `nin + nout`, since each output value comes
    # with a corresponding (optional) `out` parameter.
    @property
    def nargs(self, /) -> int: ...
    # Equivalent to `len(types)`, within numpy this is at most 24, but for 3rd
    # party ufuncs it could be more.
    @property
    def ntypes(self, /) -> int: ...
    # A list of strings (`LiteralString` can't be used for compatibility
    # reasons), with signatures in terms of `numpy.dtype.char`, that match
    # `r'(\w{nin})->(\w{nout})'`.
    # For instance, `np.frexp` has `['e->ei', 'f->fi', 'd->di', 'g->gi']`.
    # Note that the `len` of each `types` elements is `nin + nout + 2`.
    # Also note that the elements aren't necessarily unique, because the
    # available data types are system dependent.
    if _x.NP2 and not _x.NP20:
        @property
        def types(self, /) -> list[LiteralString]: ...
    else:
        @property
        def types(self, /) -> list[str]: ...

    if _x.NP2 and not _x.NP20:
        # raises `ValueError` i.f.f. `nout != 1 or bool(signature)`
        def at(self, /, *args: Any, **kwargs: Any) -> None: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        def reduce(self, /, *args: Any, **kwargs: Any) -> Any: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        def reduceat(self, /, *args: Any, **kwargs: Any) -> _AnyArray: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        def accumulate(self, /, *args: Any, **kwargs: Any) -> _AnyArray: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        def outer(self, /, *args: Any, **kwargs: Any) -> Any: ...
    else:
        # raises `ValueError` i.f.f. `nout != 1 or bool(signature)`
        @property
        def at(self, /) -> CanCall[..., None] | None: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        @property
        def reduce(self, /) -> CanCall[..., Any] | None: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        @property
        def reduceat(self, /) -> CanCall[..., _AnyArray] | None: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        @property
        def accumulate(self, /) -> CanCall[..., _AnyArray] | None: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        @property
        def outer(self, /) -> CanCall[..., Any] | None: ...


_UFuncMethodCommon: _Type = Literal[
    '__call__',
    'reduce',
    'reduceat',
    'accumulate',
    'outer',
]
if _x.NP2:
    _UFuncMethod: _Type = _UFuncMethodCommon | Literal['at']
else:
    _UFuncMethod: _Type = _UFuncMethodCommon | Literal['inner']


_UFT_contra = TypeVar(
    '_UFT_contra',
    bound=UFunc,
    contravariant=True,
    default=Any,
)
_T_co = TypeVar('_T_co', covariant=True, default=np.ufunc)


@runtime_checkable
class CanArrayUFunc(Protocol[_UFT_contra, _T_co]):
    """
    Interface for ufunc operands.

    See Also:
        - https://numpy.org/devdocs/reference/arrays.classes.html
    """
    def __array_ufunc__(
        self,
        ufunc: _UFT_contra,
        method: _UFuncMethod,
        # /,
        *args: Any,
        **kwargs: Any,
    ) -> _T_co: ...


@runtime_checkable
class CanArrayFunction(Protocol[_FT_contra, _T_co]):
    def __array_function__(
        self,
        /,
        func: _FT_contra,
        # although this could be tighter, this ensures numpy.typing compat
        types: opt.CanIter[opt.CanIterSelf[type[CanArrayFunction[Any, Any]]]],
        # ParamSpec can only be used on *args and **kwargs for some reason...
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> NotImplementedType | _T_co: ...
