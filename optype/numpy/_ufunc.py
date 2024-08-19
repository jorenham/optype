# mypy: disable-error-code="no-any-explicit,no-any-decorated"
from __future__ import annotations

import sys
from collections.abc import Callable as CanCall
from typing import (
    TYPE_CHECKING,
    Literal as L,  # noqa: N817
    TypeAlias as Alias,
)

import numpy as np

import optype.numpy._compat as _x


if sys.version_info >= (3, 13):
    from typing import LiteralString, Protocol, TypeVar, runtime_checkable
else:
    from typing_extensions import LiteralString, Protocol, TypeVar, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import NotImplementedType

    import optype as opt


__all__ = ['CanArrayFunction', 'CanArrayUFunc', 'UFunc']


_FT_co = TypeVar(
    '_FT_co',
    bound=CanCall[..., object],
    covariant=True,
    default=CanCall[..., object],
)
_NInT_co = TypeVar('_NInT_co', bound=int, covariant=True, default=int)
_NoutT_co = TypeVar('_NoutT_co', bound=int, covariant=True, default=int)
_SigT_co = TypeVar(
    '_SigT_co',
    bound=LiteralString | None,
    covariant=True,
    default=LiteralString | None,
)
# numpy < 2.1
_SigT_str_co = TypeVar(
    '_SigT_str_co',
    bound=str | None,
    covariant=True,
    default=str | None,
)
_IdT_co = TypeVar(
    '_IdT_co',
    bound=int | float | complex | bytes | str | None,
    covariant=True,
    default=float | None,
)

_AnyArray: Alias = np.ndarray[tuple[int, ...], np.dtype[np.generic]]

if _x.NP2 and not _x.NP20:
    # `numpy>=2.1`

    @runtime_checkable
    class UFunc(Protocol[_FT_co, _NInT_co, _NoutT_co, _SigT_co, _IdT_co]):
        """
        A generic interface for `numpy.ufunc` "universal function" instances,
        e.g. `numpy.exp`, `numpy.add`, `numpy.frexp`, `numpy.divmod`.

        This also includes gufunc's (generalized universion functions), which
        have a specified `signature`, and aren't necessarily element-wise
        functions (which "regular" ufuncs are).
        At the moment (`numpy>=2.0,<2.2`), the only GUFuncs within numpy are
        `matmul`, and `vecdot`.
        """
        @property
        def __call__(self, /) -> _FT_co: ...

        # The number of positional-only parameters, within numpy this is
        # either 1 or 2, but e.g. `scipy.special.pro_rad2_cv` has 5.
        @property
        def nin(self, /) -> _NInT_co: ...
        # The number of output values, within numpy this is either 1 or 2,
        # but e.g. `scipy.special.ellipj` has 4.
        @property
        def nout(self, /) -> _NoutT_co: ...
        # A string i.f.f. this is a gufunc (generalized ufunc).
        @property
        def signature(self, /) -> _SigT_co: ...

        # If `signature is None and nin == 2 and nout == 1`, this *may* be set
        # to a python scalar s.t. `self(x, identity) == x` for all possible
        # `x`.
        # Within numpy==2.0.0, this is only the case for `multiply` (`1`),
        # `logaddexp` (`-inf`), `logaddexp2` (`-inf`), `logical_and` (`True`),
        # and `bitwise_and` (`-1`).
        # Note that the `complex` return annotation implicitly includes
        # `bool | int | float` (these are its supertypes).
        @property
        def identity(self, /) -> _IdT_co: ...
        # Within numpy this is always `nin + nout`, since each output value
        # comes with a corresponding (optional) `out` parameter.
        @property
        def nargs(self, /) -> int: ...
        # Equivalent to `len(types)`, within numpy this is at most 24, but for
        # 3rd party ufuncs it could be more.
        @property
        def ntypes(self, /) -> int: ...
        # A list of strings (`LiteralString` can't be used for compatibility
        # reasons), with signatures in terms of `numpy.dtype.char`, that match
        # `r'(\w{nin})->(\w{nout})'`.
        # For instance, `np.frexp` has `['e->ei', 'f->fi', 'd->di', 'g->gi']`.
        # Note that the `len` of each `types` elements is `nin + nout + 2`.
        # Also note that the elements aren't necessarily unique, because the
        # available data types are system dependent.
        @property
        def types(self, /) -> list[LiteralString]: ...

        # raises `ValueError` i.f.f. `nout != 1 or bool(signature)`
        @property
        def at(self, /) -> CanCall[..., None]: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        @property
        def reduce(self, /) -> CanCall[..., object]: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        @property
        def reduceat(self, /) -> CanCall[..., _AnyArray]: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        @property
        def accumulate(self, /) -> CanCall[..., _AnyArray]: ...
        # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
        @property
        def outer(self, /) -> CanCall[..., object]: ...

else:
    # `numpy<2.1`

    @runtime_checkable
    class UFunc(Protocol[_FT_co, _NInT_co, _NoutT_co, _SigT_str_co, _IdT_co]):
        """
        A generic interface for `numpy.ufunc` "universal function" instances,
        e.g. `numpy.exp`, `numpy.add`, `numpy.frexp`, `numpy.divmod`.

        This also includes gufunc's (generalized universion functions), which
        have a specified `signature`, and aren't necessarily element-wise
        functions (which "regular" ufuncs are).
        At the moment (`numpy>=2.0,<2.2`), the only GUFuncs within numpy are
        `matmul`, and `vecdot`.
        """
        @property
        def __call__(self, /) -> _FT_co: ...

        @property
        def nin(self, /) -> _NInT_co: ...
        @property
        def nout(self, /) -> _NoutT_co: ...
        @property
        def signature(self, /) -> _SigT_str_co: ...
        @property
        def identity(self, /) -> _IdT_co: ...
        @property
        def nargs(self, /) -> int: ...
        @property
        def ntypes(self, /) -> int: ...
        @property
        def types(self, /) -> list[str]: ...

        # The following *methods* were incorrectly typed prior to NumPy 2.1,
        # which I (@jorenham) fixed: https://github.com/numpy/numpy/pull/26847
        @property
        def at(self, /) -> CanCall[..., None] | None: ...
        @property
        def reduce(self, /) -> CanCall[..., object] | None: ...
        @property
        def reduceat(self, /) -> CanCall[..., _AnyArray] | None: ...
        @property
        def accumulate(self, /) -> CanCall[..., _AnyArray] | None: ...
        @property
        def outer(self, /) -> CanCall[..., object] | None: ...


_MethodCommon: Alias = L['__call__', 'reduce', 'reduceat', 'accumulate', 'outer']
if _x.NP2:  # type: ignore[redundant-expr]
    _Method: Alias = L[_MethodCommon, 'at']
else:
    _Method: Alias = L[_MethodCommon, 'inner']


_UFT_contra = TypeVar('_UFT_contra', bound=UFunc, contravariant=True, default=np.ufunc)
_T_co = TypeVar('_T_co', covariant=True, default=object)


@runtime_checkable
class CanArrayUFunc(Protocol[_UFT_contra, _T_co]):
    """
    Interface for ufunc operands.

    See Also:
        - https://numpy.org/devdocs/reference/arrays.classes.html
    """

    # NOTE: Mypy doesn't understand the Liskov substitution principle when
    # positional-only arguments are involved; so `ufunc` and `method` can't
    # be made positional-only.
    def __array_ufunc__(
        self,
        /,
        ufunc: _UFT_contra,
        method: _Method,
        *args: object,
        **kwargs: object,
    ) -> _T_co: ...


_FT_contra = TypeVar(
    '_FT_contra',
    bound=CanCall[..., object],
    contravariant=True,
    default=CanCall[..., object],
)


@runtime_checkable
class CanArrayFunction(Protocol[_FT_contra, _T_co]):
    def __array_function__(
        self,
        /,
        func: _FT_contra,
        # although this could be tighter, this ensures numpy.typing compat
        types: opt.CanIter[opt.CanIterSelf[type[CanArrayFunction]]],
        # ParamSpec can only be used on *args and **kwargs for some reason...
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> NotImplementedType | _T_co: ...
