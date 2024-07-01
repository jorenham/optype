from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias as _Type

import numpy as np


if sys.version_info >= (3, 13):
    from typing import Protocol, TypeVar, runtime_checkable
else:
    from typing_extensions import Protocol, TypeVar, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Callable as CanCall, Mapping
    from types import NotImplementedType

    from optype import CanIter, CanIterSelf


__all__ = (
    'AnyUFunc',
    'CanArrayFunction',
    'CanArrayUFunc',
)


_NP_V2: Final[bool] = np.__version__.startswith('2.')


_F = TypeVar('_F', infer_variance=True, bound='CanCall[..., Any]', default=Any)
_Nin = TypeVar('_Nin', infer_variance=True, bound=int, default=int)
_Nout = TypeVar('_Nout', infer_variance=True, bound=int, default=int)
_Sig = TypeVar('_Sig', infer_variance=True, bound=str | None, default=Any)
_I = TypeVar(
    '_I',
    infer_variance=True,
    bound=complex | str | bytes | None,  # this includes `bool | int | float`
    default=float | None,
)


@runtime_checkable
class AnyUFunc(Protocol[_F, _Nin, _Nout, _Sig, _I]):
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
    def __call__(self, /) -> _F: ...

    # The number of positional-only parameters, within numpy this is either 1
    # or 2, but might be more for 3rd party ufuncs.
    @property
    def nin(self, /) -> _Nin: ...
    # The number of output values, within numpy this is either 1 or 2.
    @property
    def nout(self, /) -> _Nout: ...
    # A string i.f.f. this is a gufunc (generalized ufunc).
    @property
    def signature(self, /) -> _Sig: ...

    # If `signature is None and nin == 2 and nout == 1`, this *may* be set to
    # a python scalar s.t. `self(x, identity) == x` for all possible `x`.
    # Within numpy==2.0.0, this is only the case for `multiply` (`1`),
    # `logaddexp` (`-inf`), `logaddexp2` (`-inf`), `logical_and` (`True`),
    # and `bitwise_and` (`-1`).
    # Note that the `complex` return annotation implicitly includes
    # `bool | int | float` (these are its supertypes).
    @property
    def identity(self, /) -> _I: ...
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
    @property
    def types(self, /) -> list[str]: ...

    # The following methods are incorrectly typed (`numpy/_typing/_ufunc.pyi`);
    # they should always be methods that `typing.NoReturn` i.f.f. not available
    # for the ufunc's `nin` and `nout` (instead of `None`), so that they'd
    # match the runtime behaviour (i.e. `raise ValueError` when called)

    # raises `ValueError` i.f.f. `nout != 1 or bool(signature)`
    @property
    def at(self, /) -> CanCall[..., None] | None: ...
    # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
    @property
    def reduce(self, /) -> CanCall[..., Any] | None: ...
    # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
    @property
    def reduceat(self, /) -> CanCall[..., np.ndarray[Any, Any]] | None: ...
    # raises `ValueError` i.f.f. `nin != 2 or nout != 1 or bool(signature)`
    @property
    def accumulate(self, /) -> CanCall[..., np.ndarray[Any, Any]] | None: ...
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
if _NP_V2:
    _UFuncMethod: _Type = _UFuncMethodCommon | Literal['at']
else:
    _UFuncMethod: _Type = _UFuncMethodCommon | Literal['inner']


_U = TypeVar('_U', infer_variance=True, bound=AnyUFunc, default=Any)
_R = TypeVar('_R', infer_variance=True, bound=object, default=Any)


@runtime_checkable
class CanArrayUFunc(Protocol[_U, _R]):
    """
    Interface for ufunc operands.

    See Also:
        - https://numpy.org/devdocs/reference/arrays.classes.html
    """
    def __array_ufunc__(
        self,
        ufunc: _U,
        method: _UFuncMethod,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _R: ...


@runtime_checkable
class CanArrayFunction(Protocol[_F, _R]):
    def __array_function__(
        self,
        /,
        func: _F,
        # although this could be tighter, this ensures numpy.typing compat
        types: CanIter[CanIterSelf[type[CanArrayFunction[Any, Any]]]],
        # ParamSpec can only be used on *args and **kwargs for some reason...
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> NotImplementedType | _R: ...
