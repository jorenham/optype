import math
import sys
from collections.abc import Callable as Fn
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np

from optype.numpy import CanArrayUFunc, UFunc


if sys.version_info >= (3, 13):
    from typing import LiteralString
else:
    from typing_extensions import LiteralString


_ST__ScalarOrArray = TypeVar('_ST__ScalarOrArray', bound=np.generic)
_ScalarOrArray = (
    np.ndarray[Any, np.dtype[_ST__ScalarOrArray]]
    | _ST__ScalarOrArray
)

_N0: TypeAlias = Literal[0]
_N1: TypeAlias = Literal[1]
_N2: TypeAlias = Literal[2]

_Any2: TypeAlias = tuple[Any, Any]


def test_anyufunc_ufunc_type():
    tp: type[UFunc] = np.ufunc

    tp_1_any: type[UFunc[Any]] = np.ufunc
    tp_1_fn: type[UFunc[Fn[..., Any]]] = np.ufunc

    tp_2_any: type[UFunc[Any, Any]] = np.ufunc
    tp_2_int: type[UFunc[Any, int]] = np.ufunc
    # purposefully wrong
    tp_2_one: type[UFunc[Any, _N1]] = np.ufunc  # pyright: ignore[reportAssignmentType]

    tp_3_any: type[UFunc[Any, Any, Any]] = np.ufunc
    tp_3_int: type[UFunc[Any, Any, int]] = np.ufunc
    # purposefully wrong
    tp_3_one: type[UFunc[Any, Any, _N1]] = np.ufunc  # pyright: ignore[reportAssignmentType]

    tp_4_any: type[UFunc[Any, Any, Any, Any]] = np.ufunc
    tp_4_bound: type[UFunc[Any, Any, Any, str | None]] = np.ufunc
    # purposefully wrong
    tp_4_none: type[UFunc[Any, Any, Any, None]] = np.ufunc  # pyright: ignore[reportAssignmentType]
    # purposefully wrong
    tp_4_str: type[UFunc[Any, Any, Any, str]] = np.ufunc  # pyright: ignore[reportAssignmentType]

    assert isinstance(tp, UFunc)


def test_anyufunc_ufunc_11():
    """Tests compatibility with `numpy._typing._func._UFunc_Nin1_Nout1`."""
    fn = np.exp
    assert isinstance(fn, np.ufunc)
    assert fn.nin == 1
    assert fn.nout == 1
    assert fn.nargs == 2
    assert fn.signature is None
    assert fn.identity is None

    # accepts either 1 or 2 positional-only arguments
    fn_1: UFunc[Fn[[Any], Any]] = fn
    fn_2: UFunc[Fn[[Any, Any], Any]] = fn

    # all type params
    fn_all: UFunc[
        Fn[..., _ScalarOrArray[np.inexact[Any] | np.object_]],  # __call__
        _N1,  # nin
        _N1,  # nout
        None,  # signature
        None,  # identity
    ] = fn

    # wrong signatures
    fn_wrong_nargs_0: UFunc[Fn[[], Any]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nargs_3: UFunc[Fn[[Any, Any, Any], Any]] = fn  # pyright: ignore[reportAssignmentType]
    # `_UFunc_Nin1_Nout1.__call__` return type is `Any`, i.e. useless.

    # wrong nin/nout
    fn_wrong_nin: UFunc[Any, _N2] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nout: UFunc[Any, Any, _N2] = fn  # pyright: ignore[reportAssignmentType]

    # wrong (i.e. a defined) signature / identity
    fn_wrong_sig: UFunc[Any, Any, Any, str] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_id: UFunc[Any, Any, Any, Any, float] = fn  # pyright: ignore[reportAssignmentType]

    assert isinstance(fn, UFunc)


def test_anyufunc_ufunc_21():
    """Tests compatibility with `numpy._typing._func._UFunc_Nin2_Nout1`."""
    fn = np.add
    assert isinstance(fn, np.ufunc)
    assert fn.nin == 2
    assert fn.nout == 1
    assert fn.nargs == 3
    assert fn.signature is None
    assert fn.identity == 0

    # accepts either 2 or 3 positional-only arguments
    fn_2: UFunc[Fn[[Any, Any], Any]] = fn
    fn_3: UFunc[Fn[[Any, Any, Any], Any]] = fn

    # all type params
    fn_all: UFunc[
        Fn[[Any, Any], _ScalarOrArray[np.generic]],  # __call__
        _N2,  # nin
        _N1,  # nout
        None,  # signature
        _N0,  # identity (incorrect in np2; consider `np.add('spam', 0)`; QED)
    ] = fn

    # wrong signatures
    fn_wrong_nargs_0: UFunc[Fn[[], Any]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nargs_4: UFunc[Fn[[Any, Any, Any, Any], Any]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nargs_1: UFunc[Fn[[Any], Any]] = fn  # pyright: ignore[reportAssignmentType]
    # `_UFunc_Nin2_Nout1.__call__` return type is `Any`, i.e. useless.

    # wrong nin/nout
    fn_wrong_nin: UFunc[Any, _N1] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nout: UFunc[Any, Any, _N2] = fn  # pyright: ignore[reportAssignmentType]

    # wrong (i.e. a defined) signature / identity
    fn_wrong_sig: UFunc[Any, Any, Any, str] = fn  # pyright: ignore[reportAssignmentType]
    fn_missing_id: UFunc[Any, Any, Any, Any, None] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_id: UFunc[Any, Any, Any, Any, _N1] = fn  # pyright: ignore[reportAssignmentType]

    assert isinstance(fn, UFunc)


def test_anyufunc_ufunc_12():
    """Tests compatibility with `numpy._typing._func._UFunc_Nin1_Nout2`."""
    fn = np.frexp
    assert isinstance(fn, np.ufunc)
    assert fn.nin == 1
    assert fn.nout == 2
    assert fn.nargs == 3
    assert fn.signature is None
    assert fn.identity is None

    # accepts either 1, 2, or 3 positional-only arguments
    fn_1: UFunc[Fn[[Any], _Any2]] = fn
    fn_2: UFunc[Fn[[Any, Any], _Any2]] = fn
    fn_3: UFunc[Fn[[Any, Any, Any], _Any2]] = fn

    # all type params
    fn_all: UFunc[Fn[..., _Any2], _N1, _N2, None, None] = fn

    # wrong signatures
    fn_wrong_nargs_0: UFunc[Fn[[], _Any2]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nargs_4: UFunc[Fn[[Any, Any, Any, Any], _Any2]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nret_1: UFunc[Fn[..., tuple[Any]]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nret_3: UFunc[Fn[..., tuple[Any, Any, Any]]] = fn  # pyright: ignore[reportAssignmentType]

    # wrong nin/nout
    fn_wrong_nin: UFunc[Any, _N2] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nout: UFunc[Any, Any, _N1] = fn  # pyright: ignore[reportAssignmentType]

    # wrong (i.e. a defined) signature / identity
    fn_wrong_sig: UFunc[Any, Any, Any, str] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_id: UFunc[Any, Any, Any, Any, float] = fn  # pyright: ignore[reportAssignmentType]

    assert isinstance(fn, UFunc)


def test_anyufunc_ufunc22():
    """Tests compatibility with `numpy._typing._func._UFunc_Nin2_Nout2`."""
    fn = np.divmod
    assert isinstance(fn, np.ufunc)
    assert fn.nin == 2
    assert fn.nout == 2
    assert fn.nargs == 4
    assert fn.signature is None
    assert fn.identity is None

    # accepts either 2, 3, or 4 positional-only arguments
    fn_2: UFunc[Fn[[Any, Any], _Any2]] = fn
    fn_3: UFunc[Fn[[Any, Any, Any], _Any2]] = fn
    fn_4: UFunc[Fn[[Any, Any, Any, Any], _Any2]] = fn

    # all type params
    fn_all: UFunc[Fn[..., _Any2], _N2, _N2, None, None] = fn

    # wrong signatures
    fn_wrong_nargs_0: UFunc[Fn[[], _Any2]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nargs_1: UFunc[Fn[[Any], _Any2]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nargs_5: UFunc[Fn[[Any, Any, Any, Any, Any], _Any2]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nret_1: UFunc[Fn[..., tuple[Any]]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nret_3: UFunc[Fn[..., tuple[Any, Any, Any]]] = fn  # pyright: ignore[reportAssignmentType]

    # wrong nin/nout
    fn_wrong_nin: UFunc[Any, _N1] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nout: UFunc[Any, Any, _N1] = fn  # pyright: ignore[reportAssignmentType]

    # wrong (i.e. a defined) signature / identity
    fn_wrong_sig: UFunc[Any, Any, Any, str] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_id: UFunc[Any, Any, Any, Any, float] = fn  # pyright: ignore[reportAssignmentType]

    assert isinstance(fn, UFunc)


def test_anyufunc_gufunc21():
    """Tests compatibility with `numpy._typing._func._GUFunc_Nin2_Nout1`."""
    fn = np.matmul
    assert fn.nin == 2
    assert fn.nout == 1
    assert fn.nargs == 3
    assert fn.signature is not None
    assert fn.identity is None

    # accepts either 2 or 3 positional-only arguments
    fn_2: UFunc[Fn[[Any, Any], Any]] = fn
    fn_3: UFunc[Fn[[Any, Any, Any], Any]] = fn

    # all type params
    fn_all: UFunc[
        Fn[[Any, Any], _ScalarOrArray[np.generic]],  # __call__
        _N2,  # nin
        _N1,  # nout
        LiteralString,  # signature
        None,  # identity (why isn't this `1`?)
    ] = fn

    # wrong signatures
    fn_wrong_nargs_0: UFunc[Fn[[], Any]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nargs_1: UFunc[Fn[[Any], Any]] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nargs_4: UFunc[Fn[[Any, Any, Any, Any], Any]] = fn  # pyright: ignore[reportAssignmentType]
    # `_UFunc_Nin2_Nout1.__call__` return type is `Any`, i.e. useless.

    # wrong nin/nout
    fn_wrong_nin: UFunc[Any, _N1] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_nout: UFunc[Any, Any, _N2] = fn  # pyright: ignore[reportAssignmentType]

    # wrong (i.e. a defined) signature / identity
    fn_wrong_sig: UFunc[Any, Any, Any, None] = fn  # pyright: ignore[reportAssignmentType]
    fn_wrong_id: UFunc[Any, Any, Any, Any, _N1] = fn  # pyright: ignore[reportAssignmentType]

    assert isinstance(fn, np.ufunc)
    assert isinstance(fn, UFunc)


def beta_py(x: float, y: float, /) -> float:
    """Beta function."""
    return math.exp(math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y))


def test_anyufunc_custom_py():
    # purposefully wrong
    fn = beta_py
    assert isinstance(fn, Fn)

    not_a_ufunc: UFunc = fn  # pyright: ignore[reportAssignmentType]

    assert not isinstance(fn, np.ufunc)
    assert not isinstance(fn, UFunc)


beta_np = np.frompyfunc(beta_py, nin=2, nout=1)


def test_anyufunc_custom_np():
    fn = beta_np
    assert fn.nin == 2
    assert fn.nout == 1
    assert fn.nargs == 3
    assert fn.signature is None
    assert fn.identity is None

    fn_any: UFunc = fn

    fn_1: UFunc[Fn[[float, float], float]] = fn
    # unfortunately, `frompyfunc` ignores the passed `nin` and `nout` types
    # within the returned ufunc type (within numpy's own annotations)
    # fn_3:: AnyUFunc[Any, _N2, _N1] = beta_np
    fn_3: UFunc[Any, int, int] = fn

    # similarly, the fact that `signature = None` isn't annotated correctly
    # fn_4:: AnyUFunc[Any, Any, Any, None] = beta_np
    fn_4: UFunc[Any, Any, Any, str | None] = fn

    # the `identity` kwarg from `frompyfunc` is also ignored :(
    # fn_4:: AnyUFunc[Any, Any, Any, Any, None] = beta_np
    fn_5: UFunc[Any, Any, Any, Any, None] = fn

    assert isinstance(fn, UFunc)


def test_canarrayufunc():
    quantiles: CanArrayUFunc = np.linspace(0, 1, 100)
    assert isinstance(quantiles, CanArrayUFunc)
    assert not isinstance(list(quantiles), CanArrayUFunc)
