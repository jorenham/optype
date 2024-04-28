# pyright: reportMissingTypeStubs=false
import math
from typing import Any

import numba as nb
import numpy as np
from numpy.testing import assert_allclose

from optype.numpy import AnyUfunc, CanArrayUfunc


def _py_beta(x: float, y: float) -> float:
    return math.exp(math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y))


_np_beta = np.frompyfunc(_py_beta, nin=2, nout=1)


def test_anyufunc():
    _ufunc: type[AnyUfunc] = np.ufunc
    assert isinstance(_ufunc, AnyUfunc)

    exp: AnyUfunc = np.exp
    assert isinstance(exp, AnyUfunc)

    add: AnyUfunc = np.add
    assert isinstance(add, AnyUfunc)

    frexp: AnyUfunc = np.frexp
    assert isinstance(frexp, AnyUfunc)

    divmod_: AnyUfunc = np.divmod
    assert isinstance(divmod_, AnyUfunc)

    assert not isinstance(abs, AnyUfunc)
    assert not isinstance(math.exp, AnyUfunc)

    assert not isinstance(_py_beta, AnyUfunc)
    assert isinstance(_np_beta, AnyUfunc)


def test_canarrayufunc():
    quantiles: CanArrayUfunc[np.ufunc, ..., Any] = np.linspace(0, 1, 100)
    assert isinstance(quantiles, CanArrayUfunc)
    assert not isinstance(list(quantiles), CanArrayUfunc)


_nb_beta: AnyUfunc = nb.vectorize([  # type: ignore[numba]
    'f8(i4, i4)',
    'f8(i8, i8)',
    'f8(f4, f4)',
    'f8(f8, f8)',
])(_py_beta)


def test_canarrayufunc_numba():
    assert_allclose(
        _nb_beta([2, 3], [2, 3]),  # type: ignore[numba]
        np.array([_py_beta(2, 2), _py_beta(3, 3)]),
    )
    assert isinstance(_nb_beta, AnyUfunc)
