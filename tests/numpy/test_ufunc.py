# pyright: reportMissingTypeStubs=false
import math

import numpy as np

from optype.numpy import AnyUfunc, CanArrayUFunc


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
    quantiles: CanArrayUFunc = np.linspace(0, 1, 100)
    assert isinstance(quantiles, CanArrayUFunc)
    assert not isinstance(list(quantiles), CanArrayUFunc)
