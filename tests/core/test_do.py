import optype as op
from optype import _utils
from optype._core import _do, _does


def test_all_public() -> None:
    """Ensure all callables in `optype._do` are in `optype.__all__`."""
    callables_all = _utils.get_callables(op)
    callables_do = _utils.get_callables(_do)

    assert callables_all == callables_do


def test_static() -> None:
    _do_getitem: _does.DoesGetitem = _do.do_getitem
    _do_setitem: _does.DoesSetitem = _do.do_setitem
    _do_delitem: _does.DoesDelitem = _do.do_delitem
    _do_missing: _does.DoesMissing = _do.do_missing
    _do_contains: _does.DoesContains = _do.do_contains

    _do_radd: _does.DoesRAdd = _do.do_radd
    _do_rsub: _does.DoesRSub = _do.do_rsub
    _do_rmul: _does.DoesRMul = _do.do_rmul
    _do_rmatmul: _does.DoesRMatmul = _do.do_rmatmul
    _do_rtruediv: _does.DoesRTruediv = _do.do_rtruediv
    _do_rfloordiv: _does.DoesRFloordiv = _do.do_rfloordiv
    _do_rmod: _does.DoesRMod = _do.do_rmod
    _do_rdivmod: _does.DoesRDivmod = _do.do_rdivmod
    _do_rpow: _does.DoesRPow = _do.do_rpow
    _do_rlshift: _does.DoesRLshift = _do.do_rlshift
    _do_rrshift: _does.DoesRRshift = _do.do_rrshift
    _do_rand: _does.DoesRAnd = _do.do_rand
    _do_rxor: _does.DoesRXor = _do.do_rxor
    _do_ror: _does.DoesROr = _do.do_ror
