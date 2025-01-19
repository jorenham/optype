import optype as op
from optype import _utils
from optype._core import _do


def test_all_public() -> None:
    """Ensure all callables in `optype._do` are in `optype.__all__`."""
    callables_all = _utils.get_callables(op)
    callables_do = _utils.get_callables(_do)

    assert callables_all == callables_do
