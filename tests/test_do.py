import optype as o
from optype._core import _do, _utils


def test_all_public() -> None:
    """Ensure all callables in `optype._do` are in `optype.__all__`."""
    callables_all = _utils.get_callables(o)
    callables_do = _utils.get_callables(_do)

    assert callables_all == callables_do
