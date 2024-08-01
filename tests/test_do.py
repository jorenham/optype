import optype
import optype._do
from optype._utils import get_callables


def test_all_public() -> None:
    """Ensure all callables in `optype._do` are in `optype.__all__`."""
    callables_all = get_callables(optype)
    callables_do = get_callables(optype._do)

    assert callables_all == callables_do
