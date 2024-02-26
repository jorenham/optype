import optype
import optype._do
from .helpers import get_callable_members


def test_all_public():
    """Ensure all callables in `optype._do` are in `optype.__all__`."""
    callables_all = get_callable_members(optype)
    callables_do = get_callable_members(optype._do)

    assert callables_all == callables_do
