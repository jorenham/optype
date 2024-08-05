import optype as opt
from optype import _do  # pyright: ignore[reportPrivateUsage]
from optype._utils import get_callables


def test_all_public() -> None:
    """Ensure all callables in `optype._do` are in `optype.__all__`."""
    callables_all = get_callables(opt)
    callables_do = get_callables(_do)

    assert callables_all == callables_do
