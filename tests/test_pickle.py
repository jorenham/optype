import pytest

import optype as op
from optype.inspect import get_protocol_members, is_runtime_protocol


@pytest.mark.parametrize(
    "cls",
    [getattr(op.pickle, k) for k in op.pickle.__all__],
)
def test_protocols(cls: type) -> None:
    # ensure correct name
    assert cls.__module__ == "optype.pickle"
    assert cls.__name__ == cls.__qualname__
    assert cls.__name__.startswith("Can")

    # ensure exported
    assert cls.__name__ in op.pickle.__all__

    # ensure single-method protocols
    assert len(get_protocol_members(cls) - {"__new__"}) == 1

    # ensure @runtime_checkable
    assert is_runtime_protocol(cls)
