import sys

from optype import do_iadd


if sys.version_info >= (3, 13):
    from typing import assert_type
else:
    from typing_extensions import assert_type


def test_iadd_iadd():
    class IntIAdd:
        def __init__(self, x: int, /) -> None:
            self.x = x

        def __iadd__(self, y: int, /) -> int:
            self.x += + y
            return self.x

    lhs = IntIAdd(1)
    out = do_iadd(lhs, 2)

    assert_type(out, int)
    assert out == 3


def test_iadd_add():
    class IntAdd:
        def __init__(self, x: int, /) -> None:
            self.x = x

        def __add__(self, y: int, /) -> int:
            return self.x + y

    lhs = IntAdd(1)
    out = do_iadd(lhs, 2)

    assert_type(out, int)
    assert out == 3


def test_iadd_radd():
    class IntRAdd:
        def __init__(self, x: int, /) -> None:
            self.x = x

        def __radd__(self, y: int, /) -> int:
            return self.x + y

    rhs = IntRAdd(1)
    out = do_iadd(2, rhs)

    assert_type(out, int)
    assert out == 3


# the analogous tests for the other inplace binary ops are omitted, because
# the interface implementations are structurally equivalent, and I'm lazy
