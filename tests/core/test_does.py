from typing import assert_type, final

import optype as op


def test_iadd_iadd() -> None:
    @final
    class IntIAdd:
        def __init__(self, x: int, /) -> None:
            self.x = x

        def __iadd__(self, y: int, /) -> int:
            self.x += +y
            return self.x

    lhs = IntIAdd(1)
    out = op.do_iadd(lhs, 2)

    assert_type(out, int)
    assert out == 3


def test_iadd_add() -> None:
    @final
    class IntAdd:
        def __init__(self, x: int, /) -> None:
            self.x = x

        def __add__(self, y: int, /) -> int:
            return self.x + y

    lhs = IntAdd(1)
    out = op.do_iadd(lhs, 2)

    assert_type(out, int)
    assert out == 3


def test_iadd_radd() -> None:
    @final
    class IntRAdd:
        def __init__(self, x: int, /) -> None:
            self.x = x

        def __radd__(self, y: int, /) -> int:
            return self.x + y

    rhs = IntRAdd(1)
    out = op.do_iadd(2, rhs)

    assert_type(out, int)
    assert out == 3


# the analogous tests for the other inplace binary ops are omitted, because
# the interface implementations are structurally equivalent, and I'm lazy


def test_round() -> None:
    x = 1.45

    r = op.do_round(x)
    assert_type(r, int)
    assert isinstance(r, int)
    assert r == 1

    r_none = op.do_round(x, None)
    assert_type(r_none, int)
    assert isinstance(r_none, int)
    assert r == 1

    r_1 = op.do_round(x, 1)
    assert_type(r_1, float)
    assert isinstance(r_1, float)
