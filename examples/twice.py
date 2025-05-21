# %%
from typing import Literal, TypeAlias, TypeVar, assert_type

import optype as op

Y = TypeVar("Y")
Two: TypeAlias = Literal[2]


def twice(x: op.CanRMul[Two, Y], /) -> Y:
    return 2 * x


# %%
assert_type(twice(True), int)
assert_type(twice(1 / 137), float)
assert_type(twice(str(-1 / 12)), str)
assert_type(twice([object()]), list[object])


# %%
def twice2(x: op.CanRMul[Two, Y] | op.CanMul[Two, Y], /) -> Y:
    return 2 * x if isinstance(x, op.CanRMul) else x * 2


# %%
class RMulThing:
    def __rmul__(self, y: Two, /) -> str:
        return f"{y} * _"


assert_type(twice2(RMulThing()), str)
assert twice2(RMulThing()) == "2 * _"


# %%
class MulThing:
    def __mul__(self, y: Two, /) -> str:
        return f"_ * {y}"


assert_type(twice2(MulThing()), str)
assert twice2(MulThing()) == "_ * 2"
