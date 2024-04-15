# %%
import sys
from typing import (
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    final,
)
if sys.version_info < (3, 12):
    from typing_extensions import TypeVarTuple, Unpack, assert_type
else:
    from typing import TypeVarTuple, Unpack, assert_type

from optype import CanMul, CanRMul

Y = TypeVar('Y')
Two: TypeAlias = Literal[2]


def twice(x: CanRMul[Two, Y], /) -> Y:
    return 2 * x


# %%
assert_type(twice(True), int)
assert_type(twice(1 / 137), float)
assert_type(twice(str(-1 / 12)), str)
assert_type(twice([object()]), list[object])


# %%
Ts = TypeVarTuple('Ts')


@final
class RMulArgs(Generic[Unpack[Ts]]):
    def __init__(self, *args: Unpack[Ts]) -> None:
        self.args = args

    def __rmul__(self, y: Two, /) -> 'RMulArgs[Unpack[Ts], Unpack[Ts]]':
        if y != 2:
            return NotImplemented
        return RMulArgs(*self.args, *self.args)


assert_type(twice(RMulArgs(42, True)), RMulArgs[int, bool, int, bool])


# %%
def twice2(x: CanRMul[Two, Y] | CanMul[Two, Y], /) -> Y:
    return 2 * x if isinstance(x, CanRMul) else x * 2


# %%
class RMulThing:
    def __rmul__(self, y: Two, /) -> str:
        return f'{y} * _'


assert_type(twice2(RMulThing()), str)
assert twice2(RMulThing()) == '2 * _'


# %%
class MulThing:
    def __mul__(self, y: Two, /) -> str:
        return f'_ * {y}'


assert_type(twice2(MulThing()), str)
assert twice2(MulThing()) == '_ * 2'
