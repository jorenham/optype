import typing
import optype

Two: typing.TypeAlias = typing.Literal[2]
Y = typing.TypeVar('Y')


def twice(x: optype.CanRMul[Two, Y], /) -> Y:
    return 2 * x


typing.assert_type(twice(True), int)
typing.assert_type(twice(1 / 137), float)
typing.assert_type(twice(str(-1 / 12)), str)
typing.assert_type(twice([object()]), list[object])


Ts = typing.TypeVarTuple('Ts')


@typing.final
class RMulArgs(typing.Generic[*Ts]):
    def __init__(self, *args: *Ts) -> None:
        self.args = args

    def __rmul__(self, y: Two, /) -> 'RMulArgs[*Ts, *Ts]':
        if y != 2:
            return NotImplemented
        return RMulArgs(*self.args, *self.args)


typing.assert_type(twice(RMulArgs(42, True)), RMulArgs[int, bool, int, bool])

###


def twice2(x: optype.CanRMul[Two, Y] | optype.CanMul[Two, Y], /) -> Y:
    return 2 * x if isinstance(x, optype.CanRMul) else x * 2


class RMulThing:
    def __rmul__(self, y: Two, /) -> str:
        return f'{y} * _'


class MulThing:
    def __mul__(self, y: Two, /) -> str:
        return f'_ * {y}'


typing.assert_type(twice2(RMulThing()), str)
assert twice2(RMulThing()) == '2 * _'

typing.assert_type(twice2(MulThing()), str)
assert twice2(MulThing()) == '_ * 2'
