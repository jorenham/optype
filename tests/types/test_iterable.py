# ruff: noqa: F841
import optype as opt


def test_iterable():
    itr = (i for i in range(42))

    itr_any: opt.types.AnyIterable = itr
    itr_int: opt.types.AnyIterable[int] = itr
    itr_str: opt.types.AnyIterable[str] = itr  # pyright: ignore[reportAssignmentType]

    assert opt.types.is_iterable(itr)


class SRange:
    def __init__(self, stop: int, /) -> None:
        self._stop = stop

    def __getitem__(self, index: int, /) -> int:
        if index < 0:
            index += self._stop
        if index < 0 or index >= self._stop:
            raise IndexError
        return index


def test_sequence():
    seq = (i for i in range(42))

    itr_any: opt.types.AnyIterable = seq
    itr_int: opt.types.AnyIterable[int] = seq
    itr_str: opt.types.AnyIterable[str] = seq  # pyright: ignore[reportAssignmentType]

    assert opt.types.is_iterable(seq)


def test_dict():
    obj = {42: 'spam'}

    itr_any: opt.types.AnyIterable = obj
    itr_key: opt.types.AnyIterable[int] = obj
    itr_item: opt.types.AnyIterable[tuple[int, str]] = obj  # pyright: ignore[reportAssignmentType]
    # unfortunately this in unavoidable; type unions have no order
    itr_value: opt.types.AnyIterable[str] = obj

    assert opt.types.is_iterable(obj)


def test_not_interable():
    obj_object: opt.types.AnyIterable = object  # pyright: ignore[reportAssignmentType]
    assert not opt.types.is_iterable(object())

    obj_int: opt.types.AnyIterable = 42  # pyright: ignore[reportAssignmentType]
    assert not opt.types.is_iterable(42)

    cls_list: opt.types.AnyIterable = list  # pyright: ignore[reportAssignmentType]
    assert not opt.types.is_iterable(list)

    cls_seq: opt.types.AnyIterable = SRange  # pyright: ignore[reportAssignmentType]
    assert not opt.types.is_iterable(SRange)
