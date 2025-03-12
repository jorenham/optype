import pytest

import optype as op


# fmt: off
class A: ...
class B(A): ...  # noqa: E302
class C(B): ...  # noqa: E302
# fmt: on


def test_just_custom() -> None:
    a, b, c = A(), B(), C()

    b_b: op.Just[B] = b
    b_a: op.Just[B] = a  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    b_c: op.Just[B] = c  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_just_object() -> None:
    tn_object1: op.Just[object] = object()
    tn_object2: op.JustObject = object()
    tp_custom1: op.Just[object] = A()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_custom2: op.JustObject = A()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    tn_object_type1: type[op.Just[object]] = object
    tn_object_type2: type[op.JustObject] = object
    tp_custom_type1: type[op.Just[object]] = A  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_custom_type2: type[op.JustObject] = A  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_just_int() -> None:
    # instance assignment: true negatives
    tn_int: op.JustInt = int("42")
    tn_int_literal: op.JustInt = 42

    # instance assignment: true positives
    tp_bool: op.JustInt = bool([])  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_true: op.JustInt = True  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # type assignment
    tn_int_type: type[op.JustInt] = int
    tp_int_type: type[op.JustInt] = bool  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # docstring example
    def f(x: op.JustInt, /) -> int:
        assert type(x) is int
        return x

    def g() -> None:  # pyright: ignore[reportUnusedFunction]
        f(1337)  # accepted
        f(True)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("just_cls", "cls"),
    [
        (op.JustBytes, bytes),
        (op.JustInt, int),
        (op.JustFloat, float),
        (op.JustComplex, complex),
        (op.JustObject, object),
    ],
)
def test_just_sub_meta(just_cls: type, cls: type) -> None:
    assert isinstance(cls(), just_cls)
    assert not isinstance(bool(), just_cls)  # noqa: UP018
    assert not isinstance(cls, just_cls)

    assert issubclass(cls, just_cls)
    assert not issubclass(bool, just_cls)
    assert not issubclass(type, just_cls)

    assert issubclass(op.Just[cls], just_cls)  # type: ignore[valid-type]
    assert not issubclass(op.Just[bool], just_cls)
    assert not issubclass(op.Just[type], just_cls)


def test_just_float() -> None:
    tn_float: op.JustFloat = float("inf")
    tp_int: op.JustFloat = int("42")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_int_literal: op.JustFloat = 42  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_bool: op.JustFloat = bool([])  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_bool_literal: op.JustFloat = True  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    tn_float_type: type[op.JustFloat] = float
    tp_str_type: type[op.JustFloat] = str  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_int_type: type[op.JustFloat] = int  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_bool_type: type[op.JustFloat] = bool  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_just_complex() -> None:
    tn_complex: op.JustComplex = complex("inf")
    tp_float: op.JustComplex = float("inf")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    tn_complex_type: type[op.JustComplex] = complex
    tp_str_type: type[op.JustComplex] = str  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_float_type: type[op.JustComplex] = float  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_just_bytes() -> None:
    tn_bytes: op.JustBytes = b"yes"
    tp_str: op.JustBytes = "no"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_bytearray: op.JustBytes = bytearray(b"no")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_memoryview: op.JustBytes = memoryview(b"no")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    tn_bytes_type: type[op.JustBytes] = bytes
    tp_str_type: type[op.JustBytes] = str  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_bytearray_type: type[op.JustBytes] = bytearray  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_memoryview_type: type[op.JustBytes] = memoryview  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
