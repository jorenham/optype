import optype as op


class A: ...


class B(A): ...


class C(B): ...


def test_just_custom() -> None:
    a, b, c = A(), B(), C()

    b_b: op.Just[B] = b
    b_a: op.Just[B] = a  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    b_c: op.Just[B] = c  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_just_object() -> None:
    tn_object: op.Just[object] = object()
    tp_custom: op.Just[object] = A()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    tn_object_type: type[op.Just[object]] = object
    tp_custom_type: type[op.Just[object]] = A  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


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


def test_just_float() -> None:
    # NOTE: this currently doesn't work on mypy==1.14.1, but the next release fixes this
    tn_float: op.JustFloat = float("inf")
    tp_int: op.JustFloat = int("42")  # pyright: ignore[reportAssignmentType]
    tp_int_literal: op.JustFloat = 42  # pyright: ignore[reportAssignmentType]
    tp_bool: op.JustFloat = bool([])  # pyright: ignore[reportAssignmentType]
    tp_bool_literal: op.JustFloat = True  # pyright: ignore[reportAssignmentType]

    tn_float_type: type[op.JustFloat] = float
    tp_str_type: type[op.JustFloat] = str  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_int_type: type[op.JustFloat] = int  # pyright: ignore[reportAssignmentType]
    tp_bool_type: type[op.JustFloat] = bool  # pyright: ignore[reportAssignmentType]


def test_just_complex() -> None:
    # NOTE: this currently doesn't work on mypy==1.14.1, but the next release fixes this
    tn_complex: op.JustComplex = complex("inf")
    tp_float: op.JustComplex = float("inf")  # pyright: ignore[reportAssignmentType]

    tn_complex_type: type[op.JustComplex] = complex
    tp_str_type: type[op.JustComplex] = str  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    tp_float_type: type[op.JustComplex] = float  # pyright: ignore[reportAssignmentType]
