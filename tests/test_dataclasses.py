"""Test module for optype.dataclasses protocols."""

import dataclasses
from typing import Any, ClassVar, cast

import optype as op
from optype.inspect import is_runtime_protocol


def test_has_dataclass_fields_runtime_checkable() -> None:
    """Ensure that HasDataclassFields is @runtime_checkable."""
    assert is_runtime_protocol(op.dataclasses.HasDataclassFields)


@dataclasses.dataclass
class Point:
    """A simple dataclass for testing."""

    x: float
    y: float


@dataclasses.dataclass(frozen=True)
class ImmutablePoint:
    """A frozen dataclass for testing."""

    x: float
    y: float


class FakeDataclass:
    """A class with __dataclass_fields__ attribute but not a real dataclass."""

    __dataclass_fields__: ClassVar[dict[str, Any]] = {}


class NotADataclass:
    """A regular class without __dataclass_fields__."""


def test_has_dataclass_fields_issubclass_real_dataclass() -> None:
    """Test issubclass with real dataclasses."""
    assert issubclass(  # type: ignore[misc]
        Point,
        op.dataclasses.HasDataclassFields,  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[invalid-argument, unsafe-overlap]
    )
    assert issubclass(  # type: ignore[misc]
        ImmutablePoint,
        op.dataclasses.HasDataclassFields,  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[invalid-argument, unsafe-overlap]
    )


def test_has_dataclass_fields_issubclass_fake_dataclass() -> None:
    """Test issubclass with a fake dataclass that has __dataclass_fields__."""
    assert issubclass(  # type: ignore[misc]
        FakeDataclass,
        op.dataclasses.HasDataclassFields,  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[invalid-argument]
    )


def test_has_dataclass_fields_issubclass_not_dataclass() -> None:
    """Test issubclass with a regular class without __dataclass_fields__."""
    assert not issubclass(  # type: ignore[misc]
        NotADataclass,
        op.dataclasses.HasDataclassFields,  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[invalid-argument]
    )


def test_has_dataclass_fields_isinstance_real_dataclass() -> None:
    """Test isinstance with real dataclass instances."""
    point = Point(x=1.0, y=2.0)
    immutable_point = ImmutablePoint(x=3.0, y=4.0)

    # pyrefly: ignore[unsafe-overlap]
    assert isinstance(point, op.dataclasses.HasDataclassFields)
    # pyrefly: ignore[unsafe-overlap]
    assert isinstance(immutable_point, op.dataclasses.HasDataclassFields)


def test_has_dataclass_fields_isinstance_fake_dataclass() -> None:
    """Test isinstance with a fake dataclass instance."""
    fake = FakeDataclass()
    assert isinstance(fake, op.dataclasses.HasDataclassFields)


def test_has_dataclass_fields_isinstance_not_dataclass() -> None:
    """Test isinstance with a regular object."""
    obj = NotADataclass()
    assert not isinstance(obj, op.dataclasses.HasDataclassFields)


def test_has_dataclass_fields_isinstance_builtin_types() -> None:
    """Test isinstance with builtin types."""
    assert not isinstance(42, op.dataclasses.HasDataclassFields)
    assert not isinstance("string", op.dataclasses.HasDataclassFields)
    assert not isinstance([], op.dataclasses.HasDataclassFields)
    assert not isinstance({}, op.dataclasses.HasDataclassFields)


def test_has_dataclass_fields_protocol_conformance() -> None:
    """Test basic protocol conformance for HasDataclassFields."""
    point_fields = Point.__dataclass_fields__
    assert isinstance(point_fields, dict)

    # Check that the class structurally matches the protocol
    assert isinstance(
        Point,
        op.dataclasses.HasDataclassFields,  # pyrefly: ignore[unsafe-overlap]
    )

    # This should type check correctly as a HasDataclassFields type
    protocol_type = cast("type[op.dataclasses.HasDataclassFields]", Point)
    assert issubclass(
        protocol_type,
        op.dataclasses.HasDataclassFields,  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[invalid-argument]
    )
