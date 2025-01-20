import sys
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeAlias,
    _ProtocolMeta,  # noqa: PLC2701
)

if sys.version_info >= (3, 13):
    from typing import Self, TypeIs, TypeVar, final, override, runtime_checkable
else:
    from typing_extensions import (
        Self,
        TypeIs,
        TypeVar,
        final,
        override,
        runtime_checkable,
    )

from ._can import CanFloat, CanIndex

__all__ = ["Just", "JustBytes", "JustComplex", "JustFloat", "JustInt", "JustObject"]


def __dir__() -> list[str]:
    return __all__


###


_T = TypeVar("_T")
_ObjectT = TypeVar("_ObjectT", default=object)

_ToFloat: TypeAlias = CanFloat | CanIndex


###


class Just(Protocol[_T]):
    """
    An runtime-checkable invariant type "wrapper", where `Just[T]` only accepts
    instances of `T`, and but rejects instances of any strict subtypes of `T`.

    Note that e.g. `Literal[""]` and `LiteralString` are not a strict `str` subtypes,
    and are therefore assignable to `Just[str]`, but instances of `class S(str): ...`
    are **not** assignable to `Just[str]`.

    Warning (`pyright<1.1.390` and `basedpyright<1.22.1`):
        On `pyright<1.1.390` and `basedpyright<1.22.1` this `Just[T]` type does not
        work, due to a bug in the `typeshed` stubs for `object.__class__`, which was
        fixed in https://github.com/python/typeshed/pull/13021.

        However, you could use `JustInt`, `JustFloat`, and `JustComplex` types work
        around this: These already work on `pyright<1.1.390` without problem.

    Warning (`mypy<1.15` and `basedmypy<2.10`):
        On `mypy<1.15` this does not work with promoted types, such as `float` and
        `complex`, which was fixed in https://github.com/python/mypy/pull/18360.
        For other ("unpromoted") types like `Just[int]`, this already works, even
        before the `typeshed` fix above (mypy ignores `@property` setter types and
        overwrites it with the getter's return type).

    Note:
        This protocol is not marked with `@runtime_checkable`, because using
        `isinstance(value, Just)` would be `True` for any type or object, and has no
        semantic meaning.
    """

    @property  # type: ignore[explicit-override]  # mypy bug?
    @override
    def __class__(self, /) -> type[_T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @__class__.setter
    @override
    def __class__(self, t: type[_T], /) -> None: ...


class _JustMeta(_ProtocolMeta, Generic[_ObjectT]):
    # There's nothing wrong with the following parametrized `ClassVar`, and the typing
    # spec should have never disallowed it.
    __just_class__: ClassVar[type[_ObjectT]]  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]

    @override
    def __instancecheck__(self, instance: object) -> TypeIs[_ObjectT]:
        return self.__subclasscheck__(type(instance))

    @override
    def __subclasscheck__(self, subclass: Any) -> TypeIs[type[_ObjectT]]:
        tp = self.__just_class__

        if isinstance(subclass, int):
            # basedmypy "bare" bool and int literals
            subclass = type(subclass)

        if not isinstance(subclass, type):
            # unwrap subscripted generics, with special-casing for `Just[...]`
            from types import GenericAlias  # noqa: I001, PLC0415
            from optype.types._typeforms import GenericType  # noqa: PLC0415

            while type(subclass) in {GenericType, GenericAlias}:
                origin = subclass.__origin__
                if origin is Just:
                    (subclass,) = subclass.__args__
                else:
                    subclass = origin

        if not isinstance(subclass, type):
            raise TypeError("issubclass() arg 1 must be a class") from None

        return subclass is tp


@final
class _JustBytesMeta(_JustMeta[bytes]):
    __just_class__ = bytes


class JustBytes(Just[bytes], Protocol, metaclass=_JustBytesMeta):
    """
    A runtime checkable `Just[bytes]`, that also works on `pyright<1.1.390`.

    Useful as workaround for `mypy`'s `bytes` promotion (which can be disabled with the
    undocumented `--disable-bytearray-promotion` and  `--disable-memoryview-promotion`
    flags). See https://github.com/python/mypy/issues/15313s for more info.
    Note that this workaround requires `mypy >=1.15` or the `--disable-*-promotion`
    flags to work.
    """


@final
class _JustIntMeta(_JustMeta[int]):
    __just_class__ = int


@runtime_checkable
class JustInt(Just[int], Protocol, metaclass=_JustIntMeta):
    """
    A runtime-checkable `Just[int]` that's compatible with `pyright<1.1.390`.

    Example:
        This example shows a situation you want to accept instances of just `int`,
        and reject and other instances, even they're subtypes of `int` such as `bool`,

        ```python
        def f(x: int, /) -> int:
            assert type(x) is int
            return x


        f(1337)  # accepted
        f(True)  # accepted :(
        ```

        But because `bool <: int`, booleans are also assignable to `int`, which we
        don't want to allow.

        Previously, it was considered impossible to achieve this using just python
        typing, and it was told that a PEP would be necessary to make it work.

        But this is not the case at all, and `optype` provides a clean counter-example
        that works with pyright and (even) mypy:

        ```python
        import optype as op


        def f_fixed(x: op.JustInt, /) -> int:
            assert type(x) is int
            return x  # accepted


        f_fixed(1337)  # accepted
        f_fixed(True)  # rejected :)
        ```

    Note:
        On `mypy` this behaves in the same way as `Just[int]`, which accidentally
        already works because of a mypy bug.
    """

    # workaround for `pyright<1.1.390` and `basedpyright<1.22.1`
    def __new__(cls, x: str | bytes | bytearray, /, base: CanIndex) -> Self: ...


@final
class _JustFloatMeta(_JustMeta[float]):
    __just_class__ = float


class JustFloat(Just[float], Protocol, metaclass=_JustFloatMeta):
    """
    A runtime checkable `Just[float]` that also works on `pyright<1.1.390`.

    Warning:
        Unlike `JustInt`, this does not work on `mypy<1.41.2`.
    """

    # workaround for `pyright<1.1.390` and `basedpyright<1.22.1`
    def hex(self, /) -> str: ...


@final
class _JustComplexMeta(_JustMeta[complex]):
    __just_class__ = complex


class JustComplex(Just[complex], Protocol, metaclass=_JustComplexMeta):
    """
    A runtime checkable `Just[complex]`, that also works on `pyright<1.1.390`.

    Warning:
        Unlike `JustInt`, this does not work on `mypy<1.41.2`.
    """

    # workaround for `pyright<1.1.390` and `basedpyright<1.22.1`
    def __new__(cls, /, real: _ToFloat, imag: _ToFloat) -> Self: ...


@final
class _JustObjectMeta(_JustMeta[object]):
    __just_class__ = object


class JustObject(Just[object], Protocol, metaclass=_JustObjectMeta):
    """
    A runtime checkable `Just[object]`, that also works on `pyright<1.1.390`.

    Useful for typing `object()` sentinels.
    """
