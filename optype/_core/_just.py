import sys
from types import GenericAlias
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    _ProtocolMeta,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    cast,
)

from optype.types._typeforms import GenericType


if sys.version_info >= (3, 13):
    from typing import Self, TypeIs, TypeVar, override, runtime_checkable
else:
    from typing_extensions import Self, TypeIs, TypeVar, override, runtime_checkable


from ._can import CanFloat, CanIndex


__all__ = ["Just", "JustComplex", "JustFloat", "JustInt"]


def __dir__() -> list[str]:
    return __all__


###


_T = TypeVar("_T")
_ObjectT = TypeVar("_ObjectT", default=object)

_ToFloat: TypeAlias = CanFloat | CanIndex


###


class _JustMeta(_ProtocolMeta, Generic[_ObjectT]):
    def __just_type__(self, /) -> type[_ObjectT]:  # noqa: PLW3201
        raise NotImplementedError

    @override
    def __instancecheck__(self, instance: object) -> TypeIs[_ObjectT]:
        return self.__subclasscheck__(type(instance))

    @override
    def __subclasscheck__(self, subclass: type) -> TypeIs[type[_ObjectT]]:
        try:
            tp = self.__just_type__()
        except NotImplementedError:
            pass
        else:
            return subclass is tp

        return super().__subclasscheck__(subclass)


class _JustAlias(GenericType, Generic[_ObjectT], _root=True):
    @override
    def __instancecheck__(self, instance: object) -> TypeIs[_ObjectT]:
        return self.__subclasscheck__(type(instance))

    @override
    def __subclasscheck__(self, subclass: type) -> TypeIs[type[_ObjectT]]:
        assert len(self.__args__) == 1
        arg: _JustMeta | _JustAlias | type = self.__args__[0]
        if isinstance(arg, _JustMeta):
            arg_: _JustMeta = arg
            return arg_.__subclasscheck__(subclass)
        if isinstance(arg, _JustAlias):
            return arg.__subclasscheck__(subclass)
        return subclass is arg

    @override
    def __repr__(self) -> str:
        return super().__repr__().replace("typing.", "optype.")

    @classmethod
    def __class_getitem__(cls, arg: type, /) -> GenericAlias:
        # not sure why, but subscripting crashes without this
        return GenericAlias(cls, arg)


@runtime_checkable
class Just(Protocol[_T]):
    """
    An runtime-checkable invariant type "wrapper", where `Just[T]` only accepts
    instances of `T`, and but rejects instances of any strict subtypes of `T`.

    Note that e.g. `Literal[""]` and `LiteralString` are not a strict `str` subtypes,
    and are therefore assignable to `Just[str]`, but instances of `class S(str): ...`
    are **not** assignable to `Just[str]`.

    Warning (`pyright<1.390` and `basedpyright<1.22.1`):
        On `pyright<1.390` and `basedpyright<1.22.1` this `Just[T]` type does not work,
        due to a bug in the `typeshed` stubs for `object.__class__`, which was fixed in
        https://github.com/python/typeshed/pull/13021.

        However, you could use `JustInt`, `JustFloat`, and `JustComplex` types work
        around this: These already work on `pyright<1.390` without problem.

    Warning (`mypy<1.14.2` and `basedmypy<2.9.2`):
        On `mypy<1.41.2` this does not work with promoted types, such as `float` and
        `complex`, which was fixed in https://github.com/python/mypy/pull/18360.
        For other ("unpromoted") types like `Just[int]`, this already works, even
        before the `typeshed` fix above (mypy ignores `@property` setter types and
        overwrites it with the getter's return type).

    Note:
        This, and the other `Just` protocols, are not `@runtime_checkable`, because
        using `isinstance` would then be `True` for any type or object, which is
        maximally useless.
    """

    @property  # type: ignore[explicit-override]  # mypy bug?
    @override
    def __class__(self, /) -> type[_T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @__class__.setter
    @override
    def __class__(self, t: type[_T], /) -> None: ...

    @classmethod
    def __class_getitem__(cls, arg: type[_JustAlias], /) -> _JustAlias[_JustAlias]:
        generic_alias: GenericType = cast("Any", super()).__class_getitem__(arg)  # type: ignore[no-any-explicit]  # pyright: ignore[reportAny, reportExplicitAny]
        return _JustAlias(generic_alias.__origin__, generic_alias.__args__)


class _JustIntMeta(_JustMeta[int]):
    @override
    def __just_type__(self, /) -> type[int]:
        return int


@runtime_checkable
class JustInt(Just[int], Protocol, metaclass=_JustIntMeta):
    """
    A runtime-checkable `Just[int]` that's compatible with `pyright<1.390`.

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

    # workaround for `pyright<1.390` and `basedpyright<1.22.1`
    def __new__(cls, x: str | bytes | bytearray, /, base: CanIndex) -> Self: ...


class _JustFloatMeta(_JustMeta[float]):
    @override
    def __just_type__(self, /) -> type[float]:
        return float


class JustFloat(Just[float], Protocol, metaclass=_JustFloatMeta):
    """
    A runtime checkable `Just[float]` that also works on `pyright<1.390`.

    Warning:
        Unlike `JustInt`, this does not work on `mypy<1.41.2`.
    """

    # workaround for `pyright<1.390` and `basedpyright<1.22.1`
    def hex(self, /) -> str: ...


class _JustComplexMeta(_JustMeta[complex]):
    @override
    def __just_type__(self, /) -> type[complex]:
        return complex


class JustComplex(Just[complex], Protocol, metaclass=_JustComplexMeta):
    """
    A runtime checkable `Just[complex]`, that also works on `pyright<1.390`.

    Warning:
        Unlike `JustInt`, this does not work on `mypy<1.41.2`.
    """

    # workaround for `pyright<1.390` and `basedpyright<1.22.1`
    def __new__(cls, /, real: _ToFloat, imag: _ToFloat) -> Self: ...
