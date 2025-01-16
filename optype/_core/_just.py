import sys
from typing import Protocol, TypeAlias


if sys.version_info >= (3, 13):
    from typing import Self, TypeVar, override
else:
    from typing_extensions import Self, TypeVar, override


from ._can import CanFloat, CanIndex


__all__ = ["Just", "JustComplex", "JustFloat", "JustInt"]


def __dir__() -> list[str]:
    return __all__


###


_T = TypeVar("_T")

_ToFloat: TypeAlias = CanFloat | CanIndex


###


class Just(Protocol[_T]):
    """
    An invariant type "wrapper", where `Just[T]` only accepts instances of `T`, and
    but rejects instances of any strict subtypes of `T`.

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

    @property  # type: ignore[explicit-override]  # seriously..?
    @override
    def __class__(self, /) -> type[_T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @__class__.setter
    @override
    def __class__(self, t: type[_T], /) -> None: ...


class JustInt(Just[int], Protocol):
    """
    A `pyright<1.390` -compatible version of `Just[int]`.

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


class JustFloat(Just[float], Protocol):
    """
    Like `Just[float]`, but also works on `pyright<1.390`.

    Warning:
        Unlike `JustInt`, this does not work on `mypy<1.41.2`.
    """

    # workaround for `pyright<1.390` and `basedpyright<1.22.1`
    def hex(self, /) -> str: ...


class JustComplex(Just[complex], Protocol):
    """
    Like `Just[complex]`, but also works on `pyright<1.390`.

    Warning:
        Unlike `JustInt`, this does not work on `mypy<1.41.2`.
    """

    # workaround for `pyright<1.390` and `basedpyright<1.22.1`
    def __new__(cls, /, real: _ToFloat, imag: _ToFloat) -> Self: ...
