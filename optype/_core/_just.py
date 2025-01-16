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
    Experimental "invariant" wrapper type, so that `Invariant[int]` only accepts `int`
    but not `bool` (or any other `int` subtypes).

    Important:
        This requires `pyright>=1.390` / `basedpyright>=1.22.1` to work.

    Warning:
        In mypy this doesn't work with the special-cased `float` and `complex`,
        caused by (at least) one of the >1200 confirmed mypy bugs.

    Note:
        The only reason that this worked on mypy `1.13.0` and below, is because
        of (yet another) bug, where mypy blindly aassumes that the setter of a property
        has the same parameter type as the return type of the getter, even though that's
        the main usecase of `@property`...
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
        def f_fixed(x: JustInt, /) -> int:
            assert type(x) is int
            return x  # accepted


        f_fixed(1337)  # accepted
        f_fixed(True)  # rejected :)
        ```

    Note:
        On `mypy` this behaves in the same way as `Just[int]`, which accidentally
        already works because of a mypy bug.
    """

    def __new__(cls, x: str | bytes | bytearray, /, base: CanIndex) -> Self: ...


class JustFloat(Just[float], Protocol):
    """
    Like `Just[float]`, but also works on `pyright<1.390`.

    Warning:
        Unlike `JustInt`, this does not work on `mypy<1.41.2`.
    """

    def hex(self, /) -> str: ...


class JustComplex(Just[complex], Protocol):
    """
    Like `Just[complex]`, but also works on `pyright<1.390`.

    Warning:
        Unlike `JustInt`, this does not work on `mypy<1.41.2`.
    """

    def __new__(cls, /, real: _ToFloat, imag: _ToFloat) -> Self: ...
