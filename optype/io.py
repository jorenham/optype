import sys
from typing import Protocol, TypeAlias


if sys.version_info >= (3, 13):
    from typing import TypeVar, runtime_checkable
else:
    from typing_extensions import TypeVar, runtime_checkable


__all__ = (
    "CanFSPath",
    "CanFileno",
    "CanFlush",
    "CanRead",
    "CanReadN",
    "CanReadline",
    "CanReadlineN",
    "CanWrite",
    "ToFileno",
    "ToPath",
)


def __dir__() -> tuple[str, ...]:
    return __all__


###

# not a type parameter
_StrOrBytes = TypeVar("_StrOrBytes", str, bytes, str | bytes, default=str | bytes)

_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)
_RT_co = TypeVar("_RT_co", default=object, covariant=True)

_PathT_co = TypeVar("_PathT_co", bound=str | bytes, default=str | bytes, covariant=True)

###


@runtime_checkable
class CanFSPath(Protocol[_PathT_co]):
    """
    Similar to `os.PathLike`, but is is actually a protocol, doesn't incorrectly use a
    `TypeVar` with constraints", and therefore doesn't force type-unsafe usage of `Any`
    to express `str | bytes`.
    """

    def __fspath__(self, /) -> _PathT_co: ...


@runtime_checkable
class CanFileno(Protocol):
    """Runtime-checkable equivalent of `_typeshed.HasFileno`."""

    def fileno(self, /) -> int: ...


@runtime_checkable
class CanRead(Protocol[_T_co]):
    """
    Like `_typeshed.SupportsRead`, but without the required positional `int` argument,
    and is runtime-checkable.
    """

    def read(self, /) -> _T_co: ...


@runtime_checkable
class CanReadN(Protocol[_T_co]):
    """Runtime-checkable equivalent of `_typeshed.SupportsRead`."""

    def read(self, n: int = ..., /) -> _T_co: ...


@runtime_checkable
class CanReadline(Protocol[_T_co]):
    """
    Runtime-checkable equivalent of `_typeshed.SupportsNoArgReadline`, that
    additionally allows `self` to be positional-only.
    """

    def readline(self, /) -> _T_co: ...


@runtime_checkable
class CanReadlineN(Protocol[_T_co]):
    """Runtime-checkable equivalent of `_typeshed.SupportsReadline`."""

    def readline(self, n: int = ..., /) -> _T_co: ...


@runtime_checkable
class CanWrite(Protocol[_T_contra, _RT_co]):
    """
    Runtime-checkable equivalent of `_typeshed.SupportsWrite`, with an additional
    optional type parameter for the return type, that defaults to `object`.
    """

    def write(self, data: _T_contra, /) -> _RT_co: ...


@runtime_checkable
class CanFlush(Protocol[_RT_co]):
    """
    Runtime-checkable equivalent of `_typeshed.SupportsFlush`, with an additional
    optional type parameter for the return type, that defaults to `object`.
    """

    def flush(self, /) -> _RT_co: ...


# runtime-checkable `_typeshed.{Str,Bytes,StrOrBytes,Generic}Path` alternative
ToPath: TypeAlias = _StrOrBytes | CanFSPath[_StrOrBytes]

# runtime-checkable `_typeshed.FileDescriptorLike` equivalent
ToFileno: TypeAlias = int | CanFileno
