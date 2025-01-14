import sys
from typing import Protocol, TypeAlias


if sys.version_info >= (3, 13):
    from typing import TypeVar, runtime_checkable
else:
    from typing_extensions import TypeVar, runtime_checkable


__all__ = ("CanFSPath", "ToPath")


def __dir__() -> tuple[str, str]:
    return __all__


###

# not a type parameter
_StrOrBytes = TypeVar("_StrOrBytes", str, bytes, str | bytes, default=str | bytes)

_StrOrBytesT_co = TypeVar(
    "_StrOrBytesT_co",
    bound=str | bytes,
    default=str | bytes,
    covariant=True,
)


###


@runtime_checkable
class CanFSPath(Protocol[_StrOrBytesT_co]):
    """
    Similar to `os.PathLike`, but is is actually a protocol, doesn't incorrectly use a
    `TypeVar` with constraints", and therefore doesn't force type-unsafe usage of `Any`
    to express `str | bytes`.
    """

    def __fspath__(self, /) -> _StrOrBytesT_co: ...


# A runtime-accessible alternative to `_typeshed.{Str,Bytes,StrOrBytes,Generic}Path`
ToPath: TypeAlias = _StrOrBytes | CanFSPath[_StrOrBytes]
