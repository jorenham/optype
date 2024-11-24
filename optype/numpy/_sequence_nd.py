from typing import Any, Protocol, TypeVar


__all__ = ["SequenceND"]


T_co = TypeVar("T_co", covariant=True)


class SequenceND(Protocol[T_co]):
    # a slimmed down version of `numpy._typing._NestedSequence`
    def __len__(self, /) -> int: ...
    def __getitem__(self, index: int, /) -> "T_co | SequenceND[T_co]": ...
    def __contains__(self, x: object, /) -> bool: ...
    def index(self, value: Any, /) -> int: ...  # type: ignore[no-any-explicit]  # pyright: ignore[reportAny, reportExplicitAny]
