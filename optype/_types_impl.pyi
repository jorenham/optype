import sys
import types as _types
from collections.abc import Iterator
from typing import (
    Any,
    Generic,
    Literal,
    ParamSpec,
    TypeAlias,
    _SpecialForm,  # pyright: ignore[reportPrivateUsage]
    final,
    type_check_only,
)

from typing_extensions import Self, TypeVar, TypeVarTuple, override

__all__ = 'AnnotatedAlias', 'GenericType', 'LiteralAlias', 'UnionAlias'

_TypeOrigin: TypeAlias = (
    type | _types.GenericAlias | GenericType
)
_TypeParam: TypeAlias = TypeVar | TypeVarTuple | ParamSpec

# represents `typing._GenericAlias`
# NOTE: This is different from `typing.GenericAlias`!
class GenericType:
    @property
    def __origin__(self) -> _TypeOrigin: ...
    @property
    def __args__(self) -> tuple[Any, ...]: ...
    @property
    def __parameters__(self) -> tuple[_TypeParam, ...]: ...

    def __init__(self, origin: _TypeOrigin, args: Any) -> None: ...
    def __or__(self, rhs: Any, /) -> UnionAlias: ...
    def __ror__(self, lhs: Any, /) -> UnionAlias: ...
    def __getitem__(self, args: Any, /) -> GenericType: ...
    def copy_with(self, params: Any, /) -> GenericType: ...

    if sys.version_info >= (3, 11):
        def __iter__(self, /) -> Iterator[UnpackAlias[Self]]: ...

    def __call__(self, *args: Any, **kwargs: Any) -> _SpecialForm | object: ...
    def __mro_entries__(self, bases: tuple[type, ...]) -> tuple[type, ...]: ...
    def __instancecheck__(self, obj: object, /) -> bool: ...

@final
class AnnotatedAlias(GenericType): ...
@final
class LiteralAlias(GenericType): ...
@final
class UnionAlias(GenericType): ...

_Ts_co = TypeVar(
    '_Ts_co', covariant=True, bound=tuple[Any, ...] | TypeVarTuple,
)

@type_check_only
class UnpackAlias(GenericType, Generic[_Ts_co]):
    @property
    def __typing_unpacked_tuple_args__(self) -> tuple[Any, ...] | None: ...
    @property
    def __typing_is_unpacked_typevartuple__(self) -> Literal[False, True]: ...

    @property
    @override
    def __origin__(self) -> type[_SpecialForm]: ...
    @property
    @override
    def __args__(self) -> tuple[_Ts_co]: ...
    @property
    @override
    def __parameters__(self) -> tuple[()] | tuple[TypeVarTuple]: ...

    @override
    def __init__(self, origin: _SpecialForm, args: tuple[_Ts_co]) -> None: ...
