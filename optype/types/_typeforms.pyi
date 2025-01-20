import sys
import types as _types
from collections.abc import Iterator
from typing import Any, Generic, TypeAlias, _SpecialForm, type_check_only
from typing_extensions import (
    ParamSpec,
    Self,
    TypeAliasType,
    TypeVar,
    TypeVarTuple,
    Unpack,
    final,
    override,
)

__all__ = "AnnotatedAlias", "GenericType", "LiteralAlias", "UnionAlias"

_Ts_co = TypeVar("_Ts_co", bound=tuple[object, ...] | TypeVarTuple, covariant=True)

_TypeExpr: TypeAlias = type | _types.GenericAlias | GenericType | TypeAliasType
_TypeParam: TypeAlias = (
    TypeVar | ParamSpec | TypeVarTuple | UnpackAlias[tuple[object, ...]]
)

# represents `typing._GenericAlias`
# NOTE: This is different from `typing.GenericAlias`!
class GenericType:
    @property
    def __origin__(self, /) -> _TypeExpr | _SpecialForm: ...
    @property
    def __args__(self, /) -> tuple[Any, ...]: ...
    @property
    def __parameters__(self, /) -> tuple[_TypeParam, ...]: ...
    @override
    def __init_subclass__(cls, /, *, _root: bool = ...) -> None: ...
    def __init__(
        self,
        origin: _TypeExpr | _SpecialForm,
        args: tuple[object, ...] | object,
        /,
    ) -> None: ...
    def __or__(self, rhs: type | object, /) -> UnionAlias: ...
    def __ror__(self, lhs: type | object, /) -> UnionAlias: ...
    def __getitem__(self, args: type | object, /) -> GenericType: ...
    def copy_with(self, params: object, /) -> GenericType: ...

    if sys.version_info >= (3, 11):
        def __iter__(self, /) -> Iterator[UnpackAlias[Self]]: ...

    def __call__(self, /, *args: object, **kwargs: object) -> _SpecialForm | object: ...
    def __instancecheck__(self, obj: object, /) -> bool: ...
    def __subclasscheck__(self, obj: type, /) -> bool: ...
    def __mro_entries__(self, bases: tuple[type, ...]) -> tuple[type, ...]: ...

@final
class LiteralAlias(GenericType): ...

@final
class UnionAlias(GenericType): ...

@final
class AnnotatedAlias(GenericType):
    @property
    @override
    def __origin__(self, /) -> _TypeExpr: ...
    @property
    def __metadata__(self, /) -> tuple[object, Unpack[tuple[object, ...]]]: ...

@type_check_only
class UnpackAlias(GenericType, Generic[_Ts_co]):
    @property
    def __typing_unpacked_tuple_args__(self, /) -> tuple[object, ...] | None: ...
    @property
    def __typing_is_unpacked_typevartuple__(self, /) -> bool: ...
    @property
    @override
    def __origin__(self, /) -> type[_SpecialForm]: ...
    @property
    @override
    def __args__(self, /) -> tuple[_Ts_co]: ...
    @property
    @override
    def __parameters__(self, /) -> tuple[()] | tuple[TypeVarTuple]: ...
    @override
    def __init__(self, origin: _SpecialForm, args: tuple[_Ts_co], /) -> None: ...
