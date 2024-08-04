from typing import Annotated, Generic, Literal, TypeVar


__all__ = 'AnnotatedAlias', 'GenericType', 'LiteralAlias', 'UnionAlias'


_T = TypeVar('_T')
class _C(Generic[_T]):  ...  # noqa: E302


# typing._GenericAlias
# NOTE: this is not the same as`types.GenericAlias`!
GenericType = type(_C[None])

# typing._AnnotatedAlias
AnnotatedAlias = type(Annotated[None, None])

# typing._LiteralGenericAlias
LiteralAlias = type(Literal[None])

# typing._UnionGenericAlias
# NOTE: this is not the same as `types.UnionType`!
UnionAlias = type(Literal[None] | None)
