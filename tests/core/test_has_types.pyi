# pyright: reportInvalidStubStatement=false

from typing import TypedDict, TypeVar
from typing_extensions import LiteralString, assert_type

import optype as op

_TypeT = TypeVar("_TypeT", bound=type)

class A: ...
class B(A): ...
class C(B): ...

a: A
b: B
c: C

def typeof(obj: op.HasClass[_TypeT], /) -> _TypeT: ...

###
# HasClass

a__a: op.HasClass[type[A]] = a
b__b: op.HasClass[type[B]] = b
c__c: op.HasClass[type[C]] = c

a__b: op.HasClass[type[A]] = b  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
a__c: op.HasClass[type[A]] = c  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

b__a: op.HasClass[type[B]] = a  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
b__c: op.HasClass[type[B]] = c  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

c__a: op.HasClass[type[C]] = a  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
c__b: op.HasClass[type[C]] = b  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

int_obj: int
assert_type(type(int_obj), type[int])
assert_type(typeof(int_obj), type[int])

int_str: int | str
assert_type(type(int_str), type[int | str])
typeof(int_str)  # type: ignore[misc]  # pyright: ignore[reportArgumentType]

bool_or_int: bool | int
assert_type(type(bool_or_int), type[bool] | type[int])
assert_type(typeof(bool_or_int), type[int])  # type: ignore[arg-type]  # mypy fail

lit_str: LiteralString
assert_type(type(lit_str), type[str])
assert_type(typeof(lit_str), type[str])

class TDict(TypedDict): ...

tdict: TDict
assert_type(type(tdict), type[TDict])
assert_type(typeof(tdict), type[TDict])  # type: ignore[arg-type]
