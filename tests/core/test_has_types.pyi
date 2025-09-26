# pyright: reportInvalidStubStatement=false

from typing import LiteralString, TypedDict, TypeVar, assert_type

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
# https://github.com/facebook/pyrefly/issues/1165
assert_type(type(int_str), type[int] | type[str])
typeof(int_str)  # type: ignore[misc]  # pyright: ignore[reportArgumentType]

bool_or_int: bool | int
assert_type(type(bool_or_int), type[bool] | type[int])
assert_type(typeof(bool_or_int), type[int])  # type: ignore[arg-type]  # mypy fail

lit_str: LiteralString
# https://github.com/facebook/pyrefly/issues/1166
assert_type(type(lit_str), type[str])  # pyrefly: ignore[assert-type]
assert_type(typeof(lit_str), type[str])

class TDict(TypedDict): ...

tdict: TDict
# https://github.com/facebook/pyrefly/issues/1167
assert_type(type(tdict), type[TDict])  # pyrefly: ignore[assert-type]
assert_type(typeof(tdict), type[TDict])  # type: ignore[arg-type]  # mypy fail
