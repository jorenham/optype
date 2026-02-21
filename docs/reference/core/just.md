# Just - Exact Type Matching

`Just` is an invariant type "wrapper", where `Just[T]` only accepts instances of `T`,
and rejects instances of any strict subtypes of `T`.

Note that e.g. `Literal[""]` and `LiteralString` are not a strict `str` subtypes,
and are therefore assignable to `Just[str]`, but instances of `class S(str): ...`
are **not** assignable to `Just[str]`.

Disallow passing `bool` as `int`:

```py
import optype as op


def assert_int(x: op.Just[int]) -> int:
    assert type(x) is int
    return x


assert_int(42)  # ok
assert_int(False)  # rejected
```

Annotating a sentinel:

```py
import optype as op

_DEFAULT = object()


def intmap(
    value: int,
    # same as `dict[int, int] | op.Just[object]`
    mapping: dict[int, int] | op.JustObject = _DEFAULT,
    /,
) -> int:
    # same as `type(mapping) is object`
    if isinstance(mapping, op.JustObject):
        return value

    return mapping[value]


intmap(1)  # ok
intmap(1, {1: 42})  # ok
intmap(1, "some object")  # rejected
```

Ensuring that a function returns just `Any` (statically):

```py
import optype as op
from typing import Any

def returns_any() -> Any: ...
def returns_obj() -> object: ...
def returns_set() -> set[Any]: ...

def assert_any(x: op.JustAny, /) -> None: ...

assert_any(returns_any())  # ok
assert_any(returns_obj())  # rejected
assert_any(returns_set())  # rejected
```

!!! tip

    The `Just{Bytes,Int,Float,Complex,Date,Object}` protocols are runtime-checkable,
    so that `instance(42, JustInt) is True` and `instance(bool(), JustInt) is False`.
    It's implemented through meta-classes, and type-checkers have no problem with it.

    The only exception is `JustAny`, which is not runtime-checkable.

| `optype` type | accepts instances of             |
| ------------- | -------------------------------- |
| `Just[~T]`    | `T`                              |
| `JustInt`     | `builtins.int`                   |
| `JustFloat`   | `builtins.float`                 |
| `JustComplex` | `builtins.complex`               |
| `JustBytes`   | `builtins.bytes`                 |
| `JustObject`  | `builtins.object`                |
| `JustDate`    | `datetime.date`                  |
| `JustAny`     | `typing.Any`, `type[typing.Any]` |
