---
status: new
tags:
  - experimental
  - v0.18+
---

# optype.infer

!!! warning "Experimental"

    The `optype.infer` module is experimental and its API may change without notice.

`optype.infer` works out which `optype` protocols a function requires of its parameters.
It runs the function against recording proxies that trace every operation, then renders
the result as a [PEP 695](https://peps.python.org/pep-0695/) signature.

## Usage

`infer(func, *params)` returns the inferred signature as a string:

```pycon
>>> from optype.infer import infer
>>> infer(lambda x: x + 1)
'[R](x: CanAdd[Literal[1], R]) -> R'
>>> infer(list)
'[R](iterable: CanIter[CanNext[R]] & CanLen) -> list[R]'
```

Pass parameter names or positions to report only those parameters:

```pycon
>>> infer(lambda x, y: x[y], "x")
'[T, R](x: CanGetitem[T, R]) -> R'
```

The `optype infer` command takes a Python expression; leading statements are allowed, as
long as the last line is an expression:

```console
$ optype infer "lambda x: x * 2"
[R](x: CanMul[Literal[2], R]) -> R

$ optype infer "import math; math.sqrt"
(x: CanFloat | CanIndex) -> float
```

## Reflected operators

A binary operator can dispatch to either operand, so it is reported as two overloads,
one per line:

```console
$ optype infer "lambda x, y: x * y"
[T, R](x: CanMul[T, R], y: T) -> R
[T, R](x: T, y: CanRMul[T, R]) -> R
```

## Limitations

`infer` calls the function, so it only works on functions that are safe to run with
placeholder arguments (no real side effects, no reliance on concrete values).

It can only observe operations that go through a dunder method. Anything that inspects a
parameter at the C level is invisible, so a parameter passed to `type()`, `id()`,
`isinstance()`, or an identity check (`is`) is reported as `object` rather than its real
requirement.

!!! warning "Generic bounds"

    An inferred typevar bound can itself be generic, such as `[T: CanAdd[T, R], R]` where
    `T`'s bound references `T` and `R`. Python's type system does not currently support
    generic typevar bounds, so these signatures are not always valid Python.
