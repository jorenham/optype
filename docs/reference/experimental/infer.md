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

The `optype infer` command takes a Python snippet whose final statement is
an expression or a `def`/`class` definition; any leading statements run as setup:

```console
$ optype infer "lambda x: x * 2"
[R](x: CanMul[Literal[2], R]) -> R

$ optype infer "import math; math.sqrt"
(x: CanFloat | CanIndex) -> float
```

## Overloads

A binary operator can dispatch to either operand, so it is reported as one overload per
line:

```console
$ optype infer "lambda x, y: x * y"
[T, R](x: CanMul[T, R], y: T) -> R
[T, R](x: T, y: CanRMul[T, R]) -> R
```

An operator applied to its own result yields a recursive bound, where the bound of `T`
refers back to `T`:

```console
$ optype infer "lambda x: -x + x"
[T: CanNeg[CanAdd[T, R]], R](x: T) -> R
[T, R](x: CanNeg[T] & CanRAdd[T, R]) -> R
```

## Branches

Both sides of a conditional are explored, so the parameter has to satisfy every branch
(an intersection) and the return is the union of the branch results:

```console
$ optype infer "lambda x: x if x > 0 else -x"
[T: CanGt[Literal[0], CanBool] & CanNeg[R], R](x: T) -> T | R
```

Branching and overloads combine:

```console
$ optype infer "lambda x, y: (x + y) if x else y"
[T, R](x: CanBool & CanAdd[T, R], y: T) -> R | T
[T: CanBool, U: CanRAdd[T, R], R](x: T, y: U) -> R | U
```

## Async

Coroutine functions are run to completion, so `await`, `async with`, and `async for` are
traced like their synchronous counterparts:

```console
$ optype infer "async def f(x): return await x"
[R](x: CanAwait[R]) -> R

$ optype infer "async def f(xs): return [x async for x in xs]"
[R](xs: CanAIter[CanANext[CanAwait[R]]]) -> list[R]
```

## NumPy

!!! info

    NumPy is not a required dependency, and `optype infer` works fine without it
    installed.

A [ufunc](https://numpy.org/doc/stable/reference/ufuncs.html) requires each operand to
either override it (NEP 13's `CanArrayUFunc`) or be an array-like of its widest accepted
dtype (read from its `.types`):

```console
$ optype infer "import numpy as np; np.sin"
[R](x: CanArrayUFunc[np.ufunc, R] | ToComplexND) -> R
```

A [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html) function such as
`np.mean` requires the `CanArrayFunction` override:

```console
$ optype infer "import numpy as np; np.mean"
[R](a: CanArrayFunction[CanCall[Any, R], R]) -> R
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
