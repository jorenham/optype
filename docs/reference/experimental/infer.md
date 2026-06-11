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
>>> print(infer(list))
(iterable: tuple[()] = ...) -> list[Never]
[R](iterable: CanIter[CanNext[R]] & ~tuple[()]) -> list[R]
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

## Intersections

Python has no intersection types, so the `&` is not valid Python: both requirements
apply at once. Since each `optype` protocol declares a single method, an intersection
of distinct protocols is expressible as a protocol that inherits from all members:

```python
class CanNegRAdd[T, R](CanNeg[T], CanRAdd[T, R], Protocol): ...
```

This turns the `CanNeg[T] & CanRAdd[T, R]` above into the valid `CanNegRAdd[T, R]`.

Only an intersection of the same protocol shares its method, which happens when an
operation is traced at several arities:

```console
$ optype infer "lambda x: (round(x), round(x, 2))"
[R, R2](x: CanRound[R] & CanRound[Literal[2], R2]) -> tuple[R, R2]
```

That takes `@overload`s instead, which `optype` ships as the three-parameter
`CanRound`: this intersection is `CanRound[Literal[2], R, R2]`.

## Operators

Every operation that dispatches through a dunder is traced. That includes augmented
assignments, which map to the in-place protocols:

```console
$ optype infer "def f(x, y): x += y; return x"
[T, R](x: CanIAdd[T, R], y: T) -> R
```

The same goes for builtins such as `divmod`, `round`, and `reversed`, which dispatch
through dunders too:

```console
$ optype infer "lambda x: divmod(x, 2)"
[R](x: CanDivmod[Literal[2], R]) -> R

$ optype infer "lambda x: round(x, 2)"
[R](x: CanRound[Literal[2], R]) -> R

$ optype infer "lambda x: reversed(x)"
[R](x: CanReversed[R]) -> R
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

A comparison chain short-circuits, so it branches too: a falsy first comparison is
returned as-is, and only a truthy one evaluates the second (`0 < x` reflects to
`x.__gt__`):

```console
$ optype infer "lambda x: 0 < x < 10"
[R, R2: CanBool](x: CanGt[Literal[0], R2 & CanBool] & CanLt[Literal[10], R]) -> R | R2
```

## Variadic parameters

A `*args` parameter that is only passed around as a whole is inferred as a
[PEP 646](https://peps.python.org/pep-0646/) variadic type parameter, `*Ts`:

```console
$ optype infer "lambda *args: args"
[*Ts](*args: *Ts) -> tuple[*Ts]

$ optype infer "lambda *args: (1, *args)"
[*Ts](*args: *Ts) -> tuple[Literal[1], *Ts]
```

Operating on individual elements is not expressible with a variadic type parameter, so
the elements then share a single inferred element type, as does every value of
`**kwargs`:

```console
$ optype infer "lambda *args: args[0] + args[1]"
[T: CanAdd[T, R], R](*args: T) -> R
[T: CanRAdd[T, R], R](*args: T) -> R

$ optype infer "lambda **kwargs: kwargs"
[T](**kwargs: T) -> dict[str, T]
```

## Parameter defaults

[PEP 696](https://peps.python.org/pep-0696/) type parameter defaults are used when
appropriate:

```console
$ optype infer "def f(x=0): return x"
[T = Literal[0]](x: T = 0) -> T
```

A parameter without its own typevar shows the default inline instead:

```console
$ optype infer "def f(x=0): return str(x)"
(x: CanStr = 0) -> str
```

When omission behaves differently, such as when the function branches on the default,
the call without the argument is reported as a separate overload:

```console
$ optype infer "def f(x=None): return [] if x is None else x"
(x: None = None) -> list[Never]
[T: ~None](x: T) -> T
```

The `~None` complement makes the overloads disjoint: the first one covers `f()` and
`f(None)`, and the second one everything else. Like `&`, the `~` is not valid Python;
in practice it's fine to omit it, as overloads are matched in order anyway.

## Methods

Anything callable can be inferred: not just functions, but also builtins (like
`math.sqrt` above), callable instances, and unbound method descriptors. A method
descriptor's `self` requires a real instance of its defining class, so it is reported
as that concrete type:

```console
$ optype infer "str.upper"
(self: str) -> str

$ optype infer "dict.get"
[T = None](self: dict, key: CanHash, default: T = None) -> T
```

When a builtin rejects the recording proxy for a defaulted parameter, its default is
passed instead and the parameter is pinned to it, while the accepting parameters stay
structural:

```console
$ optype infer "str.split"
(self: str, sep: None = None, maxsplit: CanIndex = -1) -> list[Never]
```

## Context managers

A `with` statement requires `CanEnter` and `CanExit`; on a clean exit, `__exit__` is
called with `(None, None, None)`:

```pycon
>>> def f(x):
...     with x as y:
...         return y
>>> infer(f)
'[R](x: CanEnter[R] & CanExit[None, None, None]) -> R'
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

`async with` requires `CanAEnter` and `CanAExit`, whose results must be awaitable:

```pycon
>>> async def f(x):
...     async with x as y:
...         return y
>>> infer(f)
'[R](x: CanAEnter[CanAwait[R]] & CanAExit[None, None, None, CanAwait]) -> R'
```

## Generators

A generator is lazy, so it is iterated to collect the types it yields:

```console
$ optype infer "def f(n): yield from range(n)"
(n: CanIndex) -> Generator[int]

$ optype infer "def f(): yield None; yield 1"
() -> Generator[None | int]

$ optype infer "async def f(xs): return (x async for x in xs)"
[R](xs: CanAIter[CanANext[CanAwait[R]]]) -> AsyncGenerator[R]
```

## Containers

Element types are tracked through the containers that hold them, so a typevar can
surface at any depth, on either side of the signature:

```console
$ optype infer "lambda x: (x + 1, x + 1)"
[R](x: CanAdd[Literal[1], R]) -> tuple[R, R]

$ optype infer "lambda x: {0: [v + 1 for v in x[0]]}"
[R](x: CanGetitem[Literal[0], CanIter[CanNext[CanAdd[Literal[1], R]]]]) -> dict[Literal[0], list[R]]
```

## Unions

A union member that is a subtype of another member is absorbed into it, following the
runtime subclass relations, such as `bool <: int` and `FileNotFoundError <: OSError`:

```console
$ optype infer "def f(): yield True; yield 1"
() -> Generator[int]

$ optype infer "def f(): yield FileNotFoundError(); yield OSError()"
() -> Generator[OSError]
```

PEP 484's `int <: float <: complex` numeric tower is a static-typing fiction with no
runtime counterpart, so it is not applied:

```console
$ optype infer "lambda x: (x + 1, x + 1.0)"
[R](x: CanAdd[Literal[1] | float, R]) -> tuple[R, R]
```

Variance is respected: a covariant `tuple` simplifies, but an invariant `list` does not:

```console
$ optype infer "lambda x: (OSError(), 'a') if x else (FileNotFoundError(), 'a')"
(x: CanBool) -> tuple[OSError, Literal['a']]

$ optype infer "lambda x: [FileNotFoundError()] if x else [OSError()]"
(x: CanBool) -> list[FileNotFoundError] | list[OSError]
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

When `infer` can't handle the input, it raises `InferError` (a `NotImplementedError`
subclass). That's the case for operations without a matching protocol, and for
arguments that aren't callable to begin with. Exceptions raised from within the
function itself aren't caught.

Variadic parameters are explored with a few placeholders, retried with more after an
out-of-range index, a failed unpacking, or a missing `**kwargs` key, and reported as an
`InferError` once the budget runs out. The placeholders remain observable: `len(args)`
reports their count, and `args` and `kwargs` are never empty.

The number of explored branches is capped, so a function with many of them gets a
signature that only covers the explored ones, along with an `InferWarning`.

It can only observe operations that go through a dunder method. Anything that inspects a
parameter at the C level is invisible, so a parameter passed to `type()`, `id()`,
`isinstance()`, or an identity check (`is`) is reported as `object` rather than its real
requirement.

!!! warning "Generic bounds"

    An inferred typevar bound can itself be generic, such as `[T: CanAdd[T, R], R]` where
    `T`'s bound references `T` and `R`. Python's type system does not currently support
    generic typevar bounds, so these signatures are not always valid Python.
