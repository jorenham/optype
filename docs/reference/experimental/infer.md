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
(tuple[()] = ...) -> list[Never]
[R](CanIter[CanNext[R]] & ~tuple[()]) -> list[R]
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
(CanFloat | CanIndex) -> float
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

A builtin without an `inspect.signature` recovers its parameters from the docstring,
reporting each documented call form as an overload:

```console
$ optype infer "iter"
[R](iterable: CanIter[R]) -> R
[R](callable: () -> R, sentinel: object) -> Iterator[R]
```

Unsatisfiable forms are dropped and same-arity forms collapse into one. The docstring is
also lossy: it marks neither `*args` nor the keyword-only `*`, so both become plain
positionals. Read these forms as the shape of the call, not a faithful signature:

```console
$ optype infer "max"
[T, U: CanGt[T, CanBool], V: CanGt[U | T, CanBool]](iterable: T, default: U, key: V) -> V | U | T
[T, U: CanGt[T, CanBool], V: CanGt[U | T, CanBool], W: CanGt[V | U | T, CanBool]](arg1: T, arg2: U, args: V, key: W) -> W | V | U | T
```

## Intersections

Python has no intersection types, so the `&` is not valid Python: both requirements
apply at once. Since each `optype` protocol declares a single method, an intersection
of distinct protocols is expressible as a protocol that inherits from all members:

```python
class CanNegRAdd[T, R](CanNeg[T], CanRAdd[T, R], Protocol): ...
```

This turns the `CanNeg[T] & CanRAdd[T, R]` above into the valid `CanNegRAdd[T, R]`.
Where `optype` already ships the combined protocol, `infer` reports it directly:
`CanGetitem & CanLen` merges into `CanSequence`, and a traced `with` statement renders
as [`CanWith`](#context-managers) rather than `CanEnter & CanExit`:

```console
$ optype infer "lambda x, i: x[i] if len(x) else None"
[T, R](x: CanSequence[T, R], i: T) -> R | None
```

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

A call dispatches through `__call__`, but a parameter that is called renders in
signature syntax rather than as `CanCall` or `Callable`, with any keyword arguments
as named parameters:

```console
$ optype infer "lambda f: f(1, b=2)"
[R](f: (Literal[1], b: Literal[2]) -> R) -> R
```

## Attributes

Attribute access dispatches through `__getattr__`, except for the few attributes
that every object already has, like `__doc__`, which require nothing. Reading a
special attribute that `optype` ships a single-member `Has*` protocol for reports
that protocol directly:

```console
$ optype infer "lambda x: x.__name__"
[R](x: HasName[R]) -> R
```

An attribute without a shipped protocol renders as the fictional inline form
`Has['name', T]`. Like `&` and `~`, it is not valid Python, but it is expressible as
a protocol with a single member. The type argument carries a polarity sigil: a read
requires only the covariant `+T`, so a `@property` getter suffices:

```python
class HasSpam[T](Protocol):
    @property
    def spam(self) -> T: ...
```

This turns `Has['spam', +R]` into the valid `HasSpam[R]`:

```console
$ optype infer "lambda x: x.spam"
[R](x: Has['spam', +R]) -> R
```

When the attribute is called, the sigil sinks into the callable's return type, where
the covariance applies: `Has['spam', () -> +R]` is the method `def spam(self) -> R`:

```console
$ optype infer "lambda x: x.spam()"
[R](x: Has['spam', () -> +R]) -> R
```

An assignment requires the contravariant `-T`: a mutable attribute that accepts the
assigned value's type, such as `spam: T` itself, or any wider type. A deletion, or a
read whose result is unused, requires only that the attribute exists (deletability
itself is not expressible):

```console
$ optype infer "def f(x): x.spam = 1; return x"
[T: Has['spam', -Literal[1]]](x: T) -> T

$ optype infer "def f(x): del x.spam"
(x: Has['spam']) -> None
```

## Classes

`type` reads the class directly instead of dispatching through a dunder, but every
recording proxy has a unique class, so the result is still tied to its parameter:

```console
$ optype infer "type"
[T](object: T) -> type[T]

$ optype infer "lambda x: type(next(x))"
[R](x: CanNext[R]) -> type[R]
```

That unique class also ties its instances back to the parameter:

```console
$ optype infer "lambda x: type(x)()"
[T](x: T) -> T
```

An attribute access on the class itself renders inside `ClassVar`, mirroring a protocol
member declared as `spam: ClassVar[...]`.

```console
$ optype infer "lambda x: type(x).spam"
[R](x: Has['spam', ClassVar[+R]]) -> R

$ optype infer "def f(x): type(x).spam = 1"
(x: Has['spam', ClassVar[-Literal[1]]]) -> None

$ optype infer "def f(x): del type(x).spam"
(x: Has['spam', ClassVar]) -> None
```

A concrete class renders parameterized when it resolves by name (a local class stays
a bare `type`), and `type` is covariant, so a subclass is absorbed by its parent:

```console
$ optype infer "lambda: bool"
() -> type[bool]

$ optype infer "lambda x: int if x else bool"
(x: CanBool) -> type[int]
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

A predicate is assumed stable within a single call, so repeating it on the same operand
agrees rather than branching again. A self-contradicting guard is therefore never
satisfiable, and its body is left untraced:

```console
$ optype infer "lambda x: x.foo() if (x and not x) else x"
[T: CanBool](x: T) -> T
```

Distinct operands stay independent, so `a in x` and `b in x` still branch separately.

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

A Python 3.15+ `sentinel` is its own type per
[PEP 661](https://peps.python.org/pep-0661/), spelled as its declared name, so the
common sentinel-default pattern renders just like the `None` default above:

```console
$ optype infer "MISSING = sentinel('MISSING')
def f(x=MISSING): return [] if x is MISSING else x"
(x: MISSING = MISSING) -> list[Never]
[T: ~MISSING](x: T) -> T
```

## Methods

Anything callable can be inferred: not just functions, but also builtins (like
`math.sqrt` above), callable instances, and unbound method descriptors. A
positional-only parameter cannot be passed by keyword, so it renders as a bare type
without its name. A method descriptor's `self` requires a real instance of its
defining class, so it is reported as that concrete type:

```console
$ optype infer "str.upper"
(str) -> str

$ optype infer "dict.get"
[T = None](dict, CanHash, T = None) -> T
```

When a builtin rejects the recording proxy for a defaulted parameter, its default is
passed instead and the parameter is pinned to it, while the accepting parameters stay
structural:

```console
$ optype infer "str.split"
(str, sep: None = None, maxsplit: CanIndex = -1) -> list[Never]
```

## Context managers

A `with` statement requires `__enter__` and `__exit__` together, which `optype` combines
as `CanWith`; the `__exit__` result is unused, so it stays `object`:

```pycon
>>> def f(x):
...     with x as y:
...         return y
>>> infer(f)
'[R](x: CanWith[R, object]) -> R'
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

`async with` combines `__aenter__` and `__aexit__` as `CanAsyncWith`, whose parameters
are the awaited results:

```pycon
>>> async def f(x):
...     async with x as y:
...         return y
>>> infer(f)
'[R](x: CanAsyncWith[R, object]) -> R'
```

## Generators and lazy iterators

A generator is lazy, so it is iterated to collect the types it yields:

```console
$ optype infer "def f(n): yield from range(n)"
(n: CanIndex) -> Generator[int]

$ optype infer "def f(): yield None; yield 1"
() -> Generator[None | int]

$ optype infer "async def f(xs): return (x async for x in xs)"
[R](xs: CanAIter[CanANext[CanAwait[R]]]) -> AsyncGenerator[R]
```

Lazy builtin iterators (`map`, `filter`, `zip`, and `enumerate`) are iterated the same
way, and a callable argument is traced right through them:

```console
$ optype infer "lambda x: map(str, x)"
(x: CanIter[CanNext[CanStr]]) -> map[str]

$ optype infer "lambda f, x: map(f, x)"
[T, R](f: (T) -> R, x: CanIter[CanNext[T]]) -> map[R]
```

## Unpacking

An unpacking iterates the parameter, and every target shares one element type, like
`*args`; a starred target collects the rest into a `list`:

```console
$ optype infer "def f(x): a, b = x; return a, b"
[R](x: CanIter[CanNext[R]]) -> tuple[R, R]

$ optype infer "def f(x): a, *b = x; return a, b"
[R](x: CanIter[CanNext[R]]) -> tuple[R, list[R]]

$ optype infer "lambda x: {k: v for k, v in x}"
[R: CanHash](x: CanIter[CanNext[CanIter[CanNext[R]]]]) -> dict[R, R]
```

## Returned functions

A returned function is lazy too, so it is explored with placeholders of its own, and
its type renders in the same signature syntax:

```console
$ optype infer "lambda x: lambda y: (x, y)"
[T, U](x: T) -> (y: U) -> tuple[T, U]
```

An operation inside the returned function is required of the closed-over parameter,
and overloads reflect as usual:

```console
$ optype infer "lambda x: lambda y: x + y"
[T, R](x: CanAdd[T, R]) -> (y: T) -> R
[T, R](x: T) -> (y: CanRAdd[T, R]) -> R
```

This applies to anything function-like: builtins, method descriptors, and a
`functools.partial` (explored with its bound arguments in place), also from within a
returned container:

```console
$ optype infer "lambda: str.upper"
() -> (str) -> str

$ optype infer "import functools
def add(x, y): return x + y
lambda: functools.partial(add, 1)"
[R]() -> (y: CanRAdd[Literal[1], R]) -> R

$ optype infer "def counter(start): return (lambda: start), (lambda by: start + by)"
[T: CanAdd[U, R], U, R](start: T) -> tuple[() -> T, (by: U) -> R]
[T, R](start: T) -> tuple[() -> T, (by: CanRAdd[T, R]) -> R]
```

A recursive function (factory) has an inexpressible type, so it stays an opaque
`function`, as does one with variadic parameters:

```console
$ optype infer "def f(x): return f"
(x: object) -> function
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

The Python 3.15+ `frozendict` parametrizes like `dict` does:

```console
$ optype infer "lambda x: frozendict({'k': x + 1})"
[R](x: CanAdd[Literal[1], R]) -> frozendict[Literal['k'], R]
```

A container that holds itself is a recursive type. The cycle is detected by identity
and tied off as a typevar bounded by its own structure:

```console
$ optype infer "def f(): x = []; x.append(x); return x"
[R: list[R]]() -> R
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
[R](a: CanArrayFunction[(Any) -> R, R]) -> R
```

## Limitations

`infer` calls the function, so it only works on functions that are safe to run with
placeholder arguments (no real side effects, no reliance on concrete values). This
extends to anything it returns: a returned function is called with placeholders of its
own, and a returned lazy iterator is iterated. A returned function that raises during
this exploration is not treated as an error: its type stays an opaque `function`.
An attribute probe with a fallback, such as `hasattr` or `getattr` with a default,
reports the attribute as a requirement anyway: a placeholder has every attribute, so
the fallback branch is never taken.

When `infer` can't handle the input, it raises `InferError` (a `NotImplementedError`
subclass). That's the case for operations without a matching protocol, such as an
attribute access whose name is not statically known (a computed `getattr`), and for
arguments that aren't callable to begin with. Exceptions raised from within the
function itself aren't caught.

Variadic parameters are explored with a few placeholders, retried with more after an
out-of-range index, a failed unpacking, or a missing `**kwargs` key, and reported as an
`InferError` once the budget runs out. The placeholders remain observable: `len(args)`
reports their count, and `args` and `kwargs` are never empty.

The number of explored branches is capped, so a function with many of them gets a
signature that only covers the explored ones, along with an `InferWarning`.

It can only observe operations that go through a dunder method. Anything that inspects a
parameter at the C level is invisible, so a parameter passed to `id()`, `isinstance()`,
or an identity check (`is`) is reported as `object` rather than its real requirement.

!!! warning "Generic bounds"

    An inferred typevar bound can itself be generic, such as `[T: CanAdd[T, R], R]` where
    `T`'s bound references `T` and `R`. Python's type system does not currently support
    generic typevar bounds, so these signatures are not always valid Python.
