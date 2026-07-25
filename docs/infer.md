---
status: new
tags:
  - experimental
  - v0.18+
---

# Type Inference

!!! warning "Experimental"

    The `optype.infer` module is experimental and its API may change without notice.

`optype.infer` works out which `optype` protocols a function requires of its parameters,
and reports the result as a [PEP 695](https://peps.python.org/pep-0695/) signature.

For the Python API surface (the `infer` function and its exceptions), see the
[`optype.infer`](reference/experimental/infer.md) reference.

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

## Output formats

The terse form above is compact but not valid Python. Pass `--format compat` (or
`backend="compat"` to `infer`) to emit a self-contained, type-checkable `.pyi` stub
instead:

```console
$ optype infer --format compat "lambda x, y: x * y"
from typing import overload
from optype import CanMul, CanRMul

@overload
def f[T, R](x: CanMul[T, R], y: T) -> R: ...
@overload
def f[T, R](x: T, y: CanRMul[T, R]) -> R: ...
```

The fictional forms below are lowered to valid Python: [intersections](#intersections)
and the inline [`Has[...]`](#attributes) form become protocols, and the
[`~` complement](#parameter-defaults) is dropped.

## Overloads

A binary operator can dispatch to either operand. Both possibilities are reported, one
overload per line:

```console
$ optype infer "lambda x, y: x * y"
[T, R](x: CanMul[T, R], y: T) -> R
[T, R](x: T, y: CanRMul[T, R]) -> R
```

An operator applied to its own result gives a bound that refers back to its own typevar:

```console
$ optype infer "lambda x: -x + x"
[T: CanNeg[CanAdd[T, R]], R](x: T) -> R
[T, R](x: CanNeg[T] & CanRAdd[T, R]) -> R
```

## Intersections

Python has no intersection types. The `&` is not valid Python: it means that both
requirements apply at once. Where `optype` already ships the combined protocol, `infer`
reports it directly: `CanGetitem & CanLen` merges into `CanSequence`, and a `with`
statement gives [`CanWith`](#context-managers) instead of `CanEnter & CanExit`:

```console
$ optype infer "lambda x, i: x[i] if len(x) else None"
[T, R](x: CanSequence[T, R], i: T) -> R | None
```

A protocol can also intersect with itself, when an operation is used at several arities:

```console
$ optype infer "lambda x: (round(x), round(x, 2))"
[R, R2](x: CanRound[R] & CanRound[Literal[2], R2]) -> tuple[R, R2]
```

Here `optype` ships the three-parameter `CanRound`, so this intersection is
`CanRound[Literal[2], R, R2]`.

## Operators

Augmented assignments map to the in-place protocols:

```console
$ optype infer "def f(x, y): x += y; return x"
[T, R](x: CanIAdd[T, R], y: T) -> R
```

The same goes for builtins such as `divmod`, `round`, and `reversed`:

```console
$ optype infer "lambda x: divmod(x, 2)"
[R](x: CanDivmod[Literal[2], R]) -> R

$ optype infer "lambda x: round(x, 2)"
[R](x: CanRound[Literal[2], R]) -> R

$ optype infer "lambda x: reversed(x)"
[R](x: CanReversed[R]) -> R
```

A parameter that is called gets signature syntax instead of `CanCall` or `Callable`, with
any keyword arguments as named parameters:

```console
$ optype infer "lambda f: f(1, b=2)"
[R](f: (Literal[1], b: Literal[2]) -> R) -> R
```

## Attributes

Where `optype` ships a `Has*` protocol for the attribute, that protocol is reported
directly. The few attributes that every object already has, like `__doc__`, require
nothing:

```console
$ optype infer "lambda x: x.__name__"
[R](x: HasName[R]) -> R
```

Attributes without a shipped protocol use the fictional inline form `Has['name', T]`.
Like `&` and `~`, that form is not valid Python. Its type argument carries a variance
sign: a read requires only the covariant `+T`, so a `@property` getter suffices:

```console
$ optype infer "lambda x: x.spam"
[R](x: Has['spam', +R]) -> R
```

!!! tip "Reading the `+`/`-` signs"

    A read is `+` and a write is `-`: covariant for values that come out, contravariant
    for values that go in.

For a called attribute the sign moves to the callable's return type, where the covariance
applies: `Has['spam', () -> +R]` is the method `def spam(self) -> R`:

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

Builtin objects use their importable `types` name: `ModuleType` for a module, `CodeType`
for a code object, and so on.

```console
$ optype infer "def f(): return f.__code__"
() -> CodeType
```

## Classes

The class of a value stays linked to the parameter it came from:

```console
$ optype infer "lambda x: type(x)"
[T](x: T) -> type[T]

$ optype infer "lambda x: type(next(x))"
[R](x: CanNext[R]) -> type[R]
```

Instantiating that class gives the parameter type back:

```console
$ optype infer "lambda x: type(x)()"
[T](x: T) -> T
```

An attribute accessed on the class itself is wrapped in `ClassVar`, matching a protocol
member declared as `spam: ClassVar[...]`.

```console
$ optype infer "lambda x: type(x).spam"
[R](x: Has['spam', ClassVar[+R]]) -> R

$ optype infer "def f(x): type(x).spam = 1"
(x: Has['spam', ClassVar[-Literal[1]]]) -> None

$ optype infer "def f(x): del type(x).spam"
(x: Has['spam', ClassVar]) -> None
```

A concrete class is parameterized when it resolves by name; a local class stays a bare
`type`. Because `type` is covariant, a subclass is absorbed by its parent:

```console
$ optype infer "lambda: bool"
() -> type[bool]

$ optype infer "lambda x: int if x else bool"
(x: CanBool) -> type[int]
```

A subscripted generic keeps its arguments, builtin or user-defined:

```console
$ optype infer "lambda: list[int]"
() -> type[list[int]]

$ optype infer "lambda: dict[str, int]"
() -> type[dict[str, int]]
```

Unions and `Callable` have no `type[...]` form, and use `TypeForm` instead
([PEP 747](https://peps.python.org/pep-0747/)):

```console
$ optype infer "lambda: int | str"
() -> TypeForm[int | str]

$ optype infer "from collections.abc import Callable
lambda: Callable[[int], str]"
() -> TypeForm[(int) -> str]
```

An origin or argument that doesn't resolve by name keeps the bare `GenericAlias`.

## Branches

Both sides of a conditional are explored. The parameter has to satisfy every branch (an
intersection), and the return type is the union of the branch results:

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

A comparison chain short-circuits, which makes it a branch too: a falsy first comparison
is returned as-is, and only a truthy one evaluates the second. Here `0 < x` reflects to
`x.__gt__`:

```console
$ optype infer "lambda x: 0 < x < 10"
[R, R2: CanBool](x: CanGt[Literal[0], R2 & CanBool] & CanLt[Literal[10], R]) -> R | R2
```

Predicates are assumed stable within a single call: repeating one on the same operand
agrees instead of branching again. A self-contradicting guard is then never satisfiable,
and its body is left unexplored:

```console
$ optype infer "lambda x: x.foo() if (x and not x) else x"
[T: CanBool](x: T) -> T
```

Distinct operands stay independent. `a in x` and `b in x` still branch separately.

## Variadic parameters

A `*args` parameter that is only passed around as a whole is inferred as a
[PEP 646](https://peps.python.org/pep-0646/) variadic type parameter, `*Ts`:

```console
$ optype infer "lambda *args: args"
[*Ts](*args: *Ts) -> tuple[*Ts]

$ optype infer "lambda *args: (1, *args)"
[*Ts](*args: *Ts) -> tuple[Literal[1], *Ts]
```

A variadic type parameter cannot express operations on individual elements. Those
elements then share one inferred element type, as does every value of `**kwargs`:

```console
$ optype infer "lambda *args: args[0] + args[1]"
[T: CanAdd[T, R], R](*args: T) -> R
[T: CanRAdd[T, R], R](*args: T) -> R

$ optype infer "lambda **kwargs: kwargs"
[T](**kwargs: T) -> dict[str, T]
```

A variadic spread into a callable becomes a single `*tuple[T, ...]`, giving the callable
the arity of the variadic itself. The same holds for element spreads, as in `map`, whose
signature and `strict` flag require Python 3.14:

```console
$ optype infer "lambda f, *args: f(*args)"
[T, R](f: (*tuple[T, ...]) -> R, *args: T) -> R

$ optype infer "map"
[T, U, R]((T, *tuple[U, ...]) -> R, CanIter[CanNext[T]], *iterables: CanIter[CanNext[U]], strict: CanBool = False) -> map[R]
```

## Parameter defaults

When the parameter has its own typevar, its default becomes a
[PEP 696](https://peps.python.org/pep-0696/) type parameter default:

```console
$ optype infer "def f(x=0): return x"
[T = Literal[0]](x: T = 0) -> T
```

A parameter without its own typevar shows the default inline instead:

```console
$ optype infer "def f(x=0): return str(x)"
(x: CanStr = 0) -> str
```

If omitting the argument behaves differently, for instance when the function branches on
the default, the call without it is reported as a separate overload:

```console
$ optype infer "def f(x=None): return [] if x is None else x"
(x: None = None) -> list[Never]
[T: ~None](x: T) -> T
```

The `~None` complement makes the overloads disjoint: the first one covers `f()` and
`f(None)`, and the second one everything else. Like `&`, the `~` is not valid Python;
in practice it's fine to omit it, as overloads are matched in order anyway.

A Python 3.15+ `sentinel` is its own type per
[PEP 661](https://peps.python.org/pep-0661/), spelled as its declared name. The common
sentinel-default pattern therefore works just like the `None` default above:

```console
$ optype infer "MISSING = sentinel('MISSING')
def f(x=MISSING): return [] if x is MISSING else x"
(x: MISSING = MISSING) -> list[Never]
[T: ~MISSING](x: T) -> T
```

## Deprecation

A call that emits a [PEP 702](https://peps.python.org/pep-0702/) `DeprecationWarning` is
reported with an `@deprecated` marker carrying the warning's message. This covers both the
`@deprecated` decorator and a plain `warnings.warn(..., DeprecationWarning)` in the body:

```console
$ optype infer "from warnings import deprecated
@deprecated('Use bar instead')
def foo(x): return x + 1"
@deprecated('Use bar instead')
[R](x: CanAdd[Literal[1], R]) -> R
```

The warning has to actually fire: only the overloads that raise it are marked. Omitting a
default that takes a quiet branch leaves that overload unmarked:

```console
$ optype infer "import warnings
def f(x, y=None):
    if y is None: return x + 1
    warnings.warn('y is deprecated', DeprecationWarning)
    return x + y"
[R](x: CanAdd[Literal[1], R], y: None = None) -> R
@deprecated('y is deprecated')
[T: ~None, R](x: CanAdd[T, R], y: T) -> R
@deprecated('y is deprecated')
[T, R](x: T, y: CanRAdd[T, R] & ~None) -> R
```

Only `DeprecationWarning` is recognized (not `PendingDeprecationWarning`, nor a
`@deprecated(..., category=...)` override), and a numpy ufunc is never marked.

## Methods

Anything callable can be inferred, including builtins (like `math.sqrt` above), callable
instances, and unbound method descriptors. Positional-only parameters cannot be passed by
keyword, and appear as a bare type without their name. The `self` of a method descriptor
requires a real instance of its defining class, which is reported as that concrete type:

```console
$ optype infer "str.upper"
(str) -> str

$ optype infer "dict.get"
[T = None](dict, CanHash, T = None) -> T
```

A builtin that only accepts concrete values for a defaulted parameter pins that
parameter to its default, while the others stay structural:

```console
$ optype infer "str.split"
(str, sep: None = None, maxsplit: CanIndex = -1) -> list[Never]
```

## Context managers

A `with` statement requires `__enter__` and `__exit__` together, which `optype` combines
as `CanWith`. The `__exit__` result is unused and stays `object`:

```pycon
>>> def f(x):
...     with x as y:
...         return y
>>> infer(f)
'[R](x: CanWith[R, object]) -> R'
```

## Async

Coroutine functions are run to completion. `await`, `async with`, and `async for` are
inferred like their synchronous counterparts:

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

Generators are lazy, and get iterated to collect the types they yield:

```console
$ optype infer "def f(n): yield from range(n)"
(n: CanIndex) -> Generator[int]

$ optype infer "def f(): yield None; yield 1"
() -> Generator[None | int]

$ optype infer "async def f(xs): return (x async for x in xs)"
[R](xs: CanAIter[CanANext[CanAwait[R]]]) -> AsyncGenerator[R]
```

Lazy builtin iterators (`map`, `filter`, `zip`, and `enumerate`) are iterated the same
way, and a callable argument is explored through them:

```console
$ optype infer "lambda x: map(str, x)"
(x: CanIter[CanNext[CanStr]]) -> map[str]

$ optype infer "lambda f, x: map(f, x)"
[T, R](f: (T) -> R, x: CanIter[CanNext[T]]) -> map[R]
```

`zip` tuples its iterables together, so the element type tracks how many there are: a
fixed call is a fixed-length tuple, and the variadic builtin is a homogeneous
`tuple[R, ...]`:

```console
$ optype infer "lambda x, y: zip(x, y)"
[R, R2](x: CanIter[CanNext[R]], y: CanIter[CanNext[R2]]) -> zip[tuple[R, R2]]

$ optype infer "zip"
[R](*iterables: CanIter[CanNext[R]], strict: CanBool = False) -> zip[tuple[R, ...]]
```

A callable that compares the elements it iterates, like `sorted` or `max`, requires the
comparison too:

```console
$ optype infer "lambda xs: sorted(xs)"
[R: CanLt[R, CanBool]](xs: CanIter[CanNext[R]]) -> list[R]

$ optype infer "lambda xs: max(xs)"
[R: CanGt[R, CanBool]](xs: CanIter[CanNext[R]]) -> R
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

A fixed unpack of three or more targets cannot be satisfied and raises an `InferError`.

## Returned functions

A returned function is explored too, and gets the same signature syntax:

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

This also covers builtins, method descriptors, and `functools.partial`, including from
within a returned container:

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

A recursive function (factory) has no expressible type, and stays an opaque
`FunctionType`. So does one with variadic parameters:

```console
$ optype infer "def f(x): return f"
(x: object) -> FunctionType
```

## Containers

Element types are tracked through the containers that hold them. A typevar can appear at
any depth, on either side of the signature:

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

A container that holds itself is a recursive type, reported as a typevar bounded by its
own structure:

```console
$ optype infer "def f(): x = []; x.append(x); return x"
[R: list[R]]() -> R
```

## Unions

A union member that is a subtype of another member is absorbed into it, following runtime
subclass relations such as `bool <: int` and `FileNotFoundError <: OSError`:

```console
$ optype infer "def f(): yield True; yield 1"
() -> Generator[int]

$ optype infer "def f(): yield FileNotFoundError(); yield OSError()"
() -> Generator[OSError]
```

PEP 484's `int <: float <: complex` numeric tower has no runtime counterpart, so it is
not applied:

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

Empty containers are the exception. The empty instance belongs to every container with
the same base, and is absorbed even when invariant:

```console
$ optype infer "lambda x: [None] if x else []"
(x: CanBool) -> list[None]
```

## Template strings

A t-string (Python 3.14+) is a `string.templatelib.Template`, and an `Interpolation`
tracks the type of its interpolated value:

```console
$ optype infer "lambda: t''"
() -> string.templatelib.Template

$ optype infer "lambda x: t'{x}'.interpolations[0]"
[T](x: T) -> string.templatelib.Interpolation[T]
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
extends to anything it returns: a returned lazy iterator is iterated, and a returned
function is called. If that call raises, its type falls back to an opaque `FunctionType`
instead of erroring.

A single-parameter function that dispatches on an attribute's presence (`hasattr`,
`getattr` with a default, or `try`/`except AttributeError`) covers both branches. The
attribute is tolerated rather than required, which widens the parameter past `Has[...]`.
If the return ignores the attribute's value, one overload covers both branches and
returns their union, which is why a `hasattr` predicate accepts any object and returns
`bool`. If the present branch returns the value, that overload sits above an `object`
fallback:

```pycon
>>> print(infer(lambda x: hasattr(x, "spam")))
(x: object) -> bool
>>> print(infer(lambda x: 0 if hasattr(x, "spam") else 1))
(x: object) -> int
>>> print(infer(lambda x: getattr(x, "spam", None)))
[R](x: Has['spam', +R]) -> R
(x: object) -> object
```

Anything else keeps the strict baseline that requires the attribute: more than one
parameter, several presence-tests at once, a present branch that needs more than the
attribute itself, or an absent branch that cannot be explored (as with `dict`).

When `infer` can't handle the input, it raises `InferError` (a `NotImplementedError`
subclass). This happens for operations without a matching protocol, like an attribute
access whose name is not statically known (a computed `getattr`); for arguments that
aren't callable to begin with; and for builtins whose signature cannot be introspected,
such as `iter`, `max`, or `type` itself (`type(x)` within a function is fine). A function
that never runs to completion, such as `lambda: 0 / 0`, raises `InferError` chained from
the triggering exception (`__cause__`).

Variadic parameters are explored with a bounded number of arguments. A function that
needs more raises an `InferError`. They are never empty, and `len(args)` reports however
many were used.

The number of explored branches is capped. A function with many of them gets a signature
covering only the explored ones, plus an `InferWarning` naming the gap and the affected
call form. The CLI prints it to standard error; `strict=True` raises an `InferError`
instead.

Only operations that go through a dunder method can be observed. A parameter passed to
`id()`, `isinstance()`, or an identity check (`is`) is reported as `object` instead of
its real requirement. Branches on a derived integer are invisible for the same reason:
`len`, `int`, and `index` return a small placeholder value, so `len(x) > 5` is never
explored.

Exploration can also crash the interpreter or hang. Either way you get an `InferError`
naming where it stopped, after at most a minute.

!!! warning "Generic bounds"

    An inferred typevar bound can itself be generic, such as `[T: CanAdd[T, R], R]` where
    `T`'s bound references `T` and `R`. Python's type system does not currently support
    generic typevar bounds, so these signatures are not always valid Python.
