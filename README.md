<h1 align="center">optype</h1>

<p align="center">
    Building blocks for precise & flexible type hints.
</p>

<p align="center">
    <a href="https://github.com/jorenham/optype/actions?query=workflow%3ACI">
        <img
            alt="Continuous Integration"
            src="https://github.com/jorenham/optype/workflows/CI/badge.svg"
        />
    </a>
    <a href="https://pypi.org/project/optype/">
        <img
            alt="PyPI"
            src="https://img.shields.io/pypi/v/optype?style=flat"
        />
    </a>
    <a href="https://github.com/jorenham/optype">
        <img
            alt="Python Versions"
            src="https://img.shields.io/pypi/pyversions/optype?style=flat"
        />
    </a>
    <a href="https://github.com/jorenham/optype">
        <img
            alt="License"
            src="https://img.shields.io/github/license/jorenham/optype?style=flat"
        />
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img
            alt="Ruff"
            src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"
        >
    </a>
    <a href="https://github.com/microsoft/pyright">
        <img
            alt="Checked with pyright"
            src="https://microsoft.github.io/pyright/img/pyright_badge.svg"
        />
    </a>
</p>

---

> [!WARNING]
> The API is not stable; use it at your own risk.


## Installation

Optype is available as [`optype`](OPTYPE) on PyPI:

```shell
pip install optype
```

[OPTYPE]: https://pypi.org/project/optype/

## Getting started

<!-- TODO -->
...


## Reference

All [typing protocols](PC) here live in the root `optype` namespace.
They are [runtime-checkable](RC) so that you can do e.g.
`isinstance('snail', optype.CanAdd)`, in case you want to check whether
`snail` implements `__add__`.

> [!NOTE]
> It is bad practice to use a [`typing.Protocol`](PC) as base class for your
> implementation. Because of [`@typing.runtime_checkable`](RC), you can use
> `isinstance` either way.

Unlike`collections.abc`, `optype`'s protocols aren't abstract base classes,
i.e. they don't extend `abc.ABC`, only `typing.Protocol`.
This allows the `optype` protocols to be used as building blocks for `.pyi`
type stubs.

[PC]: https://typing.readthedocs.io/en/latest/spec/protocol.html
[RC]: https://typing.readthedocs.io/en/latest/spec/protocol.html#runtime-checkable-decorator-and-narrowing-types-by-isinstance


### Elementary interfaces for the special methods

Single-method [`typing.Protocol`](PC) definitions for each of the "special
methods", also known as "magic"- or "dunder" methods.
See the [Python docs](SM) for details.

[SM]: https://docs.python.org/3/reference/datamodel.html#special-method-names


#### Strict type conversion

The return type of these special methods is *invariant*. Python will raise an
error if some other (sub)type is returned.
This is why these `optype` interfaces don't accept generic type arguments.

**Builtin type constructors:**

| Type         | Signature                      | Expression      |
| ------------ | ------------------------------ | --------------- |
| `CanBool`    | `__bool__(self) -> bool`       | `bool(self)`    |
| `CanInt`     | `__int__(self) -> int`         | `int(self)`     |
| `CanFloat`   | `__float__(self) -> float`     | `float(self)`   |
| `CanComplex` | `__complex__(self) -> complex` | `complex(self)` |
| `CanBytes`   | `__bytes__(self) -> bytes`     | `bytes(self)`   |
| `CanStr`     | `__str__(self) -> str`         | `str(self)`     |

**Other builtin functions:**

| Type            | Signature                      | Expression   |
| --------------- | ------------------------------ | ------------ |
| `CanRepr`       | `__repr__(self) -> str`        | `repr(self)` |
| `CanHash`       | `__hash__(self) -> int`        | `hash(self)` |
| `CanLen`        | `__len__(self) -> int`         | `len(self)`  |
| `CanLengthHint` | `__length_hint__(self) -> int` | [docs](LH)   |
| `CanIndex`      | `__index__(self) -> int`       | [docs](IX)   |


[LH]: https://docs.python.org/3/reference/datamodel.html#object.__length_hint__
[IX]: https://docs.python.org/3/reference/datamodel.html#object.__index__


#### Comparisons operators

Generally these methods return a `bool`. But in theory, anything can be
returned (even if it doesn't implement `__bool__`).

| Type           | Signature                  | Expression  | Expr. Reflected |
| -------------- | -------------------------- | ----------- | --------------- |
| `CanLt[X, Y]`  | `__lt__(self, x: X) -> Y`  | `self < x`  | `x > self`      |
| `CanLe[X, Y]`  | `__le__(self, x: X) -> Y`  | `self <= x` | `x >= self`     |
| `CanGe[X, Y]`  | `__ge__(self, x: X) -> Y`  | `self >= x` | `x <= self`     |
| `CanGt[X, Y]`  | `__gt__(self, x: X) -> Y`  | `self > x`  | `x < self`      |
| `CanEq[X, Y]`  | `__eq__(self, x: X) -> Y`  | `self == x` | `x == self`     |
| `CanNe[X, Y]`  | `__ne__(self, x: X) -> Y`  | `self != x` | `x != self`     |


#### Rounding

| Type               | Signature                     | Expression            |
| ------------------ | ----------------------------- | --------------------- |
| `CanRound1[Y1]`    | `__round__(self) -> Y1`       | `round(self)`         |
| `CanRound2[N, Y2]` | `__round__(self, n: N) -> Y2` | `round(self, n: N)`   |

For convenience, `optype` also provides their intersection type
`CanRound[N, Y1, Y2] =: CanRound1[Y1] & CanRound2[N, Y2]`, whose signature
overloads those of the `CanRound1` and `CanRound2`.

For instance, `float` is a `CanRound[int, int, float]` and `int` a
`CanRound[int, int, int]`.

| Type           | Signature               | Expression         |
| -------------- | ----------------------- | ------------------ |
| `CanTrunc[Y]`  | `__trunc__(self) -> Y`  | `math.trunc(self)` |
| `CanFloor[Y]`  | `__floor__(self) -> Y`  | `math.floor(self)` |
| `CanCeil[Y]`   | `__ceil__(self) -> Y`   | `math.ceil(self)`  |

Note that the type parameter `Y` is unbounded, because technically these
methods can return any type.


#### Arithmetic and bitwise operators

**Unary:**

| Type           | Signature               | Expression         |
| -------------- | ----------------------- | ------------------ |
| `CanPos[Y]`    | `__pos__(self) -> Y`    | `+self`            |
| `CanNeg[Y]`    | `__neg__(self) -> Y`    | `-self`            |
| `CanInvert[Y]` | `__invert__(self) -> Y` | `~self`            |
| `CanAbs[Y]`    | `__abs__(self) -> Y`    | `abs(self)`        |


**Binary:**

| Type                | Signature                       | Expression        |
| ------------------- | ------------------------------- | ----------------- |
| `CanAdd[X, Y]`      | `__add__(self, x: X) -> Y`      | `self + x`        |
| `CanSub[X, Y]`      | `__sub__(self, x: X) -> Y`      | `self - x`        |
| `CanMul[X, Y]`      | `__mul__(self, x: X) -> Y`      | `self * x`        |
| `CanMatmul[X, Y]`   | `__matmul__(self, x: X) -> Y`   | `self @ x`        |
| `CanTruediv[X, Y]`  | `__truediv__(self, x: X) -> Y`  | `self / x`        |
| `CanFloordiv[X, Y]` | `__floordiv__(self, x: X) -> Y` | `self // x`       |
| `CanMod[X, Y]`      | `__mod__(self, x: X) -> Y`      | `self % x`        |
| `CanDivmod[X, Y]`   | `__divmod__(self, x: X) -> Y`   | `divmod(self, x)` |
| `CanPow2[X, Y]`     | `__pow__(self, x: X) -> Y`      | `self ** x`       |
| `CanPow3[X, M, Y]`  | `__pow__(self, x: X, m: M) -> Y`| `pow(self, x, m)` |
| `CanLshift[X, Y]`   | `__lshift__(self, x: X) -> Y`   | `self << x`       |
| `CanRshift[X, Y]`   | `__rshift__(self, x: X) -> Y`   | `self >> x`       |
| `CanAnd[X, Y]`      | `__and__(self, x: X) -> Y`      | `self & x`        |
| `CanXor[X, Y]`      | `__xor__(self, x: X) -> Y`      | `self ^ x`        |
| `CanOr[X, Y]`       | `__or__(self, x: X) -> Y`       | `self \| x`       |

Additionally, there is the intersection type
`CanPow[X, M, Y2, Y3] =: CanPow2[X, Y2] & CanPow3[X, M, Y3]`, whose signature
overloads those of the `CanPow2` and `CanPow3`.

**Binary (reflected):**

| Type                 | Signature                        | Expression        |
| -------------------- | -------------------------------- | ----------------- |
| `CanRAdd[X, Y]`      | `__radd__(self, x: X) -> Y`      | `x + self`        |
| `CanRSub[X, Y]`      | `__rsub__(self, x: X) -> Y`      | `x - self`        |
| `CanRMul[X, Y]`      | `__rmul__(self, x: X) -> Y`      | `x * self`        |
| `CanRMatmul[X, Y]`   | `__rmatmul__(self, x: X) -> Y`   | `x @ self`        |
| `CanRTruediv[X, Y]`  | `__rtruediv__(self, x: X) -> Y`  | `x / self`        |
| `CanRFloordiv[X, Y]` | `__rfloordiv__(self, x: X) -> Y` | `x // self`       |
| `CanRMod[X, Y]`      | `__rmod__(self, x: X) -> Y`      | `x % self`        |
| `CanRDivmod[X, Y]`   | `__rdivmod__(self, x: X) -> Y`   | `divmod(x, self)` |
| `CanRPow[X, Y]`      | `__rpow__(self, x: X) -> Y`      | `x ** self`       |
| `CanRLshift[X, Y]`   | `__rlshift__(self, x: X) -> Y`   | `x << self`       |
| `CanRRshift[X, Y]`   | `__rrshift__(self, x: X) -> Y`   | `x >> self`       |
| `CanRAnd[X, Y]`      | `__rand__(self, x: X) -> Y`      | `x & self`        |
| `CanRXor[X, Y]`      | `__rxor__(self, x: X) -> Y`      | `x ^ self`        |
| `CanROr[X, Y]`       | `__ror__(self, x: X) -> Y`       | `x \| self`       |


**Binary (augmented / in-place):**

| Type                 | Signature                        | Expression   |
| -------------------- | -------------------------------- | ------------ |
| `CanIAdd[X, Y]`      | `__iadd__(self, x: X) -> Y`      | `self += x`  |
| `CanISub[X, Y]`      | `__isub__(self, x: X) -> Y`      | `self -= x`  |
| `CanIMul[X, Y]`      | `__imul__(self, x: X) -> Y`      | `self *= x`  |
| `CanIMatmul[X, Y]`   | `__imatmul__(self, x: X) -> Y`   | `self @= x`  |
| `CanITruediv[X, Y]`  | `__itruediv__(self, x: X) -> Y`  | `self /= x`  |
| `CanIFloordiv[X, Y]` | `__ifloordiv__(self, x: X) -> Y` | `self //= x` |
| `CanIMod[X, Y]`      | `__imod__(self, x: X) -> Y`      | `self %= x`  |
| `CanIPow[X, Y]`      | `__ipow__(self, x: X) -> Y`      | `self **= x` |
| `CanILshift[X, Y]`   | `__ilshift__(self, x: X) -> Y`   | `self <<= x` |
| `CanIRshift[X, Y]`   | `__irshift__(self, x: X) -> Y`   | `self >>= x` |
| `CanIAnd[X, Y]`      | `__iand__(self, x: X) -> Y`      | `self &= x`  |
| `CanIXor[X, Y]`      | `__ixor__(self, x: X) -> Y`      | `self ^= x`  |
| `CanIOr[X, Y]`       | `__ior__(self, x: X) -> Y`       | `self \|= x` |

<!-- TODO:  implement separate ternary pow -->


### Containers

| Type               | Signature                              | Expression    |
| ------------------ | -------------------------------------- | ------------- |
| `CanContains[K]`   | `__contains__(self, k: K) -> bool`     | `x in self`   |
| `CanDelitem[K]`    | `__delitem__(self, k: K) -> None`      | `del self[k]` |
| `CanGetitem[K, V]` | `__getitem__(self, k: K) -> V`         | `self[k]`     |
| `CanMissing[K, V]` | `__missing__(self, k: K) -> V`         | [docs](GM)    |
| `CanSetitem[K, V]` | `__setitem__(self, k: K, v: V) -> None`| `self[k] = v` |


[GM]: https://docs.python.org/3/reference/datamodel.html#object.__missing__


### Iteration

**Sync**

| Type                       | Signature                 | Expression       |
| -------------------------- | ------------------------- | ---------------- |
| `CanNext[V]`               | `__next__(self) -> V`     | `next(self)`     |
| `CanIter[Y: CanNext[Any]]` | `__iter__(self) -> Y`     | `iter(self)`     |
| `CanReversed[Y]` (*)       | `__reversed__(self) -> Y` | `reversed(self)` |

(*) Although not strictly required, `Y@CanReversed` should be iterable.

**Async**

| Type                         | Signature               | Expression       |
| ---------------------------- | ----------------------- | ---------------- |
| `CanAnext[V]` (**)           | `__anext__(self) -> V`  | `anext(self)`    |
| `CanAiter[Y: CanAnext[Any]]` | `__aiter__(self) -> Y`  | `aiter(self)`    |

(**) Although not strictly required, `V@CanAnext` should be an `Awaitable`.


### Generic interfaces for builtins

#### `optype.Slice[A, B, S]`

A generic interface of the builin
[`slice`](https://docs.python.org/3/library/functions.html#slice) object.

**Signatures**:

- `(B) -> Slice[None, B, None]`
- `(A, B) -> Slice[A, B, None]`
- `(A, B, S) -> Slice[A, B, S]`

these are valid for the `slice(start?, stop, step?)` constructor,
and for the extended indexing syntax `_[start? : stop? : step?]` (the `?`
denotes an optional parameter).

**Decorators**:
- `@typing.runtime_checkable`
- `@typing.final`


## Future plans

- Build a replacement for the `operator` standard library, with
  runtime-accessible type annotations
- Protocols for numpy's dunder methods
- Backport to Python 3.11 and 3.10
