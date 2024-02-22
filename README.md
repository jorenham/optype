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
> The API is not stable; use at your own risk.


## Installation

Optype is available as [`optype`](https://pypi.org/project/optype/) on PyPI:

```shell
pip install optype
```

## Getting started

*Coming soon*


## Reference

### Elementary interfaces for the special methods

Single-method `typing.Protocol` definitions for each of the "special methods",
also known as "magic"- or "dunder"- methods. See the [python documentation
](https://docs.python.org/3/reference/datamodel.html#special-method-names) for
details.


####

#### Comparisons

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


#### Arithmetic and bitwise operators

**Unary:**

| Type           | Signature               | Expression         |
| -------------- | ----------------------- | ------------------ |
| `CanPos[Y]`    | `__pos__(self) -> Y`    | `+self`            |
| `CanNeg[Y]`    | `__neg__(self) -> Y`    | `-self`            |
| `CanInvert[Y]` | `__invert__(self) -> Y` | `~self`            |
| `CanAbs[Y]`    | `__abs__(self) -> Y`    | `abs(self)`        |
| `CanRound0[Y]` | `__round__(self) -> Y`  | `round(self)`      |
| `CanTrunc[Y]`  | `__trunc__(self) -> Y`  | `math.trunc(self)` |
| `CanFloor[Y]`  | `__floor__(self) -> Y`  | `math.floor(self)` |
| `CanCeil[Y]`   | `__ceil__(self) -> Y`   | `math.ceil(self)`  |


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
| `CanPow[X, Y]`      | `__pow__(self, x: X) -> Y`      | `self ** x`       |
| `CanLshift[X, Y]`   | `__lshift__(self, x: X) -> Y`   | `self << x`       |
| `CanRshift[X, Y]`   | `__rshift__(self, x: X) -> Y`   | `self >> x`       |
| `CanAnd[X, Y]`      | `__and__(self, x: X) -> Y`      | `self & x`        |
| `CanXor[X, Y]`      | `__xor__(self, x: X) -> Y`      | `self ^ x`        |
| `CanOr[X, Y]`       | `__or__(self, x: X) -> Y`       | `self \| x`       |

<!-- TODO; implement separate binary round -->


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


**Ternary**

<!-- TODO:  implement separate ternary pow -->
...

### Containers

<!-- TODO: CanContains, CanGetitem, CanSetitem, CanDelitem, CanMissing  -->


### Iteration

**Sync**

<!-- TODO -->
...

**Async**

<!-- TODO -->
...



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
