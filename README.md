<h1 align="center">optype</h1>

<p align="center">
    Building blocks for precise & flexible type hints.
</p>

<p align="center">
    <a href="https://pypi.org/project/optype/">
        <img
            alt="optype - PyPI"
            src="https://img.shields.io/pypi/v/optype?style=flat"
        />
    </a>
    <a href="https://github.com/jorenham/optype">
        <img
            alt="optype - Python Versions"
            src="https://img.shields.io/pypi/pyversions/optype?style=flat"
        />
    </a>
    <a href="https://github.com/jorenham/optype">
        <img
            alt="optype - license"
            src="https://img.shields.io/github/license/jorenham/optype?style=flat"
        />
    </a>
</p>
<p align="center">
    <a href="https://github.com/jorenham/optype/actions?query=workflow%3ACI">
        <img
            alt="optype - CI"
            src="https://github.com/jorenham/optype/workflows/CI/badge.svg"
        />
    </a>
    <a href="https://github.com/pre-commit/pre-commit">
        <img
            alt="optype - pre-commit"
            src="https://img.shields.io/badge/pre--commit-enabled-orange?logo=pre-commit"
        />
    </a>
    <a href="https://detachhead.github.io/basedpyright">
        <img
            alt="optype - basedpyright"
            src="https://img.shields.io/badge/basedpyright-checked-42b983"
        />
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img
            alt="optype - ruff"
            src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"
        />
    </a>
</p>

---

## Installation

Optype is available as [`optype`][OPTYPE] on PyPI:

```shell
pip install optype
```

For optional [NumPy][NUMPY] support, it is recommended to use the
`numpy` extra.
This ensures that the installed `numpy` version is compatible with
`optype`, following [NEP 29][NEP29] and [SPEC 0][SPEC0].

```shell
pip install "optype[numpy]"
```

See the [`optype.numpy` docs](#numpy) for more info.

[OPTYPE]: https://pypi.org/project/optype/
[NUMPY]: https://github.com/numpy/numpy

## Example

Let's say you're writing a `twice(x)` function, that evaluates `2 * x`.
Implementing it is trivial, but what about the type annotations?

Because `twice(2) == 4`, `twice(3.14) == 6.28` and `twice('I') = 'II'`, it
might seem like a good idea to type it as `twice[T](x: T) -> T: ...`.
However, that wouldn't include cases such as `twice(True) == 2` or
`twice((42, True)) == (42, True, 42, True)`, where the input- and output types
differ.
Moreover, `twice` should accept *any* type with a custom `__rmul__` method
that accepts `2` as argument.

This is where `optype` comes in handy, which has single-method protocols for
*all* the builtin special methods.
For `twice`, we can use `optype.CanRMul[T, R]`, which, as the name suggests,
is a protocol with (only) the `def __rmul__(self, lhs: T) -> R: ...` method.
With this, the `twice` function can written as:

<table>
<tr>
<th width="415px">Python 3.10</th>
<th width="415px">Python 3.12+</th>
</tr>
<tr>
<td>

```python
from typing import Literal
from typing import TypeAlias, TypeVar
from optype import CanRMul

R = TypeVar('R')
Two: TypeAlias = Literal[2]
RMul2: TypeAlias = CanRMul[Two, R]

def twice(x: RMul2[R]) -> R:
    return 2 * x
```

</td>
<td>

```python
from typing import Literal
from optype import CanRMul

type Two = Literal[2]
type RMul2[R] = CanRMul[Two, R]

def twice[R](x: RMul2[R]) -> R:
    return 2 * x
```

</td>
</tr>
</table>

But what about types that implement `__add__` but not `__radd__`?
In this case, we could return `x * 2` as fallback (assuming commutativity).
Because the `optype.Can*` protocols are runtime-checkable, the revised
`twice2` function can be compactly written as:

<table>
<tr>
<th width="415px">Python 3.10</th>
<th width="415px">Python 3.12+</th>
</tr>
<tr>
<td>

```python
from optype import CanMul

Mul2: TypeAlias = CanMul[Two, R]
CMul2: TypeAlias = Mul2[R] | RMul2[R]

def twice2(x: CMul2[R]) -> R:
    if isinstance(x, CanRMul):
        return 2 * x
    else:
        return x * 2
```

</td>
<td>

```python
from optype import CanMul

type Mul2[R] = CanMul[Two, R]
type CMul2[R] = Mul2[R] | RMul2[R]

def twice2[R](x: CMul2[R]) -> R:
    if isinstance(x, CanRMul):
        return 2 * x
    else:
        return x * 2
```

</td>
</tr>
</table>

See [`examples/twice.py`](examples/twice.py) for the full example.

## Reference

The API of `optype` is flat; a single `import optype` is all you need.

There are four flavors of things that live within `optype`,

-
    `optype.Can{}` types describe *what can be done* with it.
    For instance, any `CanAbs[T]` type can be used as argument to the `abs()`
    builtin function with return type `T`. Most `Can{}` implement a single
    special method, whose name directly matched that of the type. `CanAbs`
    implements `__abs__`, `CanAdd` implements `__add__`, etc.
-
    `optype.Has{}` is the analogue of `Can{}`, but for special *attributes*.
    `HasName` has a `__name__` attribute, `HasDict` has a `__dict__`, etc.
-
    `optype.Does{}` describe the *type of operators*.
    So `DoesAbs` is the type of the `abs({})` builtin function,
    and `DoesPos` the type of the `+{}` prefix operator.
-
    `optype.do_{}` are the correctly-typed implementations of `Does{}`. For
    each `do_{}` there is a `Does{}`, and vice-versa.
    So `do_abs: DoesAbs` is the typed alias of `abs({})`,
    and `do_pos: DoesPos` is a typed version of `operator.pos`.
    The `optype.do_` operators are more complete than `operators`,
    have runtime-accessible type annotations, and have names you don't
    need to know by heart.

The reference docs are structured as follows:

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Core functionality](#core-functionality)
    - [Builtin type conversion](#builtin-type-conversion)
    - [Rich relations](#rich-relations)
    - [Binary operations](#binary-operations)
    - [Reflected operations](#reflected-operations)
    - [Inplace operations](#inplace-operations)
    - [Unary operations](#unary-operations)
    - [Rounding](#rounding)
    - [Callables](#callables)
    - [Iteration](#iteration)
    - [Awaitables](#awaitables)
    - [Async Iteration](#async-iteration)
    - [Containers](#containers)
    - [Attributes](#attributes)
    - [Context managers](#context-managers)
    - [Descriptors](#descriptors)
    - [Buffer types](#buffer-types)
- [Standard libs](#standard-libs)
    - [`copy`](#copy)
    - [`pickle`](#pickle)
    - [`dataclasses`](#dataclasses)
- [NumPy](#numpy)
    - [Arrays](#arrays)
        - [`Array`](#array)
        - [`AnyArray`](#anyarray)
        - [`CanArray*`](#canarray)
        - [`HasArray*`](#hasarray)
    - [Scalars](#scalars)
        - [`Scalar`](#scalar)
        - [`Any*Value`](#anyvalue)
        - [`Any*Type`](#anytype)
    - [Data type objects](#data-type-objects)
        - [`DType`](#dtype)
        - [`HasDType`](#hasdtype)
        - [`AnyDType`](#anydtype)
    - [Universal functions](#universal-functions)
        - [`AnyUFunc`](#anyufunc)
        - [`CanArrayUFunc`](#canarrayufunc)

<!-- TOC end -->

### Core functionality

All [typing protocols][PC] here live in the root `optype` namespace.
They are [runtime-checkable][RC] so that you can do e.g.
`isinstance('snail', optype.CanAdd)`, in case you want to check whether
`snail` implements `__add__`.

Unlike`collections.abc`, `optype`'s protocols aren't abstract base classes,
i.e. they don't extend `abc.ABC`, only `typing.Protocol`.
This allows the `optype` protocols to be used as building blocks for `.pyi`
type stubs.

[PC]: https://typing.readthedocs.io/en/latest/spec/protocol.html
[RC]: https://typing.readthedocs.io/en/latest/spec/protocol.html#runtime-checkable-decorator-and-narrowing-types-by-isinstance

#### Builtin type conversion

The return type of these special methods is *invariant*. Python will raise an
error if some other (sub)type is returned.
This is why these `optype` interfaces don't accept generic type arguments.

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>complex(_)</code></td>
        <td><code>do_complex</code></td>
        <td><code>DoesComplex</code></td>
        <td><code>__complex__</code></td>
        <td><code>CanComplex</code></td>
    </tr>
    <tr>
        <td><code>float(_)</code></td>
        <td><code>do_float</code></td>
        <td><code>DoesFloat</code></td>
        <td><code>__float__</code></td>
        <td><code>CanFloat</code></td>
    </tr>
    <tr>
        <td><code>int(_)</code></td>
        <td><code>do_int</code></td>
        <td><code>DoesInt</code></td>
        <td><code>__int__</code></td>
        <td><code>CanInt[R: int = int]</code></td>
    </tr>
    <tr>
        <td><code>bool(_)</code></td>
        <td><code>do_bool</code></td>
        <td><code>DoesBool</code></td>
        <td><code>__bool__</code></td>
        <td><code>CanBool[R: bool = bool]</code></td>
    </tr>
    <tr>
        <td><code>bytes(_)</code></td>
        <td><code>do_bytes</code></td>
        <td><code>DoesBytes</code></td>
        <td><code>__bytes__</code></td>
        <td><code>CanBytes[R: bytes = bytes]</code></td>
    </tr>
    <tr>
        <td><code>str(_)</code></td>
        <td><code>do_str</code></td>
        <td><code>DoesStr</code></td>
        <td><code>__str__</code></td>
        <td><code>CanStr[R: str = str]</code></td>
    </tr>
</table>

> [!NOTE]
> The `Can*` interfaces of the types that can used as `typing.Literal`
> accept an optional type parameter `R`.
> This can be used to indicate a literal return type,
> for surgically precise typing, e.g. `None`, `True`, and `42` are
> instances of `CanBool[Literal[False]]`, `CanInt[Literal[1]]`, and
> `CanStr[Literal['42']]`, respectively.

These formatting methods are allowed to return instances that are a subtype
of the `str` builtin. The same holds for the `__format__` argument.
So if you're a 10x developer that wants to hack Python's f-strings, but only
if your type hints are spot-on; `optype` is you friend.

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>repr(_)</code></td>
        <td><code>do_repr</code></td>
        <td><code>DoesRepr</code></td>
        <td><code>__repr__</code></td>
        <td><code>CanRepr[R: str = str]</code></td>
    </tr>
    <tr>
        <td><code>format(_, x)</code></td>
        <td><code>do_format</code></td>
        <td><code>DoesFormat</code></td>
        <td><code>__format__</code></td>
        <td><code>CanFormat[T: str = str, R: str = str]</code></td>
    </tr>
</table>

Additionally, `optype` provides protocols for types with (custom) *hash* or
*index* methods:

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>hash(_)</code></td>
        <td><code>do_hash</code></td>
        <td><code>DoesHash</code></td>
        <td><code>__hash__</code></td>
        <td><code>CanHash</code></td>
    </tr>
    <tr>
        <td>
            <code>_.__index__()</code>
            (<a href="https://docs.python.org/3/reference/datamodel.html#object.__index__">docs</a>)
        </td>
        <td><code>do_index</code></td>
        <td><code>DoesIndex</code></td>
        <td><code>__index__</code></td>
        <td><code>CanIndex[R: int = int]</code></td>
    </tr>
</table>

#### Rich relations

The "rich" comparison special methods often return a `bool`.
However, instances of any type can be returned (e.g. a numpy array).
This is why the corresponding `optype.Can*` interfaces accept a second type
argument for the return type, that defaults to `bool` when omitted.
The first type parameter matches the passed method argument, i.e. the
right-hand side operand, denoted here as `x`.

<table>
    <tr>
        <th colspan="4" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>reflected</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>_ == x</code></td>
        <td><code>x == _</code></td>
        <td><code>do_eq</code></td>
        <td><code>DoesEq</code></td>
        <td><code>__eq__</code></td>
        <td><code>CanEq[T = object, R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ != x</code></td>
        <td><code>x != _</code></td>
        <td><code>do_ne</code></td>
        <td><code>DoesNe</code></td>
        <td><code>__ne__</code></td>
        <td><code>CanNe[T = object, R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ < x</code></td>
        <td><code>x > _</code></td>
        <td><code>do_lt</code></td>
        <td><code>DoesLt</code></td>
        <td><code>__lt__</code></td>
        <td><code>CanLt[T, R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ <= x</code></td>
        <td><code>x >= _</code></td>
        <td><code>do_le</code></td>
        <td><code>DoesLe</code></td>
        <td><code>__le__</code></td>
        <td><code>CanLe[T, R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ > x</code></td>
        <td><code>x < _</code></td>
        <td><code>do_gt</code></td>
        <td><code>DoesGt</code></td>
        <td><code>__gt__</code></td>
        <td><code>CanGt[T, R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ >= x</code></td>
        <td><code>x <= _</code></td>
        <td><code>do_ge</code></td>
        <td><code>DoesGe</code></td>
        <td><code>__ge__</code></td>
        <td><code>CanGe[T, R = bool]</code></td>
    </tr>
</table>

#### Binary operations

In the [Python docs][NT], these are referred to as "arithmetic operations".
But the operands aren't limited to numeric types, and because the
operations aren't required to be commutative, might be non-deterministic, and
could have side-effects.
Classifying them "arithmetic" is, at the very least, a bit of a stretch.

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>_ + x</code></td>
        <td><code>do_add</code></td>
        <td><code>DoesAdd</code></td>
        <td><code>__add__</code></td>
        <td><code>CanAdd[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ - x</code></td>
        <td><code>do_sub</code></td>
        <td><code>DoesSub</code></td>
        <td><code>__sub__</code></td>
        <td><code>CanSub[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ * x</code></td>
        <td><code>do_mul</code></td>
        <td><code>DoesMul</code></td>
        <td><code>__mul__</code></td>
        <td><code>CanMul[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ @ x</code></td>
        <td><code>do_matmul</code></td>
        <td><code>DoesMatmul</code></td>
        <td><code>__matmul__</code></td>
        <td><code>CanMatmul[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ / x</code></td>
        <td><code>do_truediv</code></td>
        <td><code>DoesTruediv</code></td>
        <td><code>__truediv__</code></td>
        <td><code>CanTruediv[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ // x</code></td>
        <td><code>do_floordiv</code></td>
        <td><code>DoesFloordiv</code></td>
        <td><code>__floordiv__</code></td>
        <td><code>CanFloordiv[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ % x</code></td>
        <td><code>do_mod</code></td>
        <td><code>DoesMod</code></td>
        <td><code>__mod__</code></td>
        <td><code>CanMod[T, R]</code></td>
    </tr>
    <tr>
        <td><code>divmod(_, x)</code></td>
        <td><code>do_divmod</code></td>
        <td><code>DoesDivmod</code></td>
        <td><code>__divmod__</code></td>
        <td><code>CanDivmod[T, R]</code></td>
    </tr>
    <tr>
        <td>
            <code>_ ** x</code><br/>
            <code>pow(_, x)</code>
        </td>
        <td><code>do_pow/2</code></td>
        <td><code>DoesPow</code></td>
        <td><code>__pow__</code></td>
        <td>
            <code>CanPow2[T, R]</code><br/>
            <code>CanPow[T, None, R, Never]</code>
        </td>
    </tr>
    <tr>
        <td><code>pow(_, x, m)</code></td>
        <td><code>do_pow/3</code></td>
        <td><code>DoesPow</code></td>
        <td><code>__pow__</code></td>
        <td>
            <code>CanPow3[T, M, R]</code><br/>
            <code>CanPow[T, M, Never, R]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ << x</code></td>
        <td><code>do_lshift</code></td>
        <td><code>DoesLshift</code></td>
        <td><code>__lshift__</code></td>
        <td><code>CanLshift[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ >> x</code></td>
        <td><code>do_rshift</code></td>
        <td><code>DoesRshift</code></td>
        <td><code>__rshift__</code></td>
        <td><code>CanRshift[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ & x</code></td>
        <td><code>do_and</code></td>
        <td><code>DoesAnd</code></td>
        <td><code>__and__</code></td>
        <td><code>CanAnd[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ ^ x</code></td>
        <td><code>do_xor</code></td>
        <td><code>DoesXor</code></td>
        <td><code>__xor__</code></td>
        <td><code>CanXor[T, R]</code></td>
    </tr>
    <tr>
        <td><code>_ | x</code></td>
        <td><code>do_or</code></td>
        <td><code>DoesOr</code></td>
        <td><code>__or__</code></td>
        <td><code>CanOr[T, R]</code></td>
    </tr>
</table>

> [!NOTE]
> Because `pow()` can take an optional third argument, `optype`
> provides separate interfaces for `pow()` with two and three arguments.
> Additionally, there is the overloaded intersection type
> `CanPow[T, M, R, RM] =: CanPow2[T, R] & CanPow3[T, M, RM]`, as interface
> for types that can take an optional third argument.

#### Reflected operations

For the binary infix operators above, `optype` additionally provides
interfaces with *reflected* (swapped) operands, e.g. `__radd__` is a reflected
`__add__`.
They are named like the original, but prefixed with `CanR` prefix, i.e.
`__name__.replace('Can', 'CanR')`.

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>x + _</code></td>
        <td><code>do_radd</code></td>
        <td><code>DoesRAdd</code></td>
        <td><code>__radd__</code></td>
        <td><code>CanRAdd[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x - _</code></td>
        <td><code>do_rsub</code></td>
        <td><code>DoesRSub</code></td>
        <td><code>__rsub__</code></td>
        <td><code>CanRSub[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x * _</code></td>
        <td><code>do_rmul</code></td>
        <td><code>DoesRMul</code></td>
        <td><code>__rmul__</code></td>
        <td><code>CanRMul[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x @ _</code></td>
        <td><code>do_rmatmul</code></td>
        <td><code>DoesRMatmul</code></td>
        <td><code>__rmatmul__</code></td>
        <td><code>CanRMatmul[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x / _</code></td>
        <td><code>do_rtruediv</code></td>
        <td><code>DoesRTruediv</code></td>
        <td><code>__rtruediv__</code></td>
        <td><code>CanRTruediv[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x // _</code></td>
        <td><code>do_rfloordiv</code></td>
        <td><code>DoesRFloordiv</code></td>
        <td><code>__rfloordiv__</code></td>
        <td><code>CanRFloordiv[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x % _</code></td>
        <td><code>do_rmod</code></td>
        <td><code>DoesRMod</code></td>
        <td><code>__rmod__</code></td>
        <td><code>CanRMod[T, R]</code></td>
    </tr>
    <tr>
        <td><code>divmod(x, _)</code></td>
        <td><code>do_rdivmod</code></td>
        <td><code>DoesRDivmod</code></td>
        <td><code>__rdivmod__</code></td>
        <td><code>CanRDivmod[T, R]</code></td>
    </tr>
    <tr>
        <td>
            <code>x ** _</code><br/>
            <code>pow(x, _)</code>
        </td>
        <td><code>do_rpow</code></td>
        <td><code>DoesRPow</code></td>
        <td><code>__rpow__</code></td>
        <td><code>CanRPow[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x << _</code></td>
        <td><code>do_rlshift</code></td>
        <td><code>DoesRLshift</code></td>
        <td><code>__rlshift__</code></td>
        <td><code>CanRLshift[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x >> _</code></td>
        <td><code>do_rrshift</code></td>
        <td><code>DoesRRshift</code></td>
        <td><code>__rrshift__</code></td>
        <td><code>CanRRshift[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x & _</code></td>
        <td><code>do_rand</code></td>
        <td><code>DoesRAnd</code></td>
        <td><code>__rand__</code></td>
        <td><code>CanRAnd[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x ^ _</code></td>
        <td><code>do_rxor</code></td>
        <td><code>DoesRXor</code></td>
        <td><code>__rxor__</code></td>
        <td><code>CanRXor[T, R]</code></td>
    </tr>
    <tr>
        <td><code>x | _</code></td>
        <td><code>do_ror</code></td>
        <td><code>DoesROr</code></td>
        <td><code>__ror__</code></td>
        <td><code>CanROr[T, R]</code></td>
    </tr>
</table>

> [!NOTE]
> `CanRPow` corresponds to `CanPow2`; the 3-parameter "modulo" `pow` does not
> reflect in Python.
>
> According to the relevant [python docs][RPOW]:
> > Note that ternary `pow()` will not try calling `__rpow__()` (the coercion
> > rules would become too complicated).

[RPOW]: https://docs.python.org/3/reference/datamodel.html#object.__rpow__

#### Inplace operations

Similar to the reflected ops, the inplace/augmented ops are prefixed with
`CanI`, namely:

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>types</th>
    </tr>
    <tr>
        <td><code>_ += x</code></td>
        <td><code>do_iadd</code></td>
        <td><code>DoesIAdd</code></td>
        <td><code>__iadd__</code></td>
        <td>
            <code>CanIAdd[T, R]</code><br>
            <code>CanIAddSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ -= x</code></td>
        <td><code>do_isub</code></td>
        <td><code>DoesISub</code></td>
        <td><code>__isub__</code></td>
        <td>
            <code>CanISub[T, R]</code><br>
            <code>CanISubSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ *= x</code></td>
        <td><code>do_imul</code></td>
        <td><code>DoesIMul</code></td>
        <td><code>__imul__</code></td>
        <td>
            <code>CanIMul[T, R]</code><br>
            <code>CanIMulSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ @= x</code></td>
        <td><code>do_imatmul</code></td>
        <td><code>DoesIMatmul</code></td>
        <td><code>__imatmul__</code></td>
        <td>
            <code>CanIMatmul[T, R]</code><br>
            <code>CanIMatmulSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ /= x</code></td>
        <td><code>do_itruediv</code></td>
        <td><code>DoesITruediv</code></td>
        <td><code>__itruediv__</code></td>
        <td>
            <code>CanITruediv[T, R]</code><br>
            <code>CanITruedivSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ //= x</code></td>
        <td><code>do_ifloordiv</code></td>
        <td><code>DoesIFloordiv</code></td>
        <td><code>__ifloordiv__</code></td>
        <td>
            <code>CanIFloordiv[T, R]</code><br>
            <code>CanIFloordivSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ %= x</code></td>
        <td><code>do_imod</code></td>
        <td><code>DoesIMod</code></td>
        <td><code>__imod__</code></td>
        <td>
            <code>CanIMod[T, R]</code><br>
            <code>CanIModSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ **= x</code></td>
        <td><code>do_ipow</code></td>
        <td><code>DoesIPow</code></td>
        <td><code>__ipow__</code></td>
        <td>
            <code>CanIPow[T, R]</code><br>
            <code>CanIPowSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ <<= x</code></td>
        <td><code>do_ilshift</code></td>
        <td><code>DoesILshift</code></td>
        <td><code>__ilshift__</code></td>
        <td>
            <code>CanILshift[T, R]</code><br>
            <code>CanILshiftSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ >>= x</code></td>
        <td><code>do_irshift</code></td>
        <td><code>DoesIRshift</code></td>
        <td><code>__irshift__</code></td>
        <td>
            <code>CanIRshift[T, R]</code><br>
            <code>CanIRshiftSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ &= x</code></td>
        <td><code>do_iand</code></td>
        <td><code>DoesIAnd</code></td>
        <td><code>__iand__</code></td>
        <td>
            <code>CanIAnd[T, R]</code><br>
            <code>CanIAndSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ ^= x</code></td>
        <td><code>do_ixor</code></td>
        <td><code>DoesIXor</code></td>
        <td><code>__ixor__</code></td>
        <td>
            <code>CanIXor[T, R]</code><br>
            <code>CanIXorSelf[T]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ |= x</code></td>
        <td><code>do_ior</code></td>
        <td><code>DoesIOr</code></td>
        <td><code>__ior__</code></td>
        <td>
            <code>CanIOr[T, R]</code><br>
            <code>CanIOrSelf[T]</code>
        </td>
    </tr>
</table>

These inplace operators usually return itself (after some in-place mutation).
But unfortunately, it currently isn't possible to use `Self` for this (i.e.
something like `type MyAlias[T] = optype.CanIAdd[T, Self]` isn't allowed).
So to help ease this unbearable pain, `optype` comes equipped with ready-made
aliases for you to use. They bear the same name, with an additional `*Self`
suffix, e.g. `optype.CanIAddSelf[T]`.

#### Unary operations

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>types</th>
    </tr>
    <tr>
        <td><code>+_</code></td>
        <td><code>do_pos</code></td>
        <td><code>DoesPos</code></td>
        <td><code>__pos__</code></td>
        <td>
            <code>CanPos[R]</code><br>
            <code>CanPosSelf</code>
        </td>
    </tr>
    <tr>
        <td><code>-_</code></td>
        <td><code>do_neg</code></td>
        <td><code>DoesNeg</code></td>
        <td><code>__neg__</code></td>
        <td>
            <code>CanNeg[R]</code><br>
            <code>CanNegSelf</code>
        </td>
    </tr>
    <tr>
        <td><code>~_</code></td>
        <td><code>do_invert</code></td>
        <td><code>DoesInvert</code></td>
        <td><code>__invert__</code></td>
        <td>
            <code>CanInvert[R]</code><br>
            <code>CanInvertSelf</code>
        </td>
    </tr>
    <tr>
        <td><code>abs(_)</code></td>
        <td><code>do_abs</code></td>
        <td><code>DoesAbs</code></td>
        <td><code>__abs__</code></td>
        <td>
            <code>CanAbs[R]</code><br>
            <code>CanAbsSelf</code>
        </td>
    </tr>
</table>

#### Rounding

The `round()` built-in function takes an optional second argument.
From a typing perspective, `round()` has two overloads, one with 1 parameter,
and one with two.
For both overloads, `optype` provides separate operand interfaces:
`CanRound1[R]` and `CanRound2[T, RT]`.
Additionally, `optype` also provides their (overloaded) intersection type:
`CanRound[T, R, RT] = CanRound1[R] & CanRound2[T, RT]`.

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>round(_)</code></td>
        <td><code>do_round/1</code></td>
        <td><code>DoesRound</code></td>
        <td><code>__round__/1</code></td>
        <td><code>CanRound1[T = int]</code><br/></td>
    </tr>
    <tr>
        <td><code>round(_, n)</code></td>
        <td><code>do_round/2</code></td>
        <td><code>DoesRound</code></td>
        <td><code>__round__/2</code></td>
        <td><code>CanRound2[T = int, RT = float]</code><br/></td>
    </tr>
    <tr>
        <td><code>round(_, n=...)</code></td>
        <td><code>do_round</code></td>
        <td><code>DoesRound</code></td>
        <td><code>__round__</code></td>
        <td><code>CanRound[T = int, R = int, RT = float]</code></td>
    </tr>
</table>

For example, type-checkers will mark the following code as valid (tested with
pyright in strict mode):

```python
x: float = 3.14
x1: CanRound1[int] = x
x2: CanRound2[int, float] = x
x3: CanRound[int, int, float] = x
```

Furthermore, there are the alternative rounding functions from the
[`math`][MATH] standard library:

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>math.trunc(_)</code></td>
        <td><code>do_trunc</code></td>
        <td><code>DoesTrunc</code></td>
        <td><code>__trunc__</code></td>
        <td><code>CanTrunc[R = int]</code></td>
    </tr>
    <tr>
        <td><code>math.floor(_)</code></td>
        <td><code>do_floor</code></td>
        <td><code>DoesFloor</code></td>
        <td><code>__floor__</code></td>
        <td><code>CanFloor[R = int]</code></td>
    </tr>
    <tr>
        <td><code>math.ceil(_)</code></td>
        <td><code>do_ceil</code></td>
        <td><code>DoesCeil</code></td>
        <td><code>__ceil__</code></td>
        <td><code>CanCeil[R = int]</code></td>
    </tr>
</table>

Almost all implementations use `int` for `R`.
In fact, if no type for `R` is specified, it will default in `int`.
But technially speaking, these methods can be made to return anything.

[MATH]: https://docs.python.org/3/library/math.html
[NT]: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

#### Callables

Unlike `operator`, `optype` provides the operator for callable objects:
`optype.do_call(f, *args. **kwargs)`.

`CanCall` is similar to `collections.abc.Callable`, but is runtime-checkable,
and doesn't use esoteric hacks.

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>_(*args, **kwargs)</code></td>
        <td><code>do_call</code></td>
        <td><code>DoesCall</code></td>
        <td><code>__call__</code></td>
        <td><code>CanCall[**Pss, R]</code></td>
    </tr>
</table>

> [!NOTE]
> Pyright (and probably other typecheckers) tend to accept
> `collections.abc.Callable` in more places than `optype.CanCall`.
> This could be related to the lack of co/contra-variance specification for
> `typing.ParamSpec` (they should almost always be contravariant, but
> currently they can only be invariant).
>
> In case you encounter such a situation, please open an issue about it, so we
> can investigate further.

#### Iteration

The operand `x` of `iter(_)` is within Python known as an *iterable*, which is
what `collections.abc.Iterable[V]` is often used for (e.g. as base class, or
for instance checking).

The `optype` analogue is `CanIter[R]`, which as the name suggests,
also implements `__iter__`. But unlike `Iterable[V]`, its type parameter `R`
binds to the return type of `iter(_) -> R`. This makes it possible to annotate
the specific type of the *iterable* that `iter(_)` returns. `Iterable[V]` is
only able to annotate the type of the iterated value. To see why that isn't
possible, see [python/typing#548](https://github.com/python/typing/issues/548).

The `collections.abc.Iterator[V]` is even more awkward; it is a subtype of
`Iterable[V]`. For those familiar with `collections.abc` this might come as a
surprise, but an iterator only needs to implement `__next__`, `__iter__` isn't
needed. This means that the `Iterator[V]` is unnecessarily restrictive.
Apart from that being theoretically "ugly", it has significant performance
implications, because the time-complexity of `isinstance` on a
`typing.Protocol` is $O(n)$, with the $n$ referring to the amount of members.
So even if the overhead of the inheritance and the `abc.ABC` usage is ignored,
`collections.abc.Iterator` is twice as slow as it needs to be.

That's one of the (many) reasons that `optype.CanNext[V]` and
`optype.CanNext[V]` are the better alternatives to `Iterable` and `Iterator`
from the abracadabra collections. This is how they are defined:

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>next(_)</code></td>
        <td><code>do_next</code></td>
        <td><code>DoesNext</code></td>
        <td><code>__next__</code></td>
        <td><code>CanNext[V]</code></td>
    </tr>
    <tr>
        <td><code>iter(_)</code></td>
        <td><code>do_iter</code></td>
        <td><code>DoesIter</code></td>
        <td><code>__iter__</code></td>
        <td><code>CanIter[R: CanNext[Any]]</code></td>
    </tr>
</table>

For the sake of compatibility with `collections.abc`, there is
`optype.CanIterSelf[V]`, which is a protocol whose `__iter__` returns
`typing.Self`, as well as a `__next__` method that returns `T`.
I.e. it is equivalent to `collections.abc.Iterator[V]`, but without the `abc`
nonsense.

#### Awaitables

The `optype` is almost the same as `collections.abc.Awaitable[R]`, except
that `optype.CanAwait[R]` is a pure interface, whereas `Awaitable` is
also an abstract base class (making it absolutely useless when writing stubs).

<table>
    <tr>
        <th align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>await _</code></td>
        <td><code>__await__</code></td>
        <td><code>CanAwait[R]</code></td>
    </tr>
</table>

#### Async Iteration

Yes, you guessed it right; the abracadabra collections made the exact same
mistakes for the async iterablors (or was it "iteramblers"...?).

But fret not; the `optype` alternatives are right here:

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>anext(_)</code></td>
        <td><code>do_anext</code></td>
        <td><code>DoesANext</code></td>
        <td><code>__anext__</code></td>
        <td><code>CanANext[V]</code></td>
    </tr>
    <tr>
        <td><code>aiter(_)</code></td>
        <td><code>do_aiter</code></td>
        <td><code>DoesAIter</code></td>
        <td><code>__aiter__</code></td>
        <td><code>CanAIter[R: CanAnext[Any]]</code></td>
    </tr>
</table>

But wait, shouldn't `V` be a `CanAwait`? Well, only if you don't want to get
fired...
Technically speaking, `__anext__` can return any type, and `anext` will pass
it along without nagging (instance checks are slow, now stop bothering that
liberal). For details, see the discussion at [python/typeshed#7491][AN].
Just because something is legal, doesn't mean it's a good idea (don't eat the
yellow snow).

Additionally, there is `optype.CanAIterSelf[R]`, with both the
`__aiter__() -> Self` and the `__anext__() -> V` methods.

[AN]: https://github.com/python/typeshed/pull/7491

#### Containers

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>len(_)</code></td>
        <td><code>do_len</code></td>
        <td><code>DoesLen</code></td>
        <td><code>__len__</code></td>
        <td><code>CanLen[R: int = int]</code></td>
    </tr>
    <tr>
        <td>
            <code>_.__length_hint__()</code>
            (<a href="https://docs.python.org/3/reference/datamodel.html#object.__length_hint__">docs</a>)
        </td>
        <td><code>do_length_hint</code></td>
        <td><code>DoesLengthHint</code></td>
        <td><code>__length_hint__</code></td>
        <td><code>CanLengthHint[R: int = int]</code></td>
    </tr>
    <tr>
        <td><code>_[k]</code></td>
        <td><code>do_getitem</code></td>
        <td><code>DoesGetitem</code></td>
        <td><code>__getitem__</code></td>
        <td><code>CanGetitem[K, V]</code></td>
    </tr>
    <tr>
        <td>
            <code>_.__missing__()</code>
            (<a href="https://docs.python.org/3/reference/datamodel.html#object.__missing__">docs</a>)
        </td>
        <td><code>do_missing</code></td>
        <td><code>DoesMissing</code></td>
        <td><code>__missing__</code></td>
        <td><code>CanMissing[K, D]</code></td>
    </tr>
    <tr>
        <td><code>_[k] = v</code></td>
        <td><code>do_setitem</code></td>
        <td><code>DoesSetitem</code></td>
        <td><code>__setitem__</code></td>
        <td><code>CanSetitem[K, V]</code></td>
    </tr>
    <tr>
        <td><code>del _[k]</code></td>
        <td><code>do_delitem</code></td>
        <td><code>DoesDelitem</code></td>
        <td><code>__delitem__</code></td>
        <td><code>CanDelitem[K]</code></td>
    </tr>
    <tr>
        <td><code>k in _</code></td>
        <td><code>do_contains</code></td>
        <td><code>DoesContains</code></td>
        <td><code>__contains__</code></td>
        <td><code>CanContains[K = object]</code></td>
    </tr>
    <tr>
        <td><code>reversed(_)</code></td>
        <td><code>do_reversed</code></td></td>
        <td><code>DoesReversed</code></td>
        <td><code>__reversed__</code></td>
        <td>
            <code>CanReversed[R]</code>, or<br>
            <code>CanSequence[K: CanIndex, V]</code>
        </td>
    </tr>
</table>

Because `CanMissing[K, D]` generally doesn't show itself without
`CanGetitem[K, V]` there to hold its hand, `optype` conveniently stitched them
together as `optype.CanGetMissing[K, V, D=V]`.

Similarly, there is `optype.CanSequence[K: CanIndex | slice, V]`, which is the
combination of both `CanLen` and `CanItem[I, V]`, and serves as a more
specific and flexible `collections.abc.Sequence[V]`.

#### Attributes

<table>
    <tr>
        <th colspan="3" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td>
            <code>v = _.k</code> or<br/>
            <code>v = getattr(_, k)</code>
        </td>
        <td><code>do_getattr</code></td>
        <td><code>DoesGetattr</code></td>
        <td><code>__getattr__</code></td>
        <td><code>CanGetattr[K: str = str, V = Any]</code></td>
    </tr>
    <tr>
        <td>
            <code>_.k = v</code> or<br/>
            <code>setattr(_, k, v)</code>
        </td>
        <td><code>do_setattr</code></td>
        <td><code>DoesSetattr</code></td>
        <td><code>__setattr__</code></td>
        <td><code>CanSetattr[K: str = str, V = Any]</code></td>
    </tr>
    <tr>
        <td>
            <code>del _.k</code> or<br/>
            <code>delattr(_, k)</code>
        </td>
        <td><code>do_delattr</code></td>
        <td><code>DoesDelattr</code></td>
        <td><code>__delattr__</code></td>
        <td><code>CanDelattr[K: str = str]</code></td>
    </tr>
    <tr>
        <td><code>dir(_)</code></td>
        <td><code>do_dir</code></td>
        <td><code>DoesDir</code></td>
        <td><code>__dir__</code></td>
        <td><code>CanDir[R: CanIter[CanIterSelf[str]]]</code></td>
    </tr>
</table>

#### Context managers

Support for the `with` statement.

<table>
    <tr>
        <th align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>method(s)</td>
        <th>type(s)</th>
    </tr>
    <tr>
        <td></td>
        <td><code>__enter__</code></td>
        <td>
            <code>CanEnter[C]</code>, or
            <code>CanEnterSelf</code>
        </td>
    </tr>
    <tr>
        <td></td>
        <td><code>__exit__</code></td>
        <td>
            <code>CanExit[R = None]</code>
        </td>
    </tr>
    <tr>
        <td><code>with _ as c:</code></td>
        <td>
            <code>__enter__</code>, and <br>
            <code>__exit__</code>
        </td>
        <td>
            <code>CanWith[C, R=None]</code>, or<br>
            <code>CanWithSelf[R=None]</code>
        </td>
    </tr>
</table>

`CanEnterSelf` and `CanWithSelf` are (runtime-checkable) aliases for
`CanEnter[Self]` and `CanWith[Self, R]`, respectively.

For the `async with` statement the interfaces look very similar:

<table>
    <tr>
        <th align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>method(s)</td>
        <th>type(s)</th>
    </tr>
    <tr>
        <td></td>
        <td><code>__aenter__</code></td>
        <td>
            <code>CanAEnter[C]</code>, or<br>
            <code>CanAEnterSelf</code>
        </td>
    </tr>
    <tr>
        <td></td>
        <td><code>__aexit__</code></td>
        <td><code>CanAExit[R=None]</code></td>
    </tr>
    <tr>
        <td><code>async with _ as c:</code></td>
        <td>
            <code>__aenter__</code>, and<br>
            <code>__aexit__</code>
        </td>
        <td>
            <code>CanAsyncWith[C, R=None]</code>, or<br>
            <code>CanAsyncWithSelf[R=None]</code>
        </td>
    </tr>
</table>

#### Descriptors

Interfaces for [descriptors](https://docs.python.org/3/howto/descriptor.html).

<table>
    <tr>
        <th align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td>
            <code>v: V = T().d</code><br/>
            <code>vt: VT = T.d</code>
        </td>
        <td><code>__get__</code></td>
        <td><code>CanGet[T: object, V, VT = V]</code></td>
    </tr>
    <tr>
        <td><code>T().k = v</code></td>
        <td><code>__set__</code></td>
        <td><code>CanSet[T: object, V]</code></td>
    </tr>
    <tr>
        <td><code>del T().k</code></td>
        <td><code>__delete__</code></td>
        <td><code>CanDelete[T: object]</code></td>
    </tr>
    <tr>
        <td><code>class T: d = _</code></td>
        <td><code>__set_name__</code></td>
        <td><code>CanSetName[T: object, N: str = str]</code></td>
    </tr>
</table>

#### Buffer types

Interfaces for emulating buffer types using the [buffer protocol][BP].

<table>
    <tr>
        <th align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>v = memoryview(_)</code></td>
        <td><code>__buffer__</code></td>
        <td><code>CanBuffer[T: int = int]</code></td>
    </tr>
    <tr>
        <td><code>del v</code></td>
        <td><code>__release_buffer__</code></td>
        <td><code>CanReleaseBuffer</code></td>
    </tr>
</table>

[BP]: https://docs.python.org/3/reference/datamodel.html#python-buffer-protocol

### Standard libs

#### `copy`

For the [`copy`][CP] standard library, `optype` provides the following
interfaces:

<table>
    <tr>
        <th align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>copy.copy(_)</code></td>
        <td><code>__copy__</code></td>
        <td><code>CanCopy[R: object]</code></td>
    </tr>
    <tr>
        <td><code>copy.deepcopy(_, memo={})</code></td>
        <td><code>__deepcopy__</code></td>
        <td><code>CanDeepcopy[R: object]</code></td>
    </tr>
    <tr>
        <td><code>copy.replace(_, **changes: V)</code> (Python 3.13+)</td>
        <td><code>__replace__</code></td>
        <td><code>CanReplace[V, R]</code></td>
    </tr>
</table>

And for convenience, there are the runtime-checkable aliases for all three
interfaces, with `R` bound to `Self`.
These are roughly equivalent to:

```python
type CanCopySelf = CanCopy[CanCopySelf]
type CanDeepcopySelf = CanDeepcopy[CanDeepcopySelf]
type CanReplaceSelf[V] = CanReplace[V, CanReplaceSelf[V]]
```

[CP]: https://docs.python.org/3/library/copy.html

#### `pickle`

For the [`pickle`][PK] standard library, `optype` provides the following
interfaces:

[PK]: https://docs.python.org/3/library/pickle.html

<table>
    <tr>
        <th>method(s)</th>
        <th>signature (bound)</th>
        <th>type</th>
    </tr>
    <tr>
        <td><code>__reduce__</code></td>
        <td><code>() -> R</code></td>
        <td><code>CanReduce[R: str | tuple = str | tuple]</code></td>
    </tr>
    <tr>
        <td><code>__reduce_ex__</code></td>
        <td><code>(CanIndex) -> R</code></td>
        <td><code>CanReduceEx[R: str | tuple = str | tuple]</code></td>
    </tr>
    <tr>
        <td><code>__getstate__</code></td>
        <td><code>() -> S</code></td>
        <td><code>CanGetstate[S: object]</code></td>
    </tr>
    <tr>
        <td><code>__setstate__</code></td>
        <td><code>(S) -> None</code></td>
        <td><code>CanSetstate[S: object]</code></td>
    </tr>
    <tr>
        <td>
            <code>__getnewargs__</code><br>
            <code>__new__</code>
        </td>
        <td>
            <code>() -> tuple[*Vs]</code><br>
            <code>(*Vs) -> Self</code><br>
        </td>
        <td><code>CanGetnewargs[*Vs]</code></td>
    </tr>
    <tr>
        <td>
            <code>__getnewargs_ex__</code><br>
            <code>__new__</code>
        </td>
        <td>
            <code>() -> tuple[tuple[*Vs], dict[str, V]]</code><br>
            <code>(*Vs, **dict[str, V]) -> Self</code><br>
        </td>
        <td><code>CanGetnewargsEx[*Vs, V]</code></td>
    </tr>
</table>

#### `dataclasses`

For the [`dataclasses`][DC] standard library, `optype` provides the
`HasDataclassFields[V: Mapping[str, Field]]` interface.
It can conveniently be used to check whether a type or instance is a
dataclass, i.e. `isinstance(obj, optype.HasDataclassFields)`.

[DC]: https://docs.python.org/3/library/dataclasses.html

### NumPy

Optype supports both NumPy 1 and 2.
The current minimum supported version is `1.24`,
following [NEP 29][NEP29] and [SPEC 0][SPEC0].

When using `optype.numpy`, it is recommended to install `optype` with the
`numpy` extra, ensuring version compatibility:

```shell
pip install "optype[numpy]"
```

> [!NOTE]
> For the remainder of the `optype.numpy` docs, assume that the following
> import aliases are available.
>
> ```python
> from typing import Any, Literal
> import numpy as np
> import numpy.typing as npt
> import optype.numpy as onp
> ```
>
> For the sake of brevity and readability, the [PEP 695][PEP695] and
> [PEP 696][PEP696] type parameter syntax will be used, which is supported
> since Python 3.13.

#### Arrays

##### `Array`

Optype provides the generic `onp.Array` type alias for `np.ndarray`.
It is similar to `npt.NDArray`, but includes two (optional) type parameters:
one that matches the *shape type* (`ND: tuple[int, ...]`),
and one that matches the *scalar type* (`ST: np.generic`).
It is defined as:

```python
type Array[
    ND: tuple[int, ...] = tuple[int, ...],
    ST: np.generic = Any,
] = np.ndarray[ND, np.dtype[ST]]
```

Note that the shape type parameter `ND` matches the type of `np.ndarray.shape`,
and the scalar type parameter `ST` that of `np.ndarray.dtype.type`.

This way, a vector can be typed as `Array[tuple[int]]`, and a $2 \times 2$
matrix of integers as `Array[tuple[Literal[2], Literal[2]], np.integer[Any]]`.

##### `AnyArray`

Something that can be used to construct a numpy array is often referred to
as an *array-like* object, usually annotated with `npt.ArrayLike`.
But there are two main problems with `npt.ArrayLike`:

1. Its name strongly suggests that it *only* applies to arrays. However,
  "0-dimensional" are also included, i.e. "scalars" such as `bool`, and
  `complex`, but also `str`, since numpy considers unicode- and bytestrings
  to be  "scalars".
  So `a: npt.ArrayLike = 'array lie'` is a valid statement.
2. There is no way to narrow the allowed scalar-types, since it's not generic.
   So instances of `bytes` and arrays of `np.object_` are always included.

`AnyArray[ND, ST, PY]` doesn't have these problems through its (optional)
generic type parameters:

```python
type AnyArray[
    # shape type
    ND: tuple[int, ...] = tuple[int, ...],
    # numpy scalar type
    ST: np.generic = np.generic,
    # Python builtin scalar type
    # (note that `complex` includes `bool | int | float`)
    PT: complex | str | bytes = complex | str | bytes,
]
```

> [!NOTE]
> Unlike `npt.ArrayLike`, `onp.AnyArray` does not include the python scalars
> (`PT`) directly.

This makes it possible to correctly annotate e.g. a 1-d arrays-like of floats
as `a: onp.AnyArray[tuple[int], np.floating[Any], float]`.

##### `CanArray*`

<table>
    <tr>
<th>type signature</th>
        <th>method</th>
        <th>purpose</th>
    </tr>
    <tr>
<td>

```python
CanArray[
    ND: tuple[int, ...] = ...,
    ST: np.generic = ...,
]
```

</td>
        <td>
            <a href="https://numpy.org/doc/stable/user/basics.interoperability.html#the-array-method">
                <code>__array__()</code>
            </a>
        </td>
        <td>Turning itself into a numpy array.</td>
    </tr>
    <tr>
<td>

```python
CanArrayFunction[
    F: CanCall[..., Any] = ...,
    R: object = ...,
]
```

</td>
        <td>
            <a href="https://numpy.org/doc/stable/user/basics.interoperability.html#the-array-function-protocol">
                <code>__array_function__()</code>
            </a>
        </td>
        <td>
            Similar to how <code>T.__abs__()</code> implements
            <code>abs(T)</code>, but for arbitrary numpy callables
            (that aren't a ufunc).
        </td>
    </tr>
    <tr>
<td>

```python
CanArrayFinalize[
    T: object = ...,
]
```

</td>
        <td>
            <a href="https://numpy.org/doc/stable/user/c-info.beyond-basics.html#the-array-finalize-method">
                <code>__array_finalize__()</code>
            </a>
        </td>
        <td>
            Converting the return value of a numpy function back into an
            instance of the foreign object.
        </td>
    </tr>
    <tr>
<td>

```python
CanArrayWrap
```

</td>
        <td>
            <a href="https://numpy.org/doc/stable/user/c-info.beyond-basics.html#ndarray.__array_wrap__">
                <code>__array_wrap__()</code>
            </a>
        </td>
        <td>
            Takes a `np.ndarray` instance, and "wraps" it into itself, or some
            `ndarray` subtype.
        </td>
    </tr>
</table>

##### `HasArray*`

<table>
    <tr>
        <th><code>optype.numpy._</code></th>
        <th>attribute</th>
        <th>purpose</th>
    </tr>
    <tr>
<td>

```python
HasArrayInterface[
    V: Mapping[str, Any] = dict[str, Any],
]
```

</td>
        <td>
            <a href="https://numpy.org/doc/stable/reference/arrays.interface.html#python-side">
                <code>__array_interface__</code>
            </a>
        </td>
        <td>
            The array interface protocol (V3) is used to for efficient
            data-buffer sharing between array-like objects, and bridges the
            numpy C-api with the Python side.
        </td>
    </tr>
    <tr>
<td>

```python
HasArrayPriority
```

</td>
        <td>
            <a href="https://numpy.org/doc/stable/user/c-info.beyond-basics.html#the-array-priority-attribute">
                <code>__array_priority__</code>
            </a>
        </td>
        <td>
            In case an operation involves multiple sub-types, this value
            determines which one will be used as output type.
        </td>
    </tr>
</table>

#### Scalars

Optype considers the following numpy scalar types:

- *`np.generic`*
    - `np.bool_` (or `np.bool` with `numpy >= 2`)
    - `np.object_`
    - *`np.flexible`*
        - `np.void`
        - *`np.character`*
            - `np.bytes_`
            - `np.str_`
    - *`np.number[N: npt.NBitBase]`*
        - *`np.integer[N: npt.NBitBase]`*
            - *`np.unsignedinteger[N: npt.NBitBase]`*
                - `np.ubyte`
                - `np.ushort`
                - `np.uintc`
                - `np.uintp`
                - `np.ulong`
                - `np.ulonglong`
                - `np.uint{8,16,32,64}`
            - *`np.signedinteger[N: npt.NBitBase]`*
                - `np.byte`
                - `np.short`
                - `np.intc`
                - `np.intp`
                - `np.long`
                - `np.longlong`
                - `np.int{8,16,32,64}`
        - *`np.inexact[N: npt.NBitBase]`*
            - *`np.floating[N: npt.NBitBase]`*
                - `np.half`
                - `np.single`
                - `np.double`
                - `np.longdouble`
                - `np.float{16,32,64}`
            - *`np.complexfloating[N1: npt.NBitBase, N2: npt.NBitBase]`*
                - `np.csingle`
                - `np.cdouble`
                - `np.clongdouble`
                - `np.complex{64,128}`

See the [docs](https://numpy.org/doc/stable/reference/arrays.scalars.html)
for more info.

##### `Scalar`

The `optype.numpy.Scalar` interface is a generic runtime-checkable protocol,
that can be seen as a "more specific" `np.generic`, both in name, and from
a typing perspective.
Its signature looks like

```python
Scalar[
    # The "Python type", so that `Scalar.item() -> PT`.
    PT: object,
    # The "N-bits" type (without having to deal with`npt.NBitBase`).
    # It matches `SCalar.itemsize: NB`.
    NB: int = Any,
]
```

It can be used as e.g.

```python
are_birds_real: Scalar[bool, Literal[1]] = np.bool_(True)
the_answer: Scalar[int, Literal[2]] = np.uint16(42)
fine_structure_constant: Scalar[float, Literal[8]] = np.float64(1) / 137
```

> [!NOTE]
> The second type argument for `itemsize` can be omitted, which is equivalent
> to setting it to `Any`.

##### `Any*Value`

For every (standard) numpy scalar type (i.e. subtypes of `np.generic`), there
is the `optype.numpy.Any{}Value` alias (where `{}` should be replaced with the
title-cased name of the scalar, without potential trailing underscore).

So for `np.bool_` there's `onp.AnyBoolValue`,
for `np.uint8` there's `onp.AnyUInt8Value`, and
for `np.floating[N: npt.NBitBase]` there's `AnyFloating[N: npt.NBitBase]`.

> [!NOTE]
> The *extended-precision* scalar types (e.g. `np.int128`, `np.float96` and
> `np.complex512`) are not included, because their availability is
> platform-dependent.

When a value of type `Any{}Value` is passed to e.g. `np.array`,
the resulting `np.ndarray` will have a scalar type that matches
the corresponding `Any{}Value`.
For instance, passing `x: onp.AnyFloat64Value` as `np.array(x)` returns an
array of type `onp.Array[tuple[()], np.float64]`
(where `tuple[()]` implies that its shape is `()`).

Each `Any{}Value` contains at least the relevant `np.generic` subtype,
zero or more [`ctypes`][CTYPES] types, and
zero or more of the Python `builtins` types.

So for instance `type AnyUInt8 = np.uint8 | ct.c_uint8`, and
`type AnyCDouble = np.cdouble | complex`.

[CTYPES]: https://docs.python.org/3/library/ctypes.html

##### `Any*Type`

In the same way as `Any*Value`, there's a `Any*Type` for each of the numpy
scalar types.

These type aliases describe what's allowed to be passed to e.g. the
`np.dtype[ST: np.generic]` constructor, so that its scalar type `ST` matches
the one corresponding to the passed `Any*Type`.

So for example, if some `x: onp.UInt8` is passed to `np.dtype(x)`, then the
resulting type will be a `np.dtype[np.uint8]`.

This is useful when annotating an (e.g. numpy) function with a `dtype`
parameter, e.g. `np.arange`.
Then by using a `@typing.overload` for each of the allowed scalar types,
it's possible to annotate it in *the most specific way that's possible*,
whilst keeping the code readable and maintainable.

#### Data type objects

In NumPy, a *dtype* (data type) object, is an instance of the
`numpy.dtype[ST: np.generic]` type.
It's commonly used to convey metadata of a scalar type, e.g. within arrays.

##### `DType`

Because the type parameter of `np.dtype` isn't optional, it could be more
convenient to use the alias `optype.numpy.DType`, which is defined as:

```python
type DType[ST: np.generic = Any] = np.dtype[ST]
```

Apart from the "CamelCase" name, the only difference with `np.dtype` is that
the type parameter can be omitted, in which case it's equivalent to
`np.dtype[np.generic]`, but shorter.

##### `HasDType`

Many of numpy's public functions accept an (optional) `dtype` argument.
But here, the term "dtype" has a broader meaning, as it also accepts
(a subtype of) `np.generic`.
Additionally, any instance with a `dtype: DType` attribute is accepted.
The runtime-checkable interface for this is `optype.numpy.HasDType`, which is
roughly equivalent to the following definition:

```python
@runtime_checkable
class HasDType[DT: DType = DType](Protocol):
    dtype: Final[DT]
```

Since `np.ndarray` has a `dtype` attribute, it is a subtype of `HasDType`:

```pycon
>>> isinstance(np.array([42]), onp.HasDType)
True
```

##### `AnyDType`

All types that can be passed to the `np.dtype` constructor, as well as the
types of most `dtype` function parameters, are encapsulated within the
`optype.numpy.AnyDType` alias, i.e.:

```python
type AnyDType[ST: np.generic = Any] = type[ST] | DType[ST] | HasDType[DType[ST]]
```

> [!NOTE]
> NumPy's own `numpy.typing.DTypeLike` alias serves the same purpose as
> `AnyDType`.
> But `npt.DTypeLike` has several issues:
>
> - It's not generic (accepts no type parameter(s)), and cannot be narrowed to
>   allow for specific scalar types. Even though most functions don't accept
>   *all* possible scalar- and dtypes.
> - Its definition is maximally broad, e.g. `type[Any]`, and `str` are
>   included in its union.
>   So given some arbitrary function parameter `dtype: npt.DTypeLike`, passing
>   e.g. `dtype="Ceci n'est pas une dtype"` won't look like anything out of the
>   ordinary for your type checker.
>
> These issues aren't the case for `optype.numpy.AnyDType`.
> However, it (currently) isn't possible to pass scalar char-codes
> (e.g. `dtype='f8'`) or builtin python types (e.g. `dtype=int`) directly.
> If you really want to do so anyway, then just pass it to the
> `np.dtype()` constructor, e.g. `np.arange(42, dtype=np.dtype('f8'))`.

#### Universal functions

A large portion of numpy's public API consists of
[universal functions][UFUNC-DOC], i.e. (callable) instances of
[`np.ufunc`][UFUNC-REF].

> [!TIP]
> Custom ufuncs can be created using [`np.frompyfunc`][FROMPYFUNC], but also
> through a user-defined class that implements the required attributes and
> methods (i.e., duck typing).

##### `AnyUFunc`

But `np.ufunc` has a big issue; it accepts no type parameters.
This makes it very difficult to properly annotate its callable signature and
its literal attributes (e.g. `.nin` and `.identity`).

This is where `optype.numpy.AnyUFunc` comes into play:
It's a runtime-checkable generic typing protocol, that has been thoroughly
type- and unit-tested to ensure compatibility with all of numpy's ufunc
definitions.
Its generic type signature looks roughly like:

```python
AnyUFunc[
    # The type of the (bound) `__call__` method.
    Fn: Callable[..., Any] = Any,
    # The types of the `nin` and `nout` (readonly) attributes.
    # Within numpy these match either `Literal[1]` or `Literal[2]`.
    Nin: int = Any,
    Nout: int = Any,
    # The type of the `signature` (readonly) attribute;
    # Must be `None` unless this is a generalized ufunc (gufunc), e.g.
    # `np.matmul`.
    Sig: str | None = Any,
    # The type of the `identity` (readonly) attribute (used in `.reduce`).
    # Unless `Nin: Literal[2]`, `Nout: Literal[1]`, and `Sig: None`,
    # this should always be `None`.
    # Note that `complex` also includes `bool | int | float`.
    Id: complex | str | bytes | None = Any,
]
```

> [!NOTE]
> Unfortunately, the extra callable methods of `np.ufunc` (`at`, `reduce`,
> `reduceat`, `accumulate`, and `outer`), are incorrectly annotated (as `None`
> *attributes*, even though at runtime they're methods that raise a
> `ValueError` when called).
> This currently makes it impossible to properly type these in
> `optype.numpy.AnyUFunc`; doing so would make it incompatible with numpy's
> ufuncs.

[UFUNC-DOC]: https://numpy.org/doc/stable/reference/ufuncs.html
[UFUNC-REF]: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.html
[FROMPYFUNC]: https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html

##### `CanArrayUFunc`

When ufuncs are called on some inputs, the ufunc will call the
`__array_ufunc__`, which is not unlike how `abs()` calls `__abs__`.

With `optype.numpy.CanArrayUFunc`, it becomes straightworward to annotate
potential arguments to `np.ufunc`.
It's a single-method runtime-checkable protocol, whose type signature looks
roughly like:

```python
CanArrayUFunc[
    Fn: AnyUFunc = Any,
]
```

> [!NOTE]
> Due to the previously mentioned typing limitations of `np.ufunc`,
> the `*args` and `**kwargs` of `CanArrayUFunc.__array_ufunc__` are currently
> impossible to properly annotate.

[NEP29]: https://numpy.org/neps/nep-0029-deprecation_policy.html
[SPEC0]: https://scientific-python.org/specs/spec-0000/
[PEP695]: https://peps.python.org/pep-0695/
[PEP696]: https://peps.python.org/pep-0696/
