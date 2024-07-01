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

- [`optype`](#optype)
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
- [`optype.copy`](#optypecopy)
- [`optype.dataclasses`](#optypedataclasses)
- [`optype.pickle`](#optypepickle)
- [`optype.string`](#optypestring)
- [`optype.typing`](#optypetyping)
    - [`Any*` type aliases](#any-type-aliases)
    - [`Empty*` type aliases](#empty-type-aliases)
    - [Literal types](#literal-types)
- [`optype.numpy`](#numpy)
    - [Arrays](#arrays)
        - [`Array`](#array)
        - [`AnyArray`](#anyarray)
    - [Shapes](#shapes)
    - [Scalars](#scalars)
        - [`Scalar`](#scalar)
        - [`Any{Scalar}`](#anyscalar)
        - [`Any*DType`](#anydtype)
    - [Data type objects](#data-type-objects)
        - [`DType`](#dtype)
        - [`AnyDType`](#anydtype)
    - [Universal functions](#universal-functions)
    - [Low-level interfaces](#low-level-interfaces)

<!-- TOC end -->

### `optype`

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

### `optype.copy`

For the [`copy`][CP] standard library, `optype.copy` provides the following
runtime-checkable interfaces:

<table>
    <tr>
        <th align="center"><code>copy</code> standard library</th>
        <th colspan="2" align="center"><code>optype.copy</code></th>
    </tr>
    <tr>
        <td>function</td>
        <th>type</th>
        <th>method</th>
    </tr>
    <tr>
        <td><code>copy.copy(_) -> R</code></td>
        <td><code>__copy__() -> R</code></td>
        <td><code>CanCopy[R]</code></td>
    </tr>
    <tr>
        <td><code>copy.deepcopy(_, memo={}) -> R</code></td>
        <td><code>__deepcopy__(memo, /) -> R</code></td>
        <td><code>CanDeepcopy[R]</code></td>
    </tr>
    <tr>
        <td>
            <code>copy.replace(_, /, **changes: V) -> R</code>
            <sup>[1]</sup>
        </td>
        <td><code>__replace__(**changes: V) -> R</code></td>
        <td><code>CanReplace[V, R]</code></td>
    </tr>
</table>

<sup>[1]</sup> *`copy.replace` requires `python>=3.13`
(but `optype.copy.CanReplace` doesn't)*

In practice, it makes sense that a copy of an instance is the same type as the
original.
But because `typing.Self` cannot be used as a type argument, this difficult
to properly type.
Instead, you can use the `optype.copy.Can{}Self` types, which are the
runtime-checkable equivalents of the following (recursive) type aliases:

```python
type CanCopySelf = CanCopy[CanCopySelf]
type CanDeepcopySelf = CanDeepcopy[CanDeepcopySelf]
type CanReplaceSelf[V] = CanReplace[V, CanReplaceSelf[V]]
```

[CP]: https://docs.python.org/3/library/copy.html

### `optype.dataclasses`

For the [`dataclasses`][DC] standard library, `optype.dataclasses` provides the
`HasDataclassFields[V: Mapping[str, Field]]` interface.
It can conveniently be used to check whether a type or instance is a
dataclass, i.e. `isinstance(obj, HasDataclassFields)`.

[DC]: https://docs.python.org/3/library/dataclasses.html

### `optype.pickle`

For the [`pickle`][PK] standard library, `optype.pickle` provides the following
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
        <td><code>CanReduce[R: str | tuple = ...]</code></td>
    </tr>
    <tr>
        <td><code>__reduce_ex__</code></td>
        <td><code>(CanIndex) -> R</code></td>
        <td><code>CanReduceEx[R: str | tuple = ...]</code></td>
    </tr>
    <tr>
        <td><code>__getstate__</code></td>
        <td><code>() -> S</code></td>
        <td><code>CanGetstate[S]</code></td>
    </tr>
    <tr>
        <td><code>__setstate__</code></td>
        <td><code>(S) -> None</code></td>
        <td><code>CanSetstate[S]</code></td>
    </tr>
    <tr>
        <td>
            <code>__getnewargs__</code><br>
            <code>__new__</code>
        </td>
        <td>
            <code>() -> tuple[V, ...]</code><br>
            <code>(V) -> Self</code><br>
        </td>
        <td><code>CanGetnewargs[V]</code></td>
    </tr>
    <tr>
        <td>
            <code>__getnewargs_ex__</code><br>
            <code>__new__</code>
        </td>
        <td>
            <code>() -> tuple[tuple[V, ...], dict[str, KV]]</code><br>
            <code>(*tuple[V, ...], **dict[str, KV]) -> Self</code><br>
        </td>
        <td><code>CanGetnewargsEx[V, KV]</code></td>
    </tr>
</table>

### `optype.string`

The [`string`](https://docs.python.org/3/library/string.html) standard
library contains practical constants, but it has two issues:

- The constants contain a collection of characters, but are represented as
  a single string. This makes it practically impossible to type-hint the
  individual characters, so typeshed currently types these constants as a
  `LiteralString`.
- The names of the constants are inconsistent, and doesn't follow
  [PEP 8](https://peps.python.org/pep-0008/#constants).

So instead, `optype.string` provides an alternative interface, that is
compatible with `string`, but with slight differences:

- For each constant, there is a corresponding `Literal` type alias for
  the *individual* characters. Its name matches the name of the constant,
  but is singular instead of plural.
- Instead of a single string, `optype.string` uses a `tuple` of characters,
  so that each character has its own `typing.Literal` annotation.
  Note that this is only tested with (based)pyright / pylance, so it might
  not work with mypy (it has more bugs than it has lines of codes).
- The names of the constant are consistent with PEP 8, and use a postfix
  notation for variants, e.g. `DIGITS_HEX` instead of `hexdigits`.
- Unlike `string`, `optype.string` has a constant (and type alias) for
  binary digits `'0'` and `'1'`; `DIGITS_BIN` (and `DigitBin`). Because
  besides `oct` and `hex` functions in `builtins`, there's also the
  `builtins.bin` function.

<table>
    <tr>
        <th colspan="2"><code>string._</code></th>
        <th colspan="2"><code>optype.string._</code></th>
    </tr>
    <tr>
        <th>constant</th>
        <th>char type</th>
        <th>constant</th>
        <th>char type</th>
    </tr>
    <tr>
        <td colspan="2" align="center"><i>missing</i></td>
        <td><code>DIGITS_BIN</code></td>
        <td><code>DigitBin</code></td>
    </tr>
    <tr>
        <td><code>octdigits</code></td>
        <td rowspan="9"><code>LiteralString</code></td>
        <td><code>DIGITS_OCT</code></td>
        <td><code>DigitOct</code></td>
    </tr>
    <tr>
        <td><code>digits</code></td>
        <td><code>DIGITS</code></td>
        <td><code>Digit</code></td>
    </tr>
    <tr>
        <td><code>hexdigits</code></td>
        <td><code>DIGITS_HEX</code></td>
        <td><code>DigitHex</code></td>
    </tr>
    <tr>
        <td><code>ascii_letters</code></td>
        <td><code>LETTERS</code></td>
        <td><code>Letter</code></td>
    </tr>
    <tr>
        <td><code>ascii_lowercase</code></td>
        <td><code>LETTERS_LOWER</code></td>
        <td><code>LetterLower</code></td>
    </tr>
    <tr>
        <td><code>ascii_uppercase</code></td>
        <td><code>LETTERS_UPPER</code></td>
        <td><code>LetterUpper</code></td>
    </tr>
    <tr>
        <td><code>punctuation</code></td>
        <td><code>PUNCTUATION</code></td>
        <td><code>Punctuation</code></td>
    </tr>
    <tr>
        <td><code>whitespace</code></td>
        <td><code>WHITESPACE</code></td>
        <td><code>Whitespace</code></td>
    </tr>
    <tr>
        <td><code>printable</code></td>
        <td><code>PRINTABLE</code></td>
        <td><code>Printable</code></td>
    </tr>
</table>

Each of the `optype.string` constants is exactly the same as the corresponding
`string` constant (after concatenation / splitting), e.g.

```pycon
>>> import string
>>> import optype as opt
>>> ''.join(opt.string.PRINTABLE) == string.printable
True
>>> tuple(string.printable) == opt.string.PRINTABLE
True
```

Similarly, the values within a constant's `Literal` type exactly match the
values of its constant:

```pycon
>>> import optype as opt
>>> from optype.inspect import get_args
>>> get_args(opt.string.Printable) == opt.string.PRINTABLE
True
```

The `optype.inspect.get_args` is a non-broken variant of `typing.get_args`
that correctly flattens nested literals, type-unions, and PEP 695 type aliases,
so that it matches the official typing specs.
*In other words; `typing.get_args` is yet another fundamentally broken
python-typing feature that's useless in the situations where you need it
most.*

### `optype.typing`

#### `Any*` type aliases

Type aliases for anything that can *always* be passed to
`int`, `float`, `complex`, `iter`, or `typing.Literal`

<table>
    <tr>
        <th>Python constructor</th>
        <th><code>optype.typing</code> alias</th>
    </tr>
    <tr>
        <td><code>int(_)</code></td>
        <td><code>AnyInt</code></td>
    </tr>
    <tr>
        <td><code>float(_)</code></td>
        <td><code>AnyFloat</code></td>
    </tr>
    <tr>
        <td><code>complex(_)</code></td>
        <td><code>AnyComplex</code></td>
    </tr>
    <tr>
        <td><code>iter(_)</code></td>
        <td><code>AnyIterable</code></td>
    </tr>
    <tr>
        <td><code>typing.Literal[_]</code></td>
        <td><code>AnyLiteral</code></td>
    </tr>
</table>

> [!NOTE]
> Even though *some* `str` and `bytes` can be converted to `int`, `float`,
> `complex`, most of them can't, and are therefore not included in these
> type aliases.

#### `Empty*` type aliases

These are builtin types or collections that are empty, i.e. have length 0 or
yield no elements.

<table>
    <tr>
        <th>instance</th>
        <th><code>optype.typing</code> type</th>
    </tr>
    <tr>
        <td><code>''</code></td>
        <td><code>EmptyString</code></td>
    </tr>
    <tr>
        <td><code>b''</code></td>
        <td><code>EmptyBytes</code></td>
    </tr>
    <tr>
        <td><code>()</code></td>
        <td><code>EmptyTuple</code></td>
    </tr>
    <tr>
        <td><code>[]</code></td>
        <td><code>EmptyList</code></td>
    </tr>
    <tr>
        <td><code>{}</code></td>
        <td><code>EmptyDict</code></td>
    </tr>
    <tr>
        <td><code>set()</code></td>
        <td><code>EmptySet</code></td>
    </tr>
    <tr>
        <td><code>(i for i in range(0))</code></td>
        <td><code>EmptyIterable</code></td>
    </tr>
</table>

#### Literal types

<table>
    <tr>
        <th>Literal values</th>
        <th><code>optype.typing</code> type</th>
        <th>Notes</th>
    </tr>
    <tr>
        <td><code>{False, True}</code></td>
        <td><code>LiteralFalse</code></td>
        <td>
            Similar to <code>typing.LiteralString</code>, but for
            <code>bool</code>.
        </td>
    </tr>
    <tr>
        <td><code>{0, 1, ..., 255}</code></td>
        <td><code>LiteralByte</code></td>
        <td>
            Integers in the range 0-255, that make up a <code>bytes</code>
            or <code>bytearray</code> objects.
        </td>
    </tr>
</table>

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

#### Shapes

A *shape* is nothing more than a tuple of (non-negative) integers, i.e.
an instance of `tuple[int, ...]`.
The length of a shape is often referred to as the *number of dimensions*
or the *dimensionality* of the array or scalar.

> [!NOTE]
> Before NumPy 2, the maximum number of dimensions was 32, but has since
> been increased to 64.

To make typing the shape of an array easier, optype provides two families of
shape type aliases: `AtLeast{N}D` and `AtMost{N}D`.
The `{N}` should be replaced by the number of dimensions, which currently
is limited to `0`, `1`, `2`, and `3`.

Both of these families are generic, and their (optional) type parameters must
be either `int` (default), or a literal (non-negative) integer, i.e. like
`typing.Literal[N: int]`.

Without going into details, the names `AtLeast{N}D` and `AtMost{N}D` should
are probably self-explanatory:
They are the types of shapes whose `len(shape) >= N` and `len(shape) <= N`,
respectively.

Specifically, `AtLeast0D` and `AtMost0D` are defined as something similar to

```python
type _AnyShape = tuple[int, ...]  # helper alias

type AtLeast0D[*Ns = *_AnyShape] = tuple[*Ns]
type AtMost0D = tuple[()]
```

Note that `AtMost0D` only matches the empty tuple: `()`.

For higher dimensions, things get more complicated:

```python
type AtLeast1D[N0: int = int, *Ns = *_AnyShape] = tuple[N0, *Ns]
type AtMost1D[N0: int = int] = tuple[N0] | AtMost0D
```

The recursive nature of these definitions makes it straightforward to
extend these to higher dimensions (currently `2` and `3`).

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

##### `Any{Scalar}`

For every (standard) numpy scalar type (i.e. subtypes of `np.generic`), there
is the `optype.numpy.Any{Scalar}` alias (where `{Scalar}` should be replaced
with the title-cased name of the scalar, without potential trailing underscore).

So for `np.bool_` there's `onp.AnyBool`,
for `np.uint8` there's `onp.AnyUInt8`, and
for `np.floating[N: npt.NBitBase]` there's `AnyFloating[N: npt.NBitBase]`.

> [!NOTE]
> The *extended-precision* scalar types (e.g. `np.int128`, `np.float96` and
> `np.complex512`) are not included, because their availability is
> platform-dependent.

When a value of type `Any{Scalar}` is passed to e.g. `np.array`,
the resulting `np.ndarray` will have a scalar type that matches
the corresponding `Any{Scalar}`.
For instance, passing `x: onp.AnyFloat64` as `np.array(x)` returns an
array of type `onp.Array[tuple[()], np.float64]`
(where `tuple[()]` implies that its shape is `()`).

Each `Any{Scalar}` contains at least the relevant `np.generic` subtype,
zero or more [`ctypes`][LIB-CTYPES] types, and
zero or more of the Python `builtins` types.

So for instance `type AnyUInt8 = np.uint8 | ct.c_uint8`, and
`type AnyCDouble = np.cdouble | complex`.

##### `Any*DType`

In the same way as `Any*`, there's a `Any*DType` for each of the numpy
scalar types.

These type aliases describe what's allowed to be passed to e.g. the
`np.dtype[ST: np.generic]` constructor, so that its scalar type `ST` matches
the one corresponding to the passed `Any*DType`.

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

A large portion of numpy's public API consists of *universal functions*, often
denoted as [ufuncs][DOC-UFUNC], which are (callable) instances of
[`np.ufunc`][REF_UFUNC].

> [!TIP]
> Custom ufuncs can be created using [`np.frompyfunc`][REF_FROMPY], but also
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
    Fn: CanCall[..., Any] = Any,
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

#### Low-level interfaces

Within `optype.numpy` there are several `Can*` (single-method) and `Has*`
(single-attribute) protocols, related to the `__array_*__` dunders of the
NumPy Python API.
These typing protocols are, just like the `optype.Can*` and `optype.Has*` ones,
runtime-checkable and extensible (i.e. not `@final`).

> [!TIP]
> All type parameters of these protocols can be omitted, which is equivalent
> to passing `typing.Any` (or its upper type bound).

<table>
    <tr>
        <th>Protocol type signature</th>
        <th>Implements</th>
        <th>NumPy docs</th>
    </tr>
    <tr>
<td>

```python
class CanArray[
    ND: tuple[int, ...] = ...,
    ST: np.generic = ...,
]
```

</td>
<td>

```python
def __array__[RT: np.generic = ST](
    self,
    dtype: DType[RT] | None = ...,
) -> Array[ND, RT]
```

</td>
<td>

[User Guide: Interoperability with NumPy][DOC-ARRAY]

</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```python
class CanArrayUFunc[
    U: AnyUFunc = ...,
    R: object = ...,
]
```

</td>
<td>

```python
def array_ufunc(
    self,
    ufunc: U,
    method: Literal['__call__', ...],
    *args: Any,
    **kwargs: Any,
) -> R
```

</td>
<td>

[NEP 13][NEP13]

</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```python
class CanArrayFunction[
    F: CanCall[..., Any] = ...,
    R: object = ...,
]
```

</td>
<td>

```python
def array_function(
    self,
    func: F,
    types: CanIterSelf[type],
    args: tuple[Any, ...],
    kwargs: Mapping[str, ...],
) -> R
```

</td>
<td>

[NEP 18][NEP18]

</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```python
class CanArrayFinalize[
    T: object = ...,
]
```

</td>
<td>

```python
def __array_finalize__(
    self,
    obj: T,
) -> None
```

</td>
<td>

[User Guide: Subclassing ndarray][DOC-AFIN]

</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```python
class CanArrayWrap
```

</td>
<td>

```python
def __array_wrap__[ND, ST](
    self,
    array: Array[ND, ST],
    context: (...) | None = ...,
    return_scalar: bool = ...,
) -> Self | Array[ND, ST]
```

</td>
<td>

[API: Standard array subclasses][REF_ARRAY-WRAP]

</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```python
class HasArrayInterface[
    V: Mapping[str, Any] = ...,
]
```

</td>
<td>

```python
@property
def __array_interface__(self) -> V
```

</td>
<td>

[API: The array interface protocol][REF_ARRAY-INTER]

</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```python
class HasArrayPriority
```

</td>
<td>

```python
@property
def __array_priority__(self) -> float
```

</td>
<td>

[API: Standard array subclasses][REF_ARRAY-PRIO]

</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```python
class HasDType[
    DT: DType = DType,
]
```

</td>
<td>

```python
@property
def dtype(self) -> DT
```

</td>
<td>

[API: Specifying and constructing data types][REF_DTYPE]

</td>
    </tr>
</table>

<!-- references -->

[LIB-CTYPES]: https://docs.python.org/3/library/ctypes.html

[DOC-UFUNC]: https://numpy.org/doc/stable/reference/ufuncs.html
[DOC-ARRAY]: https://numpy.org/doc/stable/user/basics.interoperability.html#the-array-method
[DOC-AFIN]: https://numpy.org/doc/stable/user/basics.subclassing.html#the-role-of-array-finalize

[REF_UFUNC]: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.html
[REF_FROMPY]: https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html
[REF_ARRAY-WRAP]: https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_wrap__
[REF_ARRAY-INTER]: https://numpy.org/doc/stable/reference/arrays.interface.html#python-side
[REF_ARRAY-PRIO]: https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_priority__
[REF_DTYPE]: https://numpy.org/doc/stable/reference/arrays.dtypes.html#specifying-and-constructing-data-types

[NEP13]: https://numpy.org/neps/nep-0013-ufunc-overrides.html
[NEP18]: https://numpy.org/neps/nep-0018-array-function-protocol.html
[NEP29]: https://numpy.org/neps/nep-0029-deprecation_policy.html

[SPEC0]: https://scientific-python.org/specs/spec-0000/

[PEP695]: https://peps.python.org/pep-0695/
[PEP696]: https://peps.python.org/pep-0696/
