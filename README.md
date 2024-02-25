<h1 align="center">optype</h1>

<p align="center">
    <i>One protocol, one method.</i>
</p>

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
        />
    </a>
    <a href="https://github.com/microsoft/pyright">
        <img
            alt="Checked with pyright"
            src="https://microsoft.github.io/pyright/img/pyright_badge.svg"
        />
    </a>
</p>

---

## Installation

Optype is available as [`optype`](OPTYPE) on PyPI:

```shell
pip install optype
```
[OPTYPE]: https://pypi.org/project/optype/


<!-- ## Getting started -->
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


### Type conversion

The return type of these special methods is *invariant*. Python will raise an
error if some other (sub)type is returned.
This is why these `optype` interfaces don't accept generic type arguments.

| Type         | Signature                      | Expression      |
| ------------ | ------------------------------ | --------------- |
| `CanBool`    | `__bool__(self) -> bool`       | `bool(self)`    |
| `CanInt`     | `__int__(self) -> int`         | `int(self)`     |
| `CanFloat`   | `__float__(self) -> float`     | `float(self)`   |
| `CanComplex` | `__complex__(self) -> complex` | `complex(self)` |
| `CanBytes`   | `__bytes__(self) -> bytes`     | `bytes(self)`   |
| `CanStr`     | `__str__(self) -> str`         | `str(self)`     |


These formatting methods are allowed to return instances that are a subtype
of the `str` builtin. The same holds for the `__format__` argument.
So if you're a 10x developer that wants to hack Python's f-strings, but only
if your type hints are spot-on; `optype` is you friend.

| Type                       | Signature                     | Expression     |
| -------------------------- | ----------------------------- | -------------- |
| `CanRepr[Y: str]`          | `__repr__(self) -> T`         | `repr(_)`      |
| `CanFormat[X: str, Y: str]`| `__format__(self, x: X) -> Y` | `format(_, x)` |


### "Rich comparison" operators

These special methods generally a `bool`. However, instances of any type can
be returned.

| Type           | Signature                  | Expression  | Expr. Reflected |
| -------------- | -------------------------- | ----------- | --------------- |
| `CanLt[X, Y]`  | `__lt__(self, x: X) -> Y`  | `self < x`  | `x > self`      |
| `CanLe[X, Y]`  | `__le__(self, x: X) -> Y`  | `self <= x` | `x >= self`     |
| `CanEq[X, Y]`  | `__eq__(self, x: X) -> Y`  | `self == x` | `x == self`     |
| `CanNe[X, Y]`  | `__ne__(self, x: X) -> Y`  | `self != x` | `x != self`     |
| `CanGt[X, Y]`  | `__gt__(self, x: X) -> Y`  | `self > x`  | `x < self`      |
| `CanGe[X, Y]`  | `__ge__(self, x: X) -> Y`  | `self >= x` | `x <= self`     |


### Attribute access

<table>
    <tr>
        <th>Type</th>
        <th>Signature</th>
        <th>Expression</th>
    </tr>
    <tr>
        <td><code>CanGetattr[K: str, V]</code></td>
        <td><code>__getattr__(self, k: K) -> V</code></td>
        <td>
            <code>v = self.k</code> or<br/>
            <code>v = getattr(self, k)</code>
        </td>
    </tr>
    <tr>
        <td><code>CanGetattribute[K: str, V]</code></td>
        <td><code>__getattribute__(self, k: K) -> V</code></td>
        <td>
            <code>v = self.k</code> or <br/>
            <code>v = getattr(self, k)</code>
        </td>
    </tr>
    <tr>
        <td><code>CanSetattr[K: str, V]</code></td>
        <td><code>__setattr__(self, k: K, v: V)</code></td>
        <td>
            <code>self.k = v</code> or<br/>
            <code>setattr(self, k, v)</code>
        </td>
    </tr>
    <tr>
        <td><code>CanDelattr[K: str]</code></td>
        <td><code>__delattr__(self, k: K)</code></td>
        <td>
            <code>del self.k</code> or<br/>
            <code>delattr(self, k)</code>
        </td>
    </tr>
    <tr>
        <td><code>CanDir[Vs: CanIter[Any]]</code></td>
        <td><code>__dir__(self) -> Vs</code></td>
        <td><code>dir(self)</code></td>
    </tr>
</table>


### Iteration

The operand `x` of `iter(_)` is within Python known as an *iterable*, which is
what `collections.abc.Iterable[K]` is often used for (e.g. as base class, or
for instance checking).

The `optype` analogue is `CanIter[Ks]`, which as the name suggests,
also implements `__iter__`. But unlike `Iterable[K]`, its type parameter `Ks`
binds to the return type of `iter(_)`. This makes it possible to annotate the
specific type of the *iterable* that `iter(_)` returns. `Iterable[K]` is only
able to annotate the type of the iterated value. To see why that isn't
possible, see [python/typing#548](https://github.com/python/typing/issues/548).

The `collections.abc.Iterator[K]` is even more awkward; it is a subtype of
`Iterable[K]`. For those familiar with `collections.abc` this might come as a
surprise, but an iterator only needs to implement `__next__`, `__iter__` isn't
needed. This means that the `Iterator[K]` is unnecessarily restrictive.
Apart from that being theoretically "ugly", it has significant performance
implications, because the time-complexity of `isinstance` on a
`typing.Protocol` is $O(n)$, with the $n$ referring to the amount of members.
So even if the overhead of the inheritance and the `abc.ABC` usage is ignored,
`collections.abc.Iterator` is twice as slow as it needs to be.

That's one of the (many) reasons that `optype.CanNext[V]` and
`optype.CanNext[V]` are the better alternatives to `Iterable` and `Iterator`
from the abracadabra collections. This is how they are defined:

| Type                        | Signature              | Expression   |
| --------------------------- | ---------------------- | ------------ |
| `CanNext[V]`                | `__next__(self) -> V`  | `next(self)` |
| `CanIter[Vs: CanNext[Any]]` | `__iter__(self) -> Vs` | `iter(self)` |


### Containers

| Type                  | Signature                          | Expression     |
| --------------------- | ---------------------------------- | -------------- |
| `CanLen`              | `__len__(self) -> int`             | `len(self)`    |
| `CanLengthHint`       | `__length_hint__(self) -> int`     | [docs](LH)     |
| `CanGetitem[K, V]`    | `__getitem__(self, k: K) -> V`     | `self[k]`      |
| `CanSetitem[K, V]`    | `__setitem__(self, k: K, v: V)`    | `self[k] = v`  |
| `CanDelitem[K]`       | `__delitem__(self, k: K)`          | `del self[k]`  |
| `CanMissing[K, V]`    | `__missing__(self, k: K) -> V`     | [docs](GM)     |
| `CanReversed[Y]` [^4] | `__reversed__(self) -> Y`          |`reversed(self)`|
| `CanContains[K]`      | `__contains__(self, k: K) -> bool` | `x in self`    |

For indexing or locating container values, the following special methods are
relevant:

| Type       | Signature                | Expression   |
| ---------- | ------------------------ | ------------ |
| `CanHash`  | `__hash__(self) -> int`  | `hash(self)` |
| `CanIndex` | `__index__(self) -> int` | [docs](IX)   |

[^4]:  Although not strictly required, `Y@CanReversed` should be a `CanNext`.
[LH]: https://docs.python.org/3/reference/datamodel.html#object.__length_hint__
[GM]: https://docs.python.org/3/reference/datamodel.html#object.__missing__
[IX]: https://docs.python.org/3/reference/datamodel.html#object.__index__


### Descriptors

Interfaces for [descriptors](https://docs.python.org/3/howto/descriptor.html).

<table>
    <tr>
        <th>Type</th>
        <th>Signature</th>
    </tr>
    <tr>
        <td><code>CanGet[T: object, U, V]</code></td>
        <td>
            <code>__get__(self, obj: None, cls: type[T]) -> U</code><br/>
            <code>__get__(self, obj: T, cls: type[T] | None = ...) -> V</code>
        </td>
    </tr>
    <tr>
        <td><code>CanSet[T: object, V]</code></td>
        <td><code>__set__(self, obj: T, v: V) -> Any</code></td>
    </tr>
    <tr>
        <td><code>CanDelete[T: object]</code></td>
        <td><code>__delete__(self, obj: T) -> Any</code></td>
    </tr>
    <tr>
        <td><code>CanSetName[T: object]</code></td>
        <td>
            <code>__set_name__(self, cls: type[T], name: str) -> Any</code>
        </td>
    </tr>
</table>


### Callable objects

Like `collections.abc.Callable`, but without esoteric hacks.

<table>
    <tr>
        <th>Type</th>
        <th>Signature</th>
        <th>Expression</th>
    </tr>
    <tr>
        <td><code>CanCall[**Xs, Y]</code></td>
        <td>
            <code>__call__(self, *xs: Xs.args, **kxs: Xs.kwargs) -> Y</code>
        </td>
        <td><code>self(*xs, **kxs)</code></td>
    </tr>
</table>


### Numeric operations

For describing things that act like numbers. See the [Python docs](NT) for more
info.

| Type                | Signature                        | Expression         |
| ------------------- | -------------------------------- | ------------------ |
| `CanAdd[X, Y]`      | `__add__(self, x: X) -> Y`       | `self + x`         |
| `CanSub[X, Y]`      | `__sub__(self, x: X) -> Y`       | `self - x`         |
| `CanMul[X, Y]`      | `__mul__(self, x: X) -> Y`       | `self * x`         |
| `CanMatmul[X, Y]`   | `__matmul__(self, x: X) -> Y`    | `self @ x`         |
| `CanTruediv[X, Y]`  | `__truediv__(self, x: X) -> Y`   | `self / x`         |
| `CanFloordiv[X, Y]` | `__floordiv__(self, x: X) -> Y`  | `self // x`        |
| `CanMod[X, Y]`      | `__mod__(self, x: X) -> Y`       | `self % x`         |
| `CanDivmod[X, Y]`   | `__divmod__(self, x: X) -> Y`    | `divmod(self, x)`  |
| `CanPow2[X, Y]`     | `__pow__(self, x: X) -> Y`       | `self ** x`        |
| `CanPow3[X, M, Y]`  | `__pow__(self, x: X, m: M) -> Y` | `pow(self, x, m)`  |
| `CanLshift[X, Y]`   | `__lshift__(self, x: X) -> Y`    | `self << x`        |
| `CanRshift[X, Y]`   | `__rshift__(self, x: X) -> Y`    | `self >> x`        |
| `CanAnd[X, Y]`      | `__and__(self, x: X) -> Y`       | `self & x`         |
| `CanXor[X, Y]`      | `__xor__(self, x: X) -> Y`       | `self ^ x`         |
| `CanOr[X, Y]`       | `__or__(self, x: X) -> Y`        | `self \| x`        |

Additionally, there is the intersection type
`CanPow[X, M, Y2, Y3] =: CanPow2[X, Y2] & CanPow3[X, M, Y3]`, that overloads
both `__pow__` method signatures. Note that the `2` and `3` suffixes refer
to the *arity* (#parameters) of the operators.

For the binary infix operators above, `optype` additionally provides
interfaces with reflected (swapped) operands:

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

Note that `CanRPow` corresponds to `CanPow2`; the 3-parameter "modulo" `pow`
does not reflect in Python.

Similarly, the augmented assignment operators are described by the following
`optype` interfaces:

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

Additionally, there are the unary arithmetic operators:

| Type               | Signature                     | Expression        |
| ------------------ | ----------------------------- | ----------------- |
| `CanPos[Y]`        | `__pos__(self) -> Y`          | `+self`           |
| `CanNeg[Y]`        | `__neg__(self) -> Y`          | `-self`           |
| `CanInvert[Y]`     | `__invert__(self) -> Y`       | `~self`           |
| `CanAbs[Y]`        | `__abs__(self) -> Y`          | `abs(self)`       |

The `round` function comes in two flavors, and their overloaded intersection:

<table>
    <tr>
        <th>Type</th>
        <th>Signature</th>
        <th>Expression</th>
    </tr>
    <tr>
        <td><code>CanRound1[Y1]</code></td>
        <td><code>__round__(self) -> Y1</code></td>
        <td><code>round(self)</code></td>
    </tr>
    <tr>
        <td><code>CanRound2[N, Y2]</code></td>
        <td><code>__round__(self, n: N) -> Y2</code></td>
        <td><code>round(self, n)</code></td>
    </tr>
    <tr>
        <td><code>CanRound[N, Y1, Y2]</code></td>
        <td>
            <code>__round__(self) -> Y1</code></br>
            <code>__round__(self, n: N) -> Y2</code>
        </td>
        <td><code>round(self[, n: N])</code></td>
    </tr>
</table>

The last "double" signature denotes overloading.

To illustrate; `float` is a `CanRound[int, int, float]` and `int` a
`CanRound[int, int, int]`.

And finally, the remaining rounding functions:

| Type           | Signature               | Expression         |
| -------------- | ----------------------- | ------------------ |
| `CanTrunc[Y]`  | `__trunc__(self) -> Y`  | `math.trunc(self)` |
| `CanFloor[Y]`  | `__floor__(self) -> Y`  | `math.floor(self)` |
| `CanCeil[Y]`   | `__ceil__(self) -> Y`   | `math.ceil(self)`  |

Note that the type parameter `Y` is unbounded, because technically these
methods can return any type.

[NT]: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types


### Context managers

Support for the `with` statement.

| Type           | Signature                                  |
| -------------- | ------------------------------------------ |
| `CanEnter[V]`  | `__enter__(self) -> V`                     |
| `CanExit[R]`   | `__exit__(self, *exc_info: *ExcInfo) -> R` |

In case of errors, the type alias `ExcInfo` will be
`tuple[type[E], E, types.TracebackType]`, where `E` is some `BaseException`.
On the other hand, if no errors are raised (without being silenced),
then `Excinfo` will be `None` in triplicate.

Because everyone that enters must also leave (that means you too, Barry),
`optype` provides the *intersection type*
`CanWith[V, R] = CanEnter[V] & CanExit[R]`.
If you're thinking of an insect-themed sect right now, that's ok --
intersection types aren't real (yet..?).
To put your mind at ease, here's how it's implemented:

```python
class CanWith[V, R](CanEnter[V], CanExit[R]):
    # You won't find any bugs here :)
    ...
```

### Buffer types

Interfaces for emulating buffer types.

| Type                | Signature                                  |
| ------------------- | ------------------------------------------ |
| `CanBuffer[B: int]` | `__buffer__(self, flags: B) -> memoryview` |
| `CanReleaseBuffer`  | `__release_buffer__(self) -> None`         |

The `flags: B` parameter accepts integers within the `[1, 1023]` interval.
Note that the `CanReleaseBuffer` isn't always needed.
See the [Python docs](BP) or [`inspect.BufferFlags`](BF) for more info.


[BP]: https://docs.python.org/3/reference/datamodel.html#python-buffer-protocol
[BD]: https://docs.python.org/3/library/inspect.html#inspect.BufferFlags

### Async objects

The `optype` variant of `collections.abc.Awaitable[V]`. The only difference
is that `optype.CanAwait[V]` is a pure interface, whereas `Awaitable` is
also an abstract base class.

| Type          | Signature                                    | Expression   |
| ------------- | -------------------------------------------- | ------------ |
| `CanAwait[V]` | `__await__(self) -> Generator[Any, None, V]` | `await self` |


### Async Iteration

Yes, you guessed it right; the abracadabra collections repeated their mistakes
with their async iterablors (or something like that).

But fret not, the `optype` alternatives are right here:

| Type                     | Signature              | Expression    |
| ------------------------ | ---------------------- | ------------- |
| `CanAnext[V]`            | `__anext__(self) -> V` | `anext(self)` |
| `CanAiter[Vs: CanAnext]` | `__aiter__(self) -> Y` | `aiter(self)` |

But wait, shouldn't `V` be a `CanAwait`? Well, only if you don't want to get
fired...
Technically speaking, `__anext__` can return any type, and `anext` will pass
it along without nagging (instance checks are slow, now stop bothering that
liberal).
Just because something is legal, doesn't mean it's a good idea (don't eat the
yellow snow).


### Async context managers

Support for the `async with` statement.

| Type           | Signature                                                  |
| -------------- | ---------------------------------------------------------- |
| `CanAenter[V]` | `__aenter__(self) -> CanAwait[V]`                          |
| `CanAexit[R]`  | `__aexit__(self, *exc_info: *ExcInfo) -> CanAwait[R]`      |

And just like `CanWith[V, R]` for sync [context managers](#context-managers),
there is the `CanAsyncWith[V, R] = CanAenter[V] & CanAexit[R]` intersection
type.


## Future plans

- Support for Python versions before 3.12.
- A drop-in replacement for the `operator` standard library, with
  runtime-accessible type annotations, and more operators.
- More standard library protocols, e.g. `copy`, `dataclasses`, `pickle`.
- Typed mixins for DRY implementation of operator, e.g. for comparison ops
  `GeFromLt`, `GtFromLe`, etc as a typed alternative for
  `functools.total_ordering`. Similarly for numeric types, with e.g. `__add__`
  and `__neg__`  a mixin could generate `__pos__` and `__sub__`, or with
  `__mod__` and `__truediv__` a mixin could generate `__`
- Dependency-free third-party type support, e.g. protocols for `numpy`'s array
  interface.
