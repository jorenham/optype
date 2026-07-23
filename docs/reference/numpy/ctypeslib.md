# `optype.numpy.ctypeslib`

Typed `ctypes` aliases for NumPy scalar-like C types.

`optype.numpy.ctypeslib` re-exports standard-library `ctypes` constructors under
names that mirror NumPy's scalar naming — `Int32` for the `ctypes` counterpart of
`np.int32`, `LongDouble` for `np.longdouble`, and so on — and adds abstract type
aliases mirroring NumPy's scalar hierarchy (`SignedInteger`, `Inexact`,
`Generic`, …) for use in annotations.

The concrete names are plain re-exports: they *are* the `ctypes` types, so they
can be instantiated and passed to `argtypes`/`restype` as usual. The abstract
names are typing-only.

Throughout this page, `ctypes` is imported as `ct`:

```python
import ctypes as ct
```

The module assumes a C99-compatible compiler, a 32- or 64-bit system, and an
[`ILP32`, `LLP64`, or `LP64` data model][data-models]. If that isn't the case
for your platform, please [open an issue][issues].

[data-models]: https://en.cppreference.com/w/c/language/arithmetic_types
[issues]: https://github.com/jorenham/optype/issues

!!! note

    `Generic`, `Object`, `Array`, and `Bool` shadow builtins or `typing` names.
    This is intentional — it keeps the naming parallel to `np.generic`,
    `np.object_`, `np.ndarray`, and `np.bool` — and is safe as long as the
    module is imported as a namespace rather than star-imported.

## Concrete types

<table>
<tr><th align="left">alias</th><th align="left"><code>ctypes</code></th><th align="left">NumPy analogue</th><th align="left">kind</th></tr>
<tr><td align="left"><code>Bool</code></td><td align="left"><code>c_bool</code></td><td align="left"><code>np.bool</code></td><td align="left">boolean</td></tr>
<tr><td align="left"><code>Int8</code></td><td align="left"><code>c_int8</code></td><td align="left"><code>np.int8</code></td><td align="left">fixed-width integer</td></tr>
<tr><td align="left"><code>UInt8</code></td><td align="left"><code>c_uint8</code></td><td align="left"><code>np.uint8</code></td><td align="left">fixed-width integer</td></tr>
<tr><td align="left"><code>Int16</code></td><td align="left"><code>c_int16</code></td><td align="left"><code>np.int16</code></td><td align="left">fixed-width integer</td></tr>
<tr><td align="left"><code>UInt16</code></td><td align="left"><code>c_uint16</code></td><td align="left"><code>np.uint16</code></td><td align="left">fixed-width integer</td></tr>
<tr><td align="left"><code>Int32</code></td><td align="left"><code>c_int32</code></td><td align="left"><code>np.int32</code></td><td align="left">fixed-width integer</td></tr>
<tr><td align="left"><code>UInt32</code></td><td align="left"><code>c_uint32</code></td><td align="left"><code>np.uint32</code></td><td align="left">fixed-width integer</td></tr>
<tr><td align="left"><code>Int64</code></td><td align="left"><code>c_int64</code></td><td align="left"><code>np.int64</code></td><td align="left">fixed-width integer</td></tr>
<tr><td align="left"><code>UInt64</code></td><td align="left"><code>c_uint64</code></td><td align="left"><code>np.uint64</code></td><td align="left">fixed-width integer</td></tr>
<tr><td align="left"><code>Byte</code></td><td align="left"><code>c_byte</code></td><td align="left"><code>np.byte</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>UByte</code></td><td align="left"><code>c_ubyte</code></td><td align="left"><code>np.ubyte</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>Short</code></td><td align="left"><code>c_short</code></td><td align="left"><code>np.short</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>UShort</code></td><td align="left"><code>c_ushort</code></td><td align="left"><code>np.ushort</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>IntC</code></td><td align="left"><code>c_int</code></td><td align="left"><code>np.intc</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>UIntC</code></td><td align="left"><code>c_uint</code></td><td align="left"><code>np.uintc</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>IntP</code></td><td align="left"><code>c_ssize_t</code></td><td align="left"><code>np.intp</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>UIntP</code></td><td align="left"><code>c_size_t</code></td><td align="left"><code>np.uintp</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>Long</code></td><td align="left"><code>c_long</code></td><td align="left"><code>np.long</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>ULong</code></td><td align="left"><code>c_ulong</code></td><td align="left"><code>np.ulong</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>LongLong</code></td><td align="left"><code>c_longlong</code></td><td align="left"><code>np.longlong</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>ULongLong</code></td><td align="left"><code>c_ulonglong</code></td><td align="left"><code>np.ulonglong</code></td><td align="left">C-native integer</td></tr>
<tr><td align="left"><code>Float32</code></td><td align="left"><code>c_float</code></td><td align="left"><code>np.float32</code></td><td align="left">floating-point</td></tr>
<tr><td align="left"><code>Float64</code></td><td align="left"><code>c_double</code></td><td align="left"><code>np.float64</code></td><td align="left">floating-point</td></tr>
<tr><td align="left"><code>LongDouble</code></td><td align="left"><code>c_longdouble</code></td><td align="left"><code>np.longdouble</code></td><td align="left">floating-point</td></tr>
<tr><td align="left"><code>Complex64</code></td><td align="left"><code>c_float_complex</code></td><td align="left"><code>np.complex64</code></td><td align="left">complex floating-point</td></tr>
<tr><td align="left"><code>Complex128</code></td><td align="left"><code>c_double_complex</code></td><td align="left"><code>np.complex128</code></td><td align="left">complex floating-point</td></tr>
<tr><td align="left"><code>CLongDouble</code></td><td align="left"><code>c_longdouble_complex</code></td><td align="left"><code>np.clongdouble</code></td><td align="left">complex floating-point</td></tr>
<tr><td align="left"><code>Bytes</code></td><td align="left"><code>c_char</code></td><td align="left"><code>np.bytes_</code></td><td align="left">character/byte</td></tr>
<tr><td align="left"><code>Object</code></td><td align="left"><code>py_object</code></td><td align="left"><code>np.object_</code></td><td align="left">Python object reference</td></tr>
</table>

### Several of these names are aliases of each other

Unlike NumPy, `ctypes` has no distinct fixed-width types. `c_int8` *is*
`c_byte`, `c_int16` *is* `c_short`, and the rest are resolved by `ctypes` at
import time from the platform's type sizes. A type checker cannot distinguish
names that resolve to the same class:

<table>
<tr><th align="left">alias</th><th align="left"><code>ILP32</code></th><th align="left"><code>LP64</code></th><th align="left"><code>LLP64</code></th></tr>
<tr><td align="left"><code>Int8</code></td><td align="left"><code>Byte</code></td><td align="left"><code>Byte</code></td><td align="left"><code>Byte</code></td></tr>
<tr><td align="left"><code>UInt8</code></td><td align="left"><code>UByte</code></td><td align="left"><code>UByte</code></td><td align="left"><code>UByte</code></td></tr>
<tr><td align="left"><code>Int16</code></td><td align="left"><code>Short</code></td><td align="left"><code>Short</code></td><td align="left"><code>Short</code></td></tr>
<tr><td align="left"><code>UInt16</code></td><td align="left"><code>UShort</code></td><td align="left"><code>UShort</code></td><td align="left"><code>UShort</code></td></tr>
<tr><td align="left"><code>Int32</code></td><td align="left"><code>IntC</code> = <code>Long</code></td><td align="left"><code>IntC</code></td><td align="left"><code>IntC</code> = <code>Long</code></td></tr>
<tr><td align="left"><code>UInt32</code></td><td align="left"><code>UIntC</code> = <code>ULong</code></td><td align="left"><code>UIntC</code></td><td align="left"><code>UIntC</code> = <code>ULong</code></td></tr>
<tr><td align="left"><code>Int64</code></td><td align="left"><code>LongLong</code></td><td align="left"><code>Long</code> = <code>LongLong</code></td><td align="left"><code>LongLong</code></td></tr>
<tr><td align="left"><code>UInt64</code></td><td align="left"><code>ULongLong</code></td><td align="left"><code>ULong</code> = <code>ULongLong</code></td><td align="left"><code>ULongLong</code></td></tr>
<tr><td align="left"><code>IntP</code></td><td align="left"><code>IntC</code> = <code>Long</code></td><td align="left"><code>Long</code> = <code>LongLong</code></td><td align="left"><code>LongLong</code></td></tr>
<tr><td align="left"><code>UIntP</code></td><td align="left"><code>UIntC</code> = <code>ULong</code></td><td align="left"><code>ULong</code> = <code>ULongLong</code></td><td align="left"><code>ULongLong</code></td></tr>
</table>

`ctypes` aliases `c_int` to `c_long` when the two have equal size, and
`c_longlong` to `c_long` likewise. So on `LP64`, `Long` **is** `LongLong`; on
`ILP32` and `LLP64`, `IntC` **is** `Long`. There is no data model on which all
four are distinct.

Annotating with either name of a pair is equally correct. The pairs exist so
code can be written in whichever vocabulary — NumPy's or C's — reads better at
the call site.

### `LongDouble`

`ctypes` collapses `c_longdouble` into `c_double` whenever the two have the same
size, so `LongDouble` *is* `Float64` on MSVC and on arm64 macOS.

`c_longdouble` also works only as a *type*, not as a value: its `.value` is a
Python `float`, i.e. a C `double`, so it cannot carry the extra precision of an
80- or 128-bit long double. Use it in `argtypes`/`restype` and in annotations;
don't use it to hold values.

### Complex types

`c_float_complex`, `c_double_complex`, and `c_longdouble_complex` were added in
Python 3.14 and are not available on Windows.

Where they are unavailable — on Windows, or on any Python before 3.14 —
`Complex64`, `Complex128`, and `CLongDouble` are still importable but are
aliases of `Never`. They are therefore uninhabited: any value assigned to them
is a type error, and unions containing them silently drop those members.

### NumPy version notes

`UIntP` is `c_size_t`. On `numpy < 2`, `np.uintp` was `c_void_p` rather than
`c_size_t`; on every supported data model the two have the same width, so the
distinction is not observable in practice.

`np.long` and `np.ulong` are the NumPy 2.0 names for the C `long` types, and do
not exist under that spelling on `numpy < 2`.

### Types with no `ctypes` equivalent

`np.float16`, `np.str_`, `np.datetime64`, and `np.timedelta64` have no `ctypes`
counterpart and are deliberately absent. `c_wchar` is likewise not exposed,
since NumPy maps no dtype onto it.

`Bytes` maps `np.bytes_` onto `c_char`, following
[`np.ctypeslib.as_ctypes_type`][as_ctypes_type]. Note that `c_char` is a single
byte whereas `np.bytes_` is variable-length; the correspondence is with the
dtype's element type, not its length.

[as_ctypes_type]: https://numpy.org/doc/stable/reference/routines.ctypeslib.html#numpy.ctypeslib.as_ctypes_type

## Abstract type aliases

These exist only for annotations. `ct._CData` and `ct._SimpleCData` are typeshed
constructs — `_CData` has no runtime counterpart — so neither should be imported
from `ctypes` or instantiated directly.

<table>
<tr><th align="left">alias</th><th align="left">definition</th></tr>
<tr><td align="left"><code>CType</code></td><td align="left"><code>ct._CData</code></td></tr>
<tr><td align="left"><code>CScalar[T]</code></td><td align="left"><code>ct._SimpleCData[T]</code></td></tr>
<tr><td align="left"><code>Array[CT: CType]</code></td><td align="left"><code>ct.Array[CT] | ct.Array[Array[CT]]</code></td></tr>
</table>

`Array` is recursive, so it matches arbitrarily nested `ctypes` arrays —
`c_int * 3`, `c_int * 3 * 4`, and deeper.

The remaining aliases mirror the `np.generic` hierarchy:

<table>
<tr><th align="left">alias</th><th align="left">C types</th></tr>
<tr><td align="left"><code>SignedInteger</code></td><td align="left"><code>Int8 | Int16 | Int32 | Int64 | Short | IntC | IntP | Long | LongLong</code></td></tr>
<tr><td align="left"><code>UnsignedInteger</code></td><td align="left"><code>UInt8 | UInt16 | UInt32 | UInt64 | UShort | UIntC | UIntP | ULong | ULongLong</code></td></tr>
<tr><td align="left"><code>Integer</code></td><td align="left"><code>SignedInteger | UnsignedInteger</code></td></tr>
<tr><td align="left"><code>Floating</code></td><td align="left"><code>Float32 | Float64 | LongDouble</code></td></tr>
<tr><td align="left"><code>ComplexFloating</code></td><td align="left"><code>Complex64 | Complex128 | CLongDouble</code></td></tr>
<tr><td align="left"><code>Inexact</code></td><td align="left"><code>Floating | ComplexFloating</code></td></tr>
<tr><td align="left"><code>Number</code></td><td align="left"><code>Integer | Inexact</code></td></tr>
<tr><td align="left"><code>Void</code></td><td align="left"><code>ct.Structure | ct.Union</code></td></tr>
<tr><td align="left"><code>Flexible</code></td><td align="left"><code>Bytes | Void</code></td></tr>
<tr><td align="left"><code>Generic</code></td><td align="left"><code>Bool | Number | Flexible | Object</code></td></tr>
</table>

`Byte` and `UByte` are absent from the two integer unions because they are
aliases of `Int8` and `UInt8`; including them would be redundant.

### How the numeric aliases are defined

`Integer`, `Floating`, and `ComplexFloating` are not literally spelled as the
unions above. They are defined as `CScalar[int]`, `CScalar[float]`, and
`CScalar[complex]`, and `Inexact`/`Number` are built from those.

`ct._SimpleCData` is **invariant** in its type parameter, so `CScalar[int]`
resolves to exactly the C integer types listed above — and, unlike a literal
union, also admits any third-party `_SimpleCData[int]` subclass. The unions in
the table describe what these aliases match among the types in this module.

Invariance is also why `Bool` is listed separately in `Generic`: `c_bool` is a
`_SimpleCData[bool]`, and `bool` is not `int` under invariance, so `Bool` is not
covered by `Number`.

## Example

```python
import ctypes as ct

import numpy as np
import optype.numpy.ctypeslib as opct

# the concrete names are the `ctypes` types themselves
assert opct.Int32 is ct.c_int32

buf = (opct.Int32 * 4)(1, 2, 3, 4)
arr = np.ctypeslib.as_array(buf)
assert arr.dtype == np.int32
```

`ctypes` converts simple return types to Python objects on call, so these names
belong on `argtypes`/`restype`, not on the Python-side return annotation:

```python
lib = ct.CDLL("libsum.so")

# int64_t sum_i32(const int32_t *values, int n)
lib.sum_i32.argtypes = [ct.POINTER(opct.Int32), opct.IntC]
lib.sum_i32.restype = opct.Int64


def zeroed[T: opct.Number](ctype: type[T], n: int) -> opct.Array[T]:
    return (ctype * n)()
```
