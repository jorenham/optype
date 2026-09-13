# `ctypeslib` submodule

Typed `ctypes` aliases for NumPy scalar-like C types.

This module is named after [`numpy.ctypeslib`][np-ctypeslib], and follows the
dtype-to-`ctypes` correspondence that
[`np.ctypeslib.as_ctypes_type`][as_ctypes_type] implements.

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

[np-ctypeslib]: https://numpy.org/doc/stable/reference/routines.ctypeslib.html
[as_ctypes_type]: https://numpy.org/doc/stable/reference/routines.ctypeslib.html#numpy.ctypeslib.as_ctypes_type
[data-models]: https://en.cppreference.com/w/c/language/arithmetic_types
[issues]: https://github.com/jorenham/optype/issues

!!! note

    `Generic` shadows `typing.Generic`, and `Array` means something different
    here than it does in the parent `optype.numpy` namespace. Both are
    intentional (the names mirror `np.generic` and `np.ndarray`) and are safe
    as long as the module is imported as a namespace rather than star-imported.

## Concrete types

<table>
    <tr>
        <th>alias</th>
        <th><code>ctypes</code></th>
        <th>NumPy analogue</th>
        <th>kind</th>
    </tr>
    <tr>
        <td><code>Bool</code></td>
        <td><code>c_bool</code></td>
        <td><code>np.bool</code></td>
        <td>boolean</td>
    </tr>
    <tr>
        <td><code>Int8</code></td>
        <td><code>c_int8</code></td>
        <td><code>np.int8</code></td>
        <td>fixed-width integer</td>
    </tr>
    <tr>
        <td><code>UInt8</code></td>
        <td><code>c_uint8</code></td>
        <td><code>np.uint8</code></td>
        <td>fixed-width integer</td>
    </tr>
    <tr>
        <td><code>Int16</code></td>
        <td><code>c_int16</code></td>
        <td><code>np.int16</code></td>
        <td>fixed-width integer</td>
    </tr>
    <tr>
        <td><code>UInt16</code></td>
        <td><code>c_uint16</code></td>
        <td><code>np.uint16</code></td>
        <td>fixed-width integer</td>
    </tr>
    <tr>
        <td><code>Int32</code></td>
        <td><code>c_int32</code></td>
        <td><code>np.int32</code></td>
        <td>fixed-width integer</td>
    </tr>
    <tr>
        <td><code>UInt32</code></td>
        <td><code>c_uint32</code></td>
        <td><code>np.uint32</code></td>
        <td>fixed-width integer</td>
    </tr>
    <tr>
        <td><code>Int64</code></td>
        <td><code>c_int64</code></td>
        <td><code>np.int64</code></td>
        <td>fixed-width integer</td>
    </tr>
    <tr>
        <td><code>UInt64</code></td>
        <td><code>c_uint64</code></td>
        <td><code>np.uint64</code></td>
        <td>fixed-width integer</td>
    </tr>
    <tr>
        <td><code>Byte</code></td>
        <td><code>c_byte</code></td>
        <td><code>np.byte</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>UByte</code></td>
        <td><code>c_ubyte</code></td>
        <td><code>np.ubyte</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>Short</code></td>
        <td><code>c_short</code></td>
        <td><code>np.short</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>UShort</code></td>
        <td><code>c_ushort</code></td>
        <td><code>np.ushort</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>IntC</code></td>
        <td><code>c_int</code></td>
        <td><code>np.intc</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>UIntC</code></td>
        <td><code>c_uint</code></td>
        <td><code>np.uintc</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>IntP</code></td>
        <td><code>c_ssize_t</code></td>
        <td><code>np.intp</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>UIntP</code></td>
        <td><code>c_size_t</code></td>
        <td><code>np.uintp</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>Long</code></td>
        <td><code>c_long</code></td>
        <td><code>np.long</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>ULong</code></td>
        <td><code>c_ulong</code></td>
        <td><code>np.ulong</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>LongLong</code></td>
        <td><code>c_longlong</code></td>
        <td><code>np.longlong</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>ULongLong</code></td>
        <td><code>c_ulonglong</code></td>
        <td><code>np.ulonglong</code></td>
        <td>C-native integer</td>
    </tr>
    <tr>
        <td><code>Float32</code></td>
        <td><code>c_float</code></td>
        <td><code>np.float32</code></td>
        <td>floating-point</td>
    </tr>
    <tr>
        <td><code>Float64</code></td>
        <td><code>c_double</code></td>
        <td><code>np.float64</code></td>
        <td>floating-point</td>
    </tr>
    <tr>
        <td><code>LongDouble</code></td>
        <td><code>c_longdouble</code></td>
        <td><code>np.longdouble</code></td>
        <td>floating-point</td>
    </tr>
    <tr>
        <td><code>Complex64</code></td>
        <td><code>c_float_complex</code></td>
        <td><code>np.complex64</code></td>
        <td>complex floating-point</td>
    </tr>
    <tr>
        <td><code>Complex128</code></td>
        <td><code>c_double_complex</code></td>
        <td><code>np.complex128</code></td>
        <td>complex floating-point</td>
    </tr>
    <tr>
        <td><code>CLongDouble</code></td>
        <td><code>c_longdouble_complex</code></td>
        <td><code>np.clongdouble</code></td>
        <td>complex floating-point</td>
    </tr>
    <tr>
        <td><code>Bytes</code></td>
        <td><code>c_char</code></td>
        <td><code>np.bytes_</code></td>
        <td>character/byte</td>
    </tr>
    <tr>
        <td><code>Object</code></td>
        <td><code>py_object</code></td>
        <td><code>np.object_</code></td>
        <td>Python object reference</td>
    </tr>
</table>

### Complex types

`c_float_complex`, `c_double_complex`, and `c_longdouble_complex` were added in
Python 3.14 and are not available on Windows.

Where they are unavailable, `Complex64`, `Complex128`, and `CLongDouble` are
still importable but are aliases of `Never`.

### Differences between NumPy and `ctypes`

`np.float16`, `np.str_`, `np.datetime64`, and `np.timedelta64` have no `ctypes`
counterpart and are deliberately absent. `c_wchar` is likewise not exposed,
since NumPy maps no dtype onto it.

`Bytes` is `c_char`, which is what
[`np.ctypeslib.as_ctypes_type`][as_ctypes_type] returns for `np.bytes_`. Note
that `c_char` is a single byte whereas `np.bytes_` is variable-length; the
correspondence is with the dtype's element type, not its length.

## Abstract type aliases

`CType` and `CScalar` correspond to the private `ctypes` base classes that every
C type derives from. They exist at runtime, but neither is importable from
`ctypes` by name, and neither is meant to be instantiated; use them in
annotations.

<table>
    <tr>
        <th>alias</th>
        <th>definition</th>
    </tr>
    <tr>
        <td><code>CType</code></td>
        <td><code>ct._CData</code></td>
    </tr>
    <tr>
        <td><code>CScalar[T]</code></td>
        <td><code>ct._SimpleCData[T]</code></td>
    </tr>
    <tr>
        <td><code>Array[CT: CType]</code></td>
        <td><code>ct.Array[CT] | ct.Array[Array[CT]]</code></td>
    </tr>
</table>

`Array` is recursive, so it matches arbitrarily nested `ctypes` arrays:
`c_int * 3`, `c_int * 3 * 4`, and deeper.

The remaining aliases mirror the `np.generic` hierarchy:

<table>
    <tr>
        <th>alias</th>
        <th>C types</th>
    </tr>
    <tr>
        <td><code>SignedInteger</code></td>
        <td>
            <code>Int8 | Int16 | Int32 | Int64 | Short | IntC | IntP | Long | LongLong</code>
        </td>
    </tr>
    <tr>
        <td><code>UnsignedInteger</code></td>
        <td>
            <code>UInt8 | UInt16 | UInt32 | UInt64 | UShort | UIntC | UIntP | ULong | ULongLong</code>
        </td>
    </tr>
    <tr>
        <td><code>Integer</code></td>
        <td><code>SignedInteger | UnsignedInteger</code></td>
    </tr>
    <tr>
        <td><code>Floating</code></td>
        <td><code>Float32 | Float64 | LongDouble</code></td>
    </tr>
    <tr>
        <td><code>ComplexFloating</code></td>
        <td><code>Complex64 | Complex128 | CLongDouble</code></td>
    </tr>
    <tr>
        <td><code>Inexact</code></td>
        <td><code>Floating | ComplexFloating</code></td>
    </tr>
    <tr>
        <td><code>Number</code></td>
        <td><code>Integer | Inexact</code></td>
    </tr>
    <tr>
        <td><code>Void</code></td>
        <td><code>ct.Structure | ct.Union</code></td>
    </tr>
    <tr>
        <td><code>Flexible</code></td>
        <td><code>Bytes | Void</code></td>
    </tr>
    <tr>
        <td><code>Generic</code></td>
        <td><code>Bool | Number | Flexible | Object</code></td>
    </tr>
</table>

`Byte` and `UByte` are absent from the two integer unions because they are
aliases of `Int8` and `UInt8`; including them would be redundant.

### How the numeric aliases are defined

`Integer`, `Floating`, and `ComplexFloating` are defined as `CScalar[int]`,
`CScalar[float]`, and `CScalar[complex]` rather than as the literal unions
above, so they also admit third-party `_SimpleCData` subclasses. `Inexact` and
`Number` are built from those. Because `CScalar` is invariant, `Bool`
(`_SimpleCData[bool]`) is not covered by `Number` and is listed separately in
`Generic`.
