# `Any*Array` and `Any*DType`

The `Any{Scalar}Array` type aliases describe array-likes that are coercible to an
`numpy.ndarray` with specific [dtype][REF-DTYPES].

Unlike `numpy.typing.ArrayLike`, these `optype.numpy` aliases **don't**
accept "bare" scalar types such as `float` and `np.float64`. However, arrays of
"zero dimensions" like `onp.Array[tuple[()], np.float64]` will be accepted.
This is in line with the behavior of [`numpy.isscalar`][REF-ISSCALAR] on `numpy >= 2`.

```py
import numpy.typing as npt
import optype.numpy as onp

v_np: npt.ArrayLike = 3.14  # accepted
v_op: onp.AnyArray = 3.14  # rejected

sigma1_np: npt.ArrayLike = [[0, 1], [1, 0]]  # accepted
sigma1_op: onp.AnyArray = [[0, 1], [1, 0]]  # accepted
```

!!! info

    The [`numpy.dtypes` docs][REF-DTYPES] exists since NumPy 1.26, but its
    type annotations were incorrect before NumPy 2.1 (see
    [numpy/numpy#27008](https://github.com/numpy/numpy/pull/27008))

See the [docs][REF-SCT] for more info on the NumPy scalar type hierarchy.

[REF-SCT]: https://numpy.org/doc/stable/reference/arrays.scalars.html
[REF-DTYPES]: https://numpy.org/doc/stable/reference/arrays.dtypes.html
[REF-ISSCALAR]: https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html

## Abstract types

<table>
    <tr>
        <th align="center" colspan="2"><code>numpy._</code></th>
        <th align="center" colspan="2"><code>optype.numpy._</code></th>
    </tr>
    <tr>
        <th>scalar</th>
        <th>scalar base</th>
        <th>array-like</th>
        <th>dtype-like</th>
    </tr>
    <tr>
        <td><code>generic</code></td>
        <td></td>
        <td><code>AnyArray</code></td>
        <td><code>AnyDType</code></td>
    </tr>
    <tr>
        <td><code>number</code></td>
        <td><code>generic</code></td>
        <td><code>AnyNumberArray</code></td>
        <td><code>AnyNumberDType</code></td>
    </tr>
    <tr>
        <td><code>integer</code></td>
        <td rowspan="2"><code>number</code></td>
        <td><code>AnyIntegerArray</code></td>
        <td><code>AnyIntegerDType</code></td>
    </tr>
    <tr>
        <td><code>inexact</code></td>
        <td><code>AnyInexactArray</code></td>
        <td><code>AnyInexactDType</code></td>
    </tr>
    <tr>
        <td><code>unsignedinteger</code></td>
        <td rowspan="2"><code>integer</code></td>
        <td><code>AnyUnsignedIntegerArray</code></td>
        <td><code>AnyUnsignedIntegerDType</code></td>
    </tr>
    <tr>
        <td><code>signedinteger</code></td>
        <td><code>AnySignedIntegerArray</code></td>
        <td><code>AnySignedIntegerDType</code></td>
    </tr>
    <tr>
        <td><code>floating</code></td>
        <td rowspan="2"><code>inexact</code></td>
        <td><code>AnyFloatingArray</code></td>
        <td><code>AnyFloatingDType</code></td>
    </tr>
    <tr>
        <td><code>complexfloating</code></td>
        <td><code>AnyComplexFloatingArray</code></td>
        <td><code>AnyComplexFloatingDType</code></td>
    </tr>
</table>

## Integers

Unsigned:

<table>
    <tr>
        <th align="center" colspan="2"><code>numpy._</code></th>
        <th align="center"><code>numpy.dtypes._</code></th>
        <th align="center" colspan="2"><code>optype.numpy._</code></th>
    </tr>
    <tr>
        <th>scalar</th>
        <th>scalar base</th>
        <th>dtype</th>
        <th>array-like</th>
        <th>dtype-like</th>
    </tr>
    <tr>
        <td><code>uint_</code></td>
        <td rowspan="9"><code>unsignedinteger</code></td>
        <td rowspan="2"></td>
        <td><code>AnyUIntArray</code></td>
        <td><code>AnyUIntDType</code></td>
    </tr>
    <tr>
        <td><code>uintp</code></td>
        <td><code>AnyUIntPArray</code></td>
        <td><code>AnyUIntPDType</code></td>
    </tr>
    <tr>
        <td><code>uint8</code>, <code>ubyte</code></td>
        <td><code>UInt8DType</code></td>
        <td><code>AnyUInt8Array</code></td>
        <td><code>AnyUInt8DType</code></td>
    </tr>
    <tr>
        <td><code>uint16</code>, <code>ushort</code></td>
        <td><code>UInt16DType</code></td>
        <td><code>AnyUInt16Array</code></td>
        <td><code>AnyUInt16DType</code></td>
    </tr>
    <tr>
        <td><code>uint32</code></td>
        <td><code>UInt32DType</code></td>
        <td><code>AnyUInt32Array</code></td>
        <td><code>AnyUInt32DType</code></td>
    </tr>
    <tr>
        <td><code>uint64</code></td>
        <td><code>UInt64DType</code></td>
        <td><code>AnyUInt64Array</code></td>
        <td><code>AnyUInt64DType</code></td>
    </tr>
    <tr>
        <td><code>uintc</code></td>
        <td><code>UIntDType</code></td>
        <td><code>AnyUIntCArray</code></td>
        <td><code>AnyUIntCDType</code></td>
    </tr>
    <tr>
        <td><code>ulong</code></td>
        <td><code>ULongDType</code></td>
        <td><code>AnyULongArray</code></td>
        <td><code>AnyULongDType</code></td>
    </tr>
    <tr>
        <td><code>ulonglong</code></td>
        <td><code>ULongLongDType</code></td>
        <td><code>AnyULongLongArray</code></td>
        <td><code>AnyULongLongDType</code></td>
    </tr>
</table>

Signed:

<table>
    <tr>
        <th align="center" colspan="2"><code>numpy._</code></th>
        <th align="center"><code>numpy.dtypes._</code></th>
        <th align="center" colspan="2"><code>optype.numpy._</code></th>
    </tr>
    <tr>
        <th>scalar</th>
        <th>scalar base</th>
        <th>dtype</th>
        <th>array-like</th>
        <th>dtype-like</th>
    </tr>
    <tr>
        <td><code>int_</code></td>
        <td rowspan="9"><code>signedinteger</code></td>
        <td rowspan="2"></td>
        <td><code>AnyIntArray</code></td>
        <td><code>AnyIntDType</code></td>
    </tr>
    <tr>
        <td><code>intp</code></td>
        <td><code>AnyIntPArray</code></td>
        <td><code>AnyIntPDType</code></td>
    </tr>
    <tr>
        <td><code>int8</code>, <code>byte</code></td>
        <td><code>Int8DType</code></td>
        <td><code>AnyInt8Array</code></td>
        <td><code>AnyInt8DType</code></td>
    </tr>
    <tr>
        <td><code>int16</code>, <code>short</code></td>
        <td><code>Int16DType</code></td>
        <td><code>AnyInt16Array</code></td>
        <td><code>AnyInt16DType</code></td>
    </tr>
    <tr>
        <td><code>int32</code></td>
        <td><code>Int32DType</code></td>
        <td><code>AnyInt32Array</code></td>
        <td><code>AnyInt32DType</code></td>
    </tr>
    <tr>
        <td><code>int64</code></td>
        <td><code>Int64DType</code></td>
        <td><code>AnyInt64Array</code></td>
        <td><code>AnyInt64DType</code></td>
    </tr>
    <tr>
        <td><span><code>intc</code></span></td>
        <td><code>IntDType</code></td>
        <td><code>AnyIntCArray</code></td>
        <td><code>AnyIntCDType</code></td>
    </tr>
    <tr>
        <td><code>long</code></td>
        <td><code>LongDType</code></td>
        <td><code>AnyLongArray</code></td>
        <td><code>AnyLongDType</code></td>
    </tr>
    <tr>
        <td><code>longlong</code></td>
        <td><code>LongLongDType</code></td>
        <td><code>AnyLongLongArray</code></td>
        <td><code>AnyLongLongDType</code></td>
    </tr>
</table>

!!! info

    Since NumPy 2, `np.uint` and `np.int_` are aliases for `np.uintp` and `np.intp`,
    respectively.

!!! info

    On unix-based platforms `np.[u]intc` are aliases for `np.[u]int32`.

!!! info

    On NumPy 1 `np.uint` and `np.int_` are what in NumPy 2 are now the `np.ulong` and
    `np.long` types, respectively.

## Real floats

<table>
    <tr>
        <th align="center" colspan="2"><code>numpy._</code></th>
        <th align="center"><code>numpy.dtypes._</code></th>
        <th align="center" colspan="2"><code>optype.numpy._</code></th>
    </tr>
    <tr>
        <th>scalar</th>
        <th>scalar base</th>
        <th>dtype</th>
        <th>array-like</th>
        <th>dtype-like</th>
    </tr>
    <tr>
        <td>
            <code>float16</code>,<br>
            <code>half</code>
        </td>
        <td rowspan="2"><code>np.floating</code></td>
        <td><code>Float16DType</code></td>
        <td><code>AnyFloat16Array</code></td>
        <td><code>AnyFloat16DType</code></td>
    </tr>
    <tr>
        <td>
            <code>float32</code>,<br>
            <code>single</code>
        </td>
        <td><code>Float32DType</code></td>
        <td><code>AnyFloat32Array</code></td>
        <td><code>AnyFloat32DType</code></td>
    </tr>
    <tr>
        <td>
            <code>float64</code>,<br>
            <code>double</code>
        </td>
        <td>
            <code>np.floating &</code><br>
            <code>builtins.float</code>
        </td>
        <td><code>Float64DType</code></td>
        <td><code>AnyFloat64Array</code></td>
        <td><code>AnyFloat64DType</code></td>
    </tr>
    <tr>
        <td><code>longdouble</code></td>
        <td><code>np.floating</code></td>
        <td><code>LongDoubleDType</code></td>
        <td><code>AnyLongDoubleArray</code></td>
        <td><code>AnyLongDoubleDType</code></td>
    </tr>
</table>

!!! info

    Depending on the platform, `np.longdouble` is (almost always) an alias for
    **either** `float128`, `float96`, or (sometimes) `float64`.

## Complex floats

<table>
    <tr>
        <th align="center" colspan="2"><code>numpy._</code></th>
        <th align="center"><code>numpy.dtypes._</code></th>
        <th align="center" colspan="2"><code>optype.numpy._</code></th>
    </tr>
    <tr>
        <th>scalar</th>
        <th>scalar base</th>
        <th>dtype</th>
        <th>array-like</th>
        <th>dtype-like</th>
    </tr>
    <tr>
        <td>
            <code>complex64</code>,<br>
            <code>csingle</code>
        </td>
        <td><code>complexfloating</code></td>
        <td><code>Complex64DType</code></td>
        <td><code>AnyComplex64Array</code></td>
        <td><code>AnyComplex64DType</code></td>
    </tr>
    <tr>
        <td>
            <code>complex128</code>,<br>
            <code>cdouble</code>
        </td>
        <td>
            <code>complexfloating &</code><br>
            <code>builtins.complex</code>
        </td>
        <td><code>Complex128DType</code></td>
        <td><code>AnyComplex128Array</code></td>
        <td><code>AnyComplex128DType</code></td>
    </tr>
    <tr>
        <td><code>clongdouble</code></td>
        <td><code>complexfloating</code></td>
        <td><code>CLongDoubleDType</code></td>
        <td><code>AnyCLongDoubleArray</code></td>
        <td><code>AnyCLongDoubleDType</code></td>
    </tr>
</table>

!!! info

    Depending on the platform, `np.clongdouble` is (almost always) an alias for
    **either** `complex256`, `complex192`, or (sometimes) `complex128`.

## "Flexible"

Scalar types with "flexible" length, whose values have a (constant) length
that depends on the specific `np.dtype` instantiation.

<table>
    <tr>
        <th align="center" colspan="2"><code>numpy._</code></th>
        <th align="center"><code>numpy.dtypes._</code></th>
        <th align="center" colspan="2"><code>optype.numpy._</code></th>
    </tr>
    <tr>
        <th>scalar</th>
        <th>scalar base</th>
        <th>dtype</th>
        <th>array-like</th>
        <th>dtype-like</th>
    </tr>
    <tr>
        <td><code>str_</code></td>
        <td rowspan="3"><code>character</code></td>
        <td><code>StrDType</code></td>
        <td><code>AnyStrArray</code></td>
        <td><code>AnyStrDType</code></td>
    </tr>
    <tr>
        <td rowspan="2"><code>bytes_</code></td>
        <td><code>BytesDType</code></td>
        <td rowspan="2"><code>AnyBytesArray</code></td>
        <td><code>AnyBytesDType</code></td>
    </tr>
    <tr>
        <td><code>dtype("c")</code></td>
        <td><code>AnyBytes8DType</code></td>
    </tr>
    <tr>
        <td><code>void</code></td>
        <td><code>flexible</code></td>
        <td><code>VoidDType</code></td>
        <td><code>AnyVoidArray</code></td>
        <td><code>AnyVoidDType</code></td>
    </tr>
</table>

## Other types

<table>
    <tr>
        <th align="center" colspan="2"><code>numpy._</code></th>
        <th align="center"><code>numpy.dtypes._</code></th>
        <th align="center" colspan="2"><code>optype.numpy._</code></th>
    </tr>
    <tr>
        <th>scalar</th>
        <th>scalar base</th>
        <th>dtype</th>
        <th>array-like</th>
        <th>dtype-like</th>
    </tr>
    <tr>
        <td><code>bool_</code></td>
        <td rowspan="3"><code>generic</code></td>
        <td><code>BoolDType</code></td>
        <td><code>AnyBoolArray</code></td>
        <td><code>AnyBoolDType</code></td>
    </tr>
    <tr>
        <td><code>object_</code></td>
        <td><code>ObjectDType</code></td>
        <td><code>AnyObjectArray</code></td>
        <td><code>AnyObjectDType</code></td>
    </tr>
    <tr>
        <td><code>datetime64</code></td>
        <td><code>DateTime64DType</code></td>
        <td><code>AnyDateTime64Array</code></td>
        <td><code>AnyDateTime64DType</code></td>
    </tr>
    <tr>
        <td><code>timedelta64</code></td>
        <td><i><code>generic</code></i></td>
        <td><code>TimeDelta64DType</code></td>
        <td><code>AnyTimeDelta64Array</code></td>
        <td><code>AnyTimeDelta64DType</code></td>
    </tr>
    <tr>
        <td colspan=2></td>
        <td><code>StringDType</code></td>
        <td><code>AnyStringArray</code></td>
        <td><code>AnyStringDType</code></td>
    </tr>
</table>

!!! info

    Since NumPy 2, `np.bool` is preferred over `np.bool_`, which only exists for
    backwards compatibility.

!!! info

    At runtime `np.timedelta64` is a subclass of `np.signedinteger`, but this is
    currently not reflected in the type annotations.

!!! info

    The `np.dypes.StringDType` has no associated numpy scalar type, and its `.type`
    attribute returns the `builtins.str` type instead. But from a typing perspective,
    such a `np.dtype[builtins.str]` isn't a valid type.
