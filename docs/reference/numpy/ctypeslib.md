# `optype.numpy.ctypeslib`

Typed `ctypes` aliases for NumPy scalar-like C types.

`optype.numpy.ctypeslib` reexports standard-library `ctypes` constructors using
names that mirror NumPy scalar naming (for example, `Int32` for `np.int32`-like
typing), and it exposes abstract `ctypes` type aliases for static typing.

The module assumes a C99-compatible platform with either a 32- or 64-bit data model.

```python
import ctypes as ct
```

## Concrete types

<table>
<tr>
  <th align="left">alias</th>
  <th align="left"><code>ctypes</code></th>
  <th align="left">kind</th>
</tr>
<tr>
  <td align="left"><code>Bool</code></td>
  <td align="left"><code>c_bool</code></td>
  <td align="left">boolean</td>
</tr>
<tr>
  <td align="left"><code>Int8</code></td>
  <td align="left"><code>c_int8</code></td>
  <td align="left">fixed-width integer</td>
</tr>
<tr>
  <td align="left"><code>UInt8</code></td>
  <td align="left"><code>c_uint8</code></td>
  <td align="left">fixed-width integer</td>
</tr>
<tr>
  <td align="left"><code>Int16</code></td>
  <td align="left"><code>c_int16</code></td>
  <td align="left">fixed-width integer</td>
</tr>
<tr>
  <td align="left"><code>UInt16</code></td>
  <td align="left"><code>c_uint16</code></td>
  <td align="left">fixed-width integer</td>
</tr>
<tr>
  <td align="left"><code>Int32</code></td>
  <td align="left"><code>c_int32</code></td>
  <td align="left">fixed-width integer</td>
</tr>
<tr>
  <td align="left"><code>UInt32</code></td>
  <td align="left"><code>c_uint32</code></td>
  <td align="left">fixed-width integer</td>
</tr>
<tr>
  <td align="left"><code>Int64</code></td>
  <td align="left"><code>c_int64</code></td>
  <td align="left">fixed-width integer</td>
</tr>
<tr>
  <td align="left"><code>UInt64</code></td>
  <td align="left"><code>c_uint64</code></td>
  <td align="left">fixed-width integer</td>
</tr>
<tr>
  <td align="left"><code>Byte</code></td>
  <td align="left"><code>c_byte</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>UByte</code></td>
  <td align="left"><code>c_ubyte</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>Short</code></td>
  <td align="left"><code>c_short</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>UShort</code></td>
  <td align="left"><code>c_ushort</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>IntC</code></td>
  <td align="left"><code>c_int</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>UIntC</code></td>
  <td align="left"><code>c_uint</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>IntP</code></td>
  <td align="left"><code>c_ssize_t</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>UIntP</code></td>
  <td align="left"><code>c_size_t</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>Long</code></td>
  <td align="left"><code>c_long</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>ULong</code></td>
  <td align="left"><code>c_ulong</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>LongLong</code></td>
  <td align="left"><code>c_longlong</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>ULongLong</code></td>
  <td align="left"><code>c_ulonglong</code></td>
  <td align="left">platform-width integer</td>
</tr>
<tr>
  <td align="left"><code>Float32</code></td>
  <td align="left"><code>c_float</code></td>
  <td align="left">floating-point</td>
</tr>
<tr>
  <td align="left"><code>Float64</code></td>
  <td align="left"><code>c_double</code></td>
  <td align="left">floating-point</td>
</tr>
<tr>
  <td align="left"><code>LongDouble</code></td>
  <td align="left"><code>c_longdouble</code></td>
  <td align="left">floating-point</td>
</tr>
<tr>
  <td align="left"><code>Complex64</code></td>
  <td align="left"><code>c_float_complex</code></td>
  <td align="left">complex floating-point; <code>Never</code> on Windows or Python &lt; 3.14</td>
</tr>
<tr>
  <td align="left"><code>Complex128</code></td>
  <td align="left"><code>c_double_complex</code></td>
  <td align="left">complex floating-point; <code>Never</code> on Windows or Python &lt; 3.14</td>
</tr>
<tr>
  <td align="left"><code>CLongDouble</code></td>
  <td align="left"><code>c_longdouble_complex</code></td>
  <td align="left">complex floating-point; <code>Never</code> on Windows or Python &lt; 3.14</td>
</tr>
<tr>
  <td align="left"><code>Bytes</code></td>
  <td align="left"><code>c_char</code></td>
  <td align="left">character/byte</td>
</tr>
<tr>
  <td align="left"><code>Object</code></td>
  <td align="left"><code>py_object</code></td>
  <td align="left">Python object reference</td>
</tr>
</table>

## Abstract type aliases

<table>
<tr>
  <th align="left">alias</th>
  <th align="left">definition</th>
</tr>
<tr>
  <td align="left"><code>CType</code></td>
  <td align="left"><code>ct._SimpleCData[T]</code></td>
</tr>
<tr>
  <td align="left"><code>CScalar</code></td>
  <td align="left"><code>ct._CData</code></td>
</tr>
<tr>
  <td align="left"><code>Array</code></td>
  <td align="left"><code>ct.Array[CT] | ct.Array["_Array[CT]"]</code></td>
</tr>
<tr>
  <td align="left"><code>SignedInteger</code></td>
  <td align="left"><code>Int8 | Int16 | Int32 | Int64 | Short | IntC | IntP | Long | LongLong</code></td>
</tr>
<tr>
  <td align="left"><code>UnsignedInteger</code></td>
  <td align="left"><code>UInt8 | UInt16 | UInt32 | UInt64 | UShort | UIntC | UIntP | ULong | ULongLong</code></td>
</tr>
<tr>
  <td align="left"><code>Integer</code></td>
  <td align="left"><code>CScalar[int]</code></td>
</tr>
<tr>
  <td align="left"><code>Void</code></td>
  <td align="left"><code>ct.Structure | ct.Union</code></td>
</tr>
<tr>
  <td align="left"><code>Floating</code></td>
  <td align="left"><code>CScalar[float]</code></td>
</tr>
<tr>
  <td align="left"><code>ComplexFloating</code></td>
  <td align="left"><code>CScalar[complex]</code></td>
</tr>
<tr>
  <td align="left"><code>Inexact</code></td>
  <td align="left"><code>CScalar[float] | CScalar[complex]</code></td>
</tr>
<tr>
  <td align="left"><code>Number</code></td>
  <td align="left"><code>CScalar[int] | CScalar[float] | CScalar[complex]</code></td>
</tr>
<tr>
  <td align="left"><code>Flexible</code></td>
  <td align="left"><code>Bytes | Void</code></td>
</tr>
<tr>
  <td align="left"><code>Generic</code></td>
  <td align="left"><code>Bool | Number | Flexible | Object</code></td>
</tr>
</table>
