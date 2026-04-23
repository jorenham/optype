# `optype.numpy.ctypeslib`

Typed `ctypes` aliases for NumPy scalar-like C types.

`optype.numpy.ctypeslib` reexports standard-library `ctypes` constructors using
names that mirror NumPy scalar naming (for example, `Int32` for `np.int32`-like
typing), and it exposes abstract `ctypes` type aliases for static typing.

The module assumes a C99-compatible platform with either a 32- or 64-bit data model.

## Reexported `ctypes` constructors

<table>
<tr>
  <th align="left"><code>optype.numpy.ctypeslib</code></th>
  <th align="left"><code>ctypes</code></th>
  <th align="left">kind</th>
  <th align="left">notes</th>
</tr>
<tr>
  <td align="left"><code>Bool</code></td>
  <td align="left"><code>c_bool</code></td>
  <td align="left">boolean</td>
  <td align="left"></td>
</tr>
<tr>
  <td align="left"><code>Int8</code>, <code>UInt8</code>, <code>Int16</code>, <code>UInt16</code>, <code>Int32</code>, <code>UInt32</code>, <code>Int64</code>, <code>UInt64</code></td>
  <td align="left"><code>c_int8</code>, <code>c_uint8</code>, <code>c_int16</code>, <code>c_uint16</code>, <code>c_int32</code>, <code>c_uint32</code>, <code>c_int64</code>, <code>c_uint64</code></td>
  <td align="left">fixed-width integers</td>
  <td align="left"></td>
</tr>
<tr>
  <td align="left"><code>Byte</code>, <code>UByte</code>, <code>Short</code>, <code>UShort</code>, <code>IntC</code>, <code>UIntC</code>, <code>IntP</code>, <code>UIntP</code>, <code>Long</code>, <code>ULong</code>, <code>LongLong</code>, <code>ULongLong</code></td>
  <td align="left"><code>c_byte</code>, <code>c_ubyte</code>, <code>c_short</code>, <code>c_ushort</code>, <code>c_int</code>, <code>c_uint</code>, <code>c_ssize_t</code>, <code>c_size_t</code>, <code>c_long</code>, <code>c_ulong</code>, <code>c_longlong</code>, <code>c_ulonglong</code></td>
  <td align="left">platform-width integers</td>
  <td align="left"><code>UIntP</code> uses <code>c_size_t</code> (<code>void_p</code> on NumPy &lt; 2, but almost always equivalent).</td>
</tr>
<tr>
  <td align="left"><code>Float32</code>, <code>Float64</code>, <code>LongDouble</code></td>
  <td align="left"><code>c_float</code>, <code>c_double</code>, <code>c_longdouble</code></td>
  <td align="left">floating-point</td>
  <td align="left"><code>LongDouble</code> is mainly useful as a type annotation target.</td>
</tr>
<tr>
  <td align="left"><code>Complex64</code>, <code>Complex128</code>, <code>CLongDouble</code></td>
  <td align="left"><code>c_float_complex</code>, <code>c_double_complex</code>, <code>c_longdouble_complex</code></td>
  <td align="left">complex floating-point</td>
  <td align="left">Available on Python 3.14+ on non-Windows only; otherwise these aliases are <code>Never</code>.</td>
</tr>
<tr>
  <td align="left"><code>Bytes</code></td>
  <td align="left"><code>c_char</code></td>
  <td align="left">character/byte</td>
  <td align="left">Not numeric; participates in <code>Flexible</code>.</td>
</tr>
<tr>
  <td align="left"><code>Object</code></td>
  <td align="left"><code>py_object</code></td>
  <td align="left">Python object reference</td>
  <td align="left"></td>
</tr>
</table>

## Abstract type aliases

<table>
<tr>
	<th align="left">alias</th>
	<th align="left">covers</th>
	<th align="left">notes</th>
</tr>
<tr>
	<td align="left"><code>CType</code></td>
	<td align="left"><code>ctypes._SimpleCData[T]</code></td>
	<td align="left">Scalar-value ctypes base type.</td>
</tr>
<tr>
	<td align="left"><code>CScalar</code></td>
	<td align="left"><code>ctypes._CData</code></td>
	<td align="left">Broad base for ctypes objects (scalar, array, pointer, structure, union, ...).</td>
</tr>
<tr>
	<td align="left"><code>Array</code></td>
	<td align="left"><code>ct.Array[CT]</code> (including nested arrays)</td>
	<td align="left"><code>CT</code> is bound to <code>CType</code>.</td>
</tr>
<tr>
	<td align="left"><code>SignedInteger</code>, <code>UnsignedInteger</code></td>
	<td align="left">signed and unsigned integer constructor groups</td>
	<td align="left"></td>
</tr>
<tr>
	<td align="left"><code>Integer</code>, <code>Floating</code>, <code>ComplexFloating</code></td>
	<td align="left">numeric scalar families</td>
	<td align="left">Modeled via <code>CScalar</code> for static typing.</td>
</tr>
<tr>
	<td align="left"><code>Inexact</code>, <code>Number</code></td>
	<td align="left">combined numeric families</td>
	<td align="left"><code>Inexact = Floating | ComplexFloating</code>, <code>Number = Integer | Floating | ComplexFloating</code>.</td>
</tr>
<tr>
	<td align="left"><code>Void</code></td>
	<td align="left"><code>ct.Structure | ct.Union</code></td>
	<td align="left">Not a constructor reexport; structural ctypes only.</td>
</tr>
<tr>
	<td align="left"><code>Flexible</code></td>
	<td align="left"><code>Bytes | Void</code></td>
	<td align="left"></td>
</tr>
<tr>
	<td align="left"><code>Generic</code></td>
	<td align="left"><code>Bool | Number | Flexible | Object</code></td>
	<td align="left">Top-level union for this module.</td>
</tr>
</table>
