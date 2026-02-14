# Array-likes

Similar to the `numpy._typing._ArrayLike{}_co` *coercible array-like* types,
`optype.numpy` provides the `optype.numpy.To{}ND`. Unlike the ones in `numpy`, these
don't accept "bare" scalar types (the `__len__` method is required).
Additionally, there are the `To{}1D`, `To{}2D`, and `To{}3D` for vector-likes,
matrix-likes, and cuboid-likes, and the `To{}` aliases for "bare" scalar types.

<table>
<tr>
    <th align="left"><code>builtins</code></th>
    <th align="left"><code>numpy</code></th>
    <th align="center" colspan="4"><code>optype.numpy</code></th>
</tr>
<tr>
    <th align="center" colspan="2"><i>exact</i> scalar types</th>
    <th align="center">scalar-like</th>
    <th align="center"><code>{1,2,3,N}</code>-d array-like</th>
    <th align="center">strict <code>{1,2,3}</code>-d array-like</th>
</tr>
<tr>
    <td align="left"><code>False</code></td>
    <td align="left"><code>False_</code></td>
    <td align="left"><code>ToJustFalse</code></td>
    <td align="left"></td>
    <td align="left"></td>
</tr>
<tr>
    <td align="left">
        <code>False</code><br>
        <code>| 0</code>
    </td>
    <td align="left"><code>False_</code></td>
    <td align="left"><code>ToFalse</code></td>
    <td align="left"></td>
    <td align="left"></td>
</tr>
<tr>
    <td align="left"><code>True</code></td>
    <td align="left"><code>True_</code></td>
    <td align="left"><code>ToJustTrue</code></td>
    <td align="left"></td>
    <td align="left"></td>
</tr>
<tr>
    <td align="left">
        <code>True</code><br>
        <code>| 1</code>
    </td>
    <td align="left"><code>True_</code></td>
    <td align="left"><code>ToTrue</code></td>
    <td align="left"></td>
    <td align="left"></td>
</tr>
<tr>
    <td align="left"><code>bool</code></td>
    <td align="left"><code>bool_</code></td>
    <td align="left"><code>ToJustBool</code></td>
    <td align="left"><code>ToJustBool{}D</code></td>
    <td align="left"><code>ToJustBoolStrict{}D</code></td>
</tr>
<tr>
    <td align="left">
        <code>bool</code><br>
        <code>| 0</code><br>
        <code>| 1</code>
    </td>
    <td align="left"><code>bool_</code></td>
    <td align="left"><code>ToBool</code></td>
    <td align="left"><code>ToBool{}D</code></td>
    <td align="left"><code>ToBoolStrict{}D</code></td>
</tr>
<tr>
    <td align="left"><code>~int</code></td>
    <td align="left"><code>integer</code></td>
    <td align="left"><code>ToJustInt</code></td>
    <td align="left"><code>ToJustInt{}D</code></td>
    <td align="left"><code>ToJustIntStrict{}D</code></td>
</tr>
<tr>
    <td align="left">
        <code>int</code><br>
        <code>| bool</code>
    </td>
    <td align="left">
        <code>integer</code><br>
        <code>| bool_</code>
    </td>
    <td align="left"><code>ToInt</code></td>
    <td align="left"><code>ToInt{}D</code></td>
    <td align="left"><code>ToIntStrict{}D</code></td>
</tr>
<tr>
    <td align="left"></td>
    <td align="left"><code>float16</code></td>
    <td align="left"><code>ToJustFloat16</code></td>
    <td align="left"><code>ToJustFloat16_{}D</code></td>
    <td align="left"><code>ToJustFloat16Strict{}D</code></td>
</tr>
<tr>
    <td align="left"></td>
    <td align="left">
        <code>float16</code><br>
        <code>| int8</code><br>
        <code>| uint8</code><br>
        <code>| bool_</code>
    </td>
    <td align="left"><code>ToFloat32</code></td>
    <td align="left"><code>ToFloat32_{}D</code></td>
    <td align="left"><code>ToFloat32Strict{}D</code></td>
</tr>
<tr>
    <td align="left"></td>
    <td align="left"><code>float32</code></td>
    <td align="left"><code>ToJustFloat32</code></td>
    <td align="left"><code>ToJustFloat32_{}D</code></td>
    <td align="left"><code>ToJustFloat32Strict{}D</code></td>
</tr>
<tr>
    <td align="left"></td>
    <td align="left">
        <code>float32</code><br>
        <code>| float16</code><br>
        <code>| int16</code><br>
        <code>| uint16</code><br>
        <code>| int8</code><br>
        <code>| uint8</code><br>
        <code>| bool_</code>
    </td>
    <td align="left"><code>ToFloat32</code></td>
    <td align="left"><code>ToFloat32_{}D</code></td>
    <td align="left"><code>ToFloat32Strict{}D</code></td>
</tr>
<tr>
    <td align="left"><code>~float</code></td>
    <td align="left"><code>float64</code></td>
    <td align="left"><code>ToJustFloat64</code></td>
    <td align="left"><code>ToJustFloat64_{}D</code></td>
    <td align="left"><code>ToJustFloat64Strict{}D</code></td>
</tr>
<tr>
    <td align="left">
        <code>float</code><br>
        <code>| int</code><br>
        <code>| bool</code>
    </td>
    <td align="left">
        <code>float64</code><br>
        <code>| float32</code><br>
        <code>| float16</code><br>
        <code>| integer</code><br>
        <code>| bool_</code>
    </td>
    <td align="left"><code>ToFloat64</code></td>
    <td align="left"><code>ToFloat64_{}D</code></td>
    <td align="left"><code>ToFloat64Strict{}D</code></td>
</tr>
<tr>
    <td align="left"><code>~float</code></td>
    <td align="left"><code>floating</code></td>
    <td align="left"><code>ToJustFloat</code></td>
    <td align="left"><code>ToJustFloat{}D</code></td>
    <td align="left"><code>ToJustFloatStrict{}D</code></td>
</tr>
<tr>
    <td align="left">
        <code>float</code><br>
        <code>| int</code><br>
        <code>| bool</code>
    </td>
    <td align="left">
        <code>floating</code><br>
        <code>| integer</code><br>
        <code>| bool_</code>
    </td>
    <td align="left"><code>ToFloat</code></td>
    <td align="left"><code>ToFloat{}D</code></td>
    <td align="left"><code>ToFloatStrict{}D</code></td>
</tr>
<tr>
    <td align="left"></td>
    <td align="left"><code>complex64</code></td>
    <td align="left"><code>ToJustComplex64</code></td>
    <td align="left"><code>ToJustComplex64_{}D</code></td>
    <td align="left"><code>ToJustComplex64Strict{}D</code></td>
</tr>
<tr>
    <td align="left"></td>
    <td align="left">
        <code>complex64</code><br>
        <code>| float32</code><br>
        <code>| float16</code><br>
        <code>| int16</code><br>
        <code>| uint16</code><br>
        <code>| int8</code><br>
        <code>| uint8</code><br>
        <code>| bool_</code>
    </td>
    <td align="left"><code>ToComplex64</code></td>
    <td align="left"><code>ToComplex64_{}D</code></td>
    <td align="left"><code>ToComplex64Strict{}D</code></td>
</tr>
<tr>
    <td align="left"><code>~complex</code></td>
    <td align="left"><code>complex128</code></td>
    <td align="left"><code>ToJustComplex128</code></td>
    <td align="left"><code>ToJustComplex128_{}D</code></td>
    <td align="left"><code>ToJustComplex128Strict{}D</code></td>
</tr>
<tr>
    <td align="left">
        <code>complex</code><br>
        <code>| float</code><br>
        <code>| int</code><br>
        <code>| bool</code>
    </td>
    <td align="left">
        <code>complex128</code><br>
        <code>| complex64</code><br>
        <code>| float64</code><br>
        <code>| float32</code><br>
        <code>| float16</code><br>
        <code>| integer</code><br>
        <code>| bool_</code>
    </td>
    <td align="left"><code>ToComplex128</code></td>
    <td align="left"><code>ToComplex128_{}D</code></td>
    <td align="left"><code>ToComplex128Strict{}D</code></td>
</tr>
<tr>
    <td align="left"><code>~complex</code></td>
    <td align="left"><code>complexfloating</code></td>
    <td align="left"><code>ToJustComplex</code></td>
    <td align="left"><code>ToJustComplex{}D</code></td>
    <td align="left"><code>ToJustComplexStrict{}D</code></td>
</tr>
<tr>
    <td align="left">
        <code>complex</code><br>
        <code>| float</code><br>
        <code>| int</code><br>
        <code>| bool</code>
    </td>
    <td align="left">
        <code>number</code><br>
        <code>| bool_</code>
    </td>
    <td align="left"><code>ToComplex</code></td>
    <td align="left"><code>ToComplex{}D</code></td>
    <td align="left"><code>ToComplexStrict{}D</code></td>
</tr>
<tr>
    <td align="left">
        <code>complex</code><br>
        <code>| float</code><br>
        <code>| int</code><br>
        <code>| bool</code>
        <code>| bytes</code><br>
        <code>| str</code><br>
    </td>
    <td align="left"><code>generic</code></td>
    <td align="left"><code>ToScalar</code></td>
    <td align="left"><code>ToArray{}D</code></td>
    <td align="left"><code>ToArrayStrict{}D</code></td>
</tr>
</table>

!!! note

    The `To*Strict{1,2,3}D` aliases were added in `optype 0.7.3`.

    These array-likes with *strict shape-type* require the shape-typed input to be
    shape-typed.
    This means that e.g. `ToFloat1D` and `ToFloat2D` are disjoint (non-overlapping),
    and makes them suitable to overload array-likes of a particular dtype for different
    numbers of dimensions.

!!! note

    The `ToJust{Bool,Float,Complex}*` type aliases were added in `optype 0.8.0`.

    See [`optype.Just`](#just) for more information.

!!! note

    The `To[Just]{False,True}` type aliases were added in `optype 0.9.1`.

    These only include the `np.bool` types on `numpy>=2.2`. Before that, `np.bool`
    wasn't generic, making it impossible to distinguish between `np.False_` and `np.True_`
    using static typing.

!!! note

    The `ToArrayStrict{1,2,3}D` types are generic since `optype 0.9.1`, analogous to
    their non-strict dual type, `ToArray{1,2,3}D`.

!!! note

    The `To[Just]{Float16,Float32,Complex64}*` type aliases were added in `optype 0.12.0`.

Source code: [`optype/numpy/_to.py`][CODE-NP-TO]

[CODE-NP-TO]: https://github.com/jorenham/optype/blob/master/optype/numpy/_to.py
