# Inplace Operations

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
            <code>CanIAdd[-T, +R]</code><br>
            <code>CanIAddSelf[-T]</code><br>
            <code>CanIAddSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ -= x</code></td>
        <td><code>do_isub</code></td>
        <td><code>DoesISub</code></td>
        <td><code>__isub__</code></td>
        <td>
            <code>CanISub[-T, +R]</code><br>
            <code>CanISubSelf[-T]</code><br>
            <code>CanISubSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ *= x</code></td>
        <td><code>do_imul</code></td>
        <td><code>DoesIMul</code></td>
        <td><code>__imul__</code></td>
        <td>
            <code>CanIMul[-T, +R]</code><br>
            <code>CanIMulSelf[-T]</code><br>
            <code>CanIMulSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ @= x</code></td>
        <td><code>do_imatmul</code></td>
        <td><code>DoesIMatmul</code></td>
        <td><code>__imatmul__</code></td>
        <td>
            <code>CanIMatmul[-T, +R]</code><br>
            <code>CanIMatmulSelf[-T]</code><br>
            <code>CanIMatmulSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ /= x</code></td>
        <td><code>do_itruediv</code></td>
        <td><code>DoesITruediv</code></td>
        <td><code>__itruediv__</code></td>
        <td>
            <code>CanITruediv[-T, +R]</code><br>
            <code>CanITruedivSelf[-T]</code><br>
            <code>CanITruedivSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ //= x</code></td>
        <td><code>do_ifloordiv</code></td>
        <td><code>DoesIFloordiv</code></td>
        <td><code>__ifloordiv__</code></td>
        <td>
            <code>CanIFloordiv[-T, +R]</code><br>
            <code>CanIFloordivSelf[-T]</code><br>
            <code>CanIFloordivSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ %= x</code></td>
        <td><code>do_imod</code></td>
        <td><code>DoesIMod</code></td>
        <td><code>__imod__</code></td>
        <td>
            <code>CanIMod[-T, +R]</code><br>
            <code>CanIModSelf[-T]</code><br>
            <code>CanIModSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ **= x</code></td>
        <td><code>do_ipow</code></td>
        <td><code>DoesIPow</code></td>
        <td><code>__ipow__</code></td>
        <td>
            <code>CanIPow[-T, +R]</code><br>
            <code>CanIPowSelf[-T]</code><br>
            <code>CanIPowSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ <<= x</code></td>
        <td><code>do_ilshift</code></td>
        <td><code>DoesILshift</code></td>
        <td><code>__ilshift__</code></td>
        <td>
            <code>CanILshift[-T, +R]</code><br>
            <code>CanILshiftSelf[-T]</code><br>
            <code>CanILshiftSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ >>= x</code></td>
        <td><code>do_irshift</code></td>
        <td><code>DoesIRshift</code></td>
        <td><code>__irshift__</code></td>
        <td>
            <code>CanIRshift[-T, +R]</code><br>
            <code>CanIRshiftSelf[-T]</code><br>
            <code>CanIRshiftSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ &= x</code></td>
        <td><code>do_iand</code></td>
        <td><code>DoesIAnd</code></td>
        <td><code>__iand__</code></td>
        <td>
            <code>CanIAnd[-T, +R]</code><br>
            <code>CanIAndSelf[-T]</code><br>
            <code>CanIAndSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ ^= x</code></td>
        <td><code>do_ixor</code></td>
        <td><code>DoesIXor</code></td>
        <td><code>__ixor__</code></td>
        <td>
            <code>CanIXor[-T, +R]</code><br>
            <code>CanIXorSelf[-T]</code><br>
            <code>CanIXorSame[-T?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ |= x</code></td>
        <td><code>do_ior</code></td>
        <td><code>DoesIOr</code></td>
        <td><code>__ior__</code></td>
        <td>
            <code>CanIOr[-T, +R]</code><br>
            <code>CanIOrSelf[-T]</code><br>
            <code>CanIOrSame[-T?]</code>
        </td>
    </tr>
</table>

These inplace operators usually return themselves (after some in-place mutation).
But unfortunately, it currently isn't possible to use `Self` for this (i.e.
something like `type MyAlias[T] = optype.CanIAdd[T, Self]` isn't allowed).
So to help ease this unbearable pain, `optype` comes equipped with ready-made
aliases for you to use. They bear the same name, with an additional `*Self`
suffix, e.g. `optype.CanIAddSelf[T]`.

!!! note

    The `CanI*Self` protocols method return `typing.Self` and optionally accept `T`. The
    `CanI*Same` protocols also return `Self`, but instead accept `rhs: Self | T`. Since
    `T` defaults to `Never`, it will accept `rhs: Self | Never` if `T` is not provided,
    which is equivalent to `rhs: Self`.

    *Available since `0.12.1`*
