# Binary Operations

In the [Python docs][NT], these are referred to as "arithmetic operations".
But the operands aren't limited to numeric types, and because the
operations aren't required to be commutative, might be non-deterministic, and
could have side-effects.
Classifying them "arithmetic" is, at the very least, a bit of a stretch.

[NT]: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

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
        <td>
            <code>CanAdd[-T, +R = T]</code><br>
            <code>CanAddSelf[-T]</code><br>
            <code>CanAddSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ - x</code></td>
        <td><code>do_sub</code></td>
        <td><code>DoesSub</code></td>
        <td><code>__sub__</code></td>
        <td>
            <code>CanSub[-T, +R = T]</code><br>
            <code>CanSubSelf[-T]</code><br>
            <code>CanSubSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ * x</code></td>
        <td><code>do_mul</code></td>
        <td><code>DoesMul</code></td>
        <td><code>__mul__</code></td>
        <td>
            <code>CanMul[-T, +R = T]</code><br>
            <code>CanMulSelf[-T]</code><br>
            <code>CanMulSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ @ x</code></td>
        <td><code>do_matmul</code></td>
        <td><code>DoesMatmul</code></td>
        <td><code>__matmul__</code></td>
        <td>
            <code>CanMatmul[-T, +R = T]</code><br>
            <code>CanMatmulSelf[-T]</code><br>
            <code>CanMatmulSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ / x</code></td>
        <td><code>do_truediv</code></td>
        <td><code>DoesTruediv</code></td>
        <td><code>__truediv__</code></td>
        <td>
            <code>CanTruediv[-T, +R = T]</code><br>
            <code>CanTruedivSelf[-T]</code><br>
            <code>CanTruedivSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ // x</code></td>
        <td><code>do_floordiv</code></td>
        <td><code>DoesFloordiv</code></td>
        <td><code>__floordiv__</code></td>
        <td>
            <code>CanFloordiv[-T, +R = T]</code><br>
            <code>CanFloordivSelf[-T]</code><br>
            <code>CanFloordivSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ % x</code></td>
        <td><code>do_mod</code></td>
        <td><code>DoesMod</code></td>
        <td><code>__mod__</code></td>
        <td>
            <code>CanMod[-T, +R = T]</code><br>
            <code>CanModSelf[-T]</code><br>
            <code>CanModSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>divmod(_, x)</code></td>
        <td><code>do_divmod</code></td>
        <td><code>DoesDivmod</code></td>
        <td><code>__divmod__</code></td>
        <td><code>CanDivmod[-T, +R]</code></td>
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
            <code>CanPow2[-T, +R = T]</code><br>
            <code>CanPowSelf[-T]</code><br>
            <code>CanPowSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>pow(_, x, m)</code></td>
        <td><code>do_pow/3</code></td>
        <td><code>DoesPow</code></td>
        <td><code>__pow__</code></td>
        <td><code>CanPow3[-T, -M, +R = int]</code></td>
    </tr>
    <tr>
        <td><code>_ << x</code></td>
        <td><code>do_lshift</code></td>
        <td><code>DoesLshift</code></td>
        <td><code>__lshift__</code></td>
        <td>
            <code>CanLshift[-T, +R = T]</code><br>
            <code>CanLshiftSelf[-T]</code><br>
            <code>CanLshiftSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ >> x</code></td>
        <td><code>do_rshift</code></td>
        <td><code>DoesRshift</code></td>
        <td><code>__rshift__</code></td>
        <td>
            <code>CanRshift[-T, +R = T]</code><br>
            <code>CanRshiftSelf[-T]</code><br>
            <code>CanRshiftSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ & x</code></td>
        <td><code>do_and</code></td>
        <td><code>DoesAnd</code></td>
        <td><code>__and__</code></td>
        <td>
            <code>CanAnd[-T, +R = T]</code><br>
            <code>CanAndSelf[-T]</code><br>
            <code>CanAndSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ ^ x</code></td>
        <td><code>do_xor</code></td>
        <td><code>DoesXor</code></td>
        <td><code>__xor__</code></td>
        <td>
            <code>CanXor[-T, +R = T]</code><br>
            <code>CanXorSelf[-T]</code><br>
            <code>CanXorSame[-T?, +R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>_ | x</code></td>
        <td><code>do_or</code></td>
        <td><code>DoesOr</code></td>
        <td><code>__or__</code></td>
        <td>
            <code>CanOr[-T, +R = T]</code><br>
            <code>CanOrSelf[-T]</code><br>
            <code>CanOrSame[-T?, +R?]</code>
        </td>
    </tr>
</table>

!!! tip

    Because `pow()` can take an optional third argument, `optype` provides separate
    interfaces for `pow()` with two and three arguments. Additionally, there is the
    overloaded intersection type
    `type CanPow[-T, -M, +R, +RM] = CanPow2[T, R] & CanPow3[T, M, RM]`, as interface
    for types that can take an optional third argument.

!!! note

    The `Can*Self` protocols method return `typing.Self` and optionally accept `T` and
    `R`. The `Can*Same` protocols also return `Self`, but instead accept `Self | T`,
    with `T` and `R` optional generic type parameters that default to `typing.Never`.
    To illustrate, `CanAddSelf[T]` implements `__add__` as `(self, rhs: T, /) -> Self`,
    while `CanAddSame[T, R]` implements it as `(self, rhs: Self | T, /) -> Self | R`,
    and `CanAddSame` (without `T` and `R`) as `(self, rhs: Self, /) -> Self`.
