# Reflected Operations

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
        <td>
            <code>CanRAdd[-T, +R=T]</code><br>
            <code>CanRAddSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x - _</code></td>
        <td><code>do_rsub</code></td>
        <td><code>DoesRSub</code></td>
        <td><code>__rsub__</code></td>
        <td>
            <code>CanRSub[-T, +R=T]</code><br>
            <code>CanRSubSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x * _</code></td>
        <td><code>do_rmul</code></td>
        <td><code>DoesRMul</code></td>
        <td><code>__rmul__</code></td>
        <td>
            <code>CanRMul[-T, +R=T]</code><br>
            <code>CanRMulSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x @ _</code></td>
        <td><code>do_rmatmul</code></td>
        <td><code>DoesRMatmul</code></td>
        <td><code>__rmatmul__</code></td>
        <td>
            <code>CanRMatmul[-T, +R=T]</code><br>
            <code>CanRMatmulSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x / _</code></td>
        <td><code>do_rtruediv</code></td>
        <td><code>DoesRTruediv</code></td>
        <td><code>__rtruediv__</code></td>
        <td>
            <code>CanRTruediv[-T, +R=T]</code><br>
            <code>CanRTruedivSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x // _</code></td>
        <td><code>do_rfloordiv</code></td>
        <td><code>DoesRFloordiv</code></td>
        <td><code>__rfloordiv__</code></td>
        <td>
            <code>CanRFloordiv[-T, +R=T]</code><br>
            <code>CanRFloordivSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x % _</code></td>
        <td><code>do_rmod</code></td>
        <td><code>DoesRMod</code></td>
        <td><code>__rmod__</code></td>
        <td>
            <code>CanRMod[-T, +R=T]</code><br>
            <code>CanRModSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>divmod(x, _)</code></td>
        <td><code>do_rdivmod</code></td>
        <td><code>DoesRDivmod</code></td>
        <td><code>__rdivmod__</code></td>
        <td><code>CanRDivmod[-T, +R]</code></td>
    </tr>
    <tr>
        <td>
            <code>x ** _</code><br/>
            <code>pow(x, _)</code>
        </td>
        <td><code>do_rpow</code></td>
        <td><code>DoesRPow</code></td>
        <td><code>__rpow__</code></td>
        <td>
            <code>CanRPow[-T, +R=T]</code><br>
            <code>CanRPowSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x << _</code></td>
        <td><code>do_rlshift</code></td>
        <td><code>DoesRLshift</code></td>
        <td><code>__rlshift__</code></td>
        <td>
            <code>CanRLshift[-T, +R=T]</code><br>
            <code>CanRLshiftSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x >> _</code></td>
        <td><code>do_rrshift</code></td>
        <td><code>DoesRRshift</code></td>
        <td><code>__rrshift__</code></td>
        <td>
            <code>CanRRshift[-T, +R=T]</code><br>
            <code>CanRRshiftSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x & _</code></td>
        <td><code>do_rand</code></td>
        <td><code>DoesRAnd</code></td>
        <td><code>__rand__</code></td>
        <td>
            <code>CanRAnd[-T, +R=T]</code><br>
            <code>CanRAndSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x ^ _</code></td>
        <td><code>do_rxor</code></td>
        <td><code>DoesRXor</code></td>
        <td><code>__rxor__</code></td>
        <td>
            <code>CanRXor[-T, +R=T]</code><br>
            <code>CanRXorSelf[-T]</code>
        </td>
    </tr>
    <tr>
        <td><code>x | _</code></td>
        <td><code>do_ror</code></td>
        <td><code>DoesROr</code></td>
        <td><code>__ror__</code></td>
        <td>
            <code>CanROr[-T, +R=T]</code><br>
            <code>CanROrSelf[-T]</code>
        </td>
    </tr>
</table>

!!! note

    `CanRPow` corresponds to `CanPow2`; the 3-parameter "modulo" `pow` does not reflect
    in Python.

    According to the relevant [python docs][RPOW]:

    > Note that ternary `pow()` will not try calling `__rpow__()` (the coercion
    > rules would become too complicated).

[RPOW]: https://docs.python.org/3/reference/datamodel.html#object.__rpow__
