# Unary Operations

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
        <td><code>+_</code></td>
        <td><code>do_pos</code></td>
        <td><code>DoesPos</code></td>
        <td><code>__pos__</code></td>
        <td>
            <code>CanPos[+R]</code>,
            <code>CanPosSelf[+R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>-_</code></td>
        <td><code>do_neg</code></td>
        <td><code>DoesNeg</code></td>
        <td><code>__neg__</code></td>
        <td>
            <code>CanNeg[+R]</code>,
            <code>CanNegSelf[+R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>~_</code></td>
        <td><code>do_invert</code></td>
        <td><code>DoesInvert</code></td>
        <td><code>__invert__</code></td>
        <td>
            <code>CanInvert[+R]</code>,
            <code>CanInvertSelf[+R?]</code>
        </td>
    </tr>
    <tr>
        <td><code>abs(_)</code></td>
        <td><code>do_abs</code></td>
        <td><code>DoesAbs</code></td>
        <td><code>__abs__</code></td>
        <td>
            <code>CanAbs[+R]</code>,
            <code>CanAbsSelf[+R?]</code>
        </td>
    </tr>
</table>

The `Can*Self` variants return `-> Self` instead of `R`. Since optype 0.12.1 these
also accept an optional `R` type parameter (with a default of `Never`), which, when
provided, will result in a return type of `-> Self | R`.
