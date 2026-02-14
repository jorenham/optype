# Awaitables

The `optype.CanAwait[R]` is almost the same as `collections.abc.Awaitable[R]`, except
that `optype.CanAwait[R]` is a pure interface, whereas `Awaitable` is
also an abstract base class (making it absolutely useless when writing stubs).

<table>
    <tr>
        <th align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>await _</code></td>
        <td><code>__await__</code></td>
        <td><code>CanAwait[+R]</code></td>
    </tr>
</table>
