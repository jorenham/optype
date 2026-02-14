# Context Managers

Support for the `with` statement.

<table>
    <tr>
        <th align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>method(s)</td>
        <th>type(s)</th>
    </tr>
    <tr>
        <td></td>
        <td><code>__enter__</code></td>
        <td>
            <code>CanEnter[+C]</code>, or <br>
            <code>CanEnterSelf</code>
        </td>
    </tr>
    <tr>
        <td></td>
        <td><code>__exit__</code></td>
        <td><code>CanExit[+R = None]</code></td>
    </tr>
    <tr>
        <td><code>with _ as c: ...</code></td>
        <td><code>__enter__ & __exit__</code></td>
        <td>
            <code>CanWith[+C, +R = None]</code>, or<br>
            <code>CanWithSelf[+R = None]</code>
        </td>
    </tr>
</table>

`CanEnterSelf` and `CanWithSelf` are (runtime-checkable) aliases for
`CanEnter[Self]` and `CanWith[Self, R]`, respectively.

For the `async with` statement the interfaces look very similar:
