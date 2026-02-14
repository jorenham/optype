# optype.pickle

For the [`pickle`][PK] standard library, `optype.pickle` provides the following
interfaces:

[PK]: https://docs.python.org/3/library/pickle.html

<table>
    <tr>
        <th>method(s)</th>
        <th>signature (bound)</th>
        <th>type</th>
    </tr>
    <tr>
        <td><code>__reduce__</code></td>
        <td><code>() -> R</code></td>
        <td><code>CanReduce[+R: str | tuple =]</code></td>
    </tr>
    <tr>
        <td><code>__reduce_ex__</code></td>
        <td><code>(CanIndex) -> R</code></td>
        <td><code>CanReduceEx[+R: str | tuple =]</code></td>
    </tr>
    <tr>
        <td><code>__getstate__</code></td>
        <td><code>() -> S</code></td>
        <td><code>CanGetstate[+S]</code></td>
    </tr>
    <tr>
        <td><code>__setstate__</code></td>
        <td><code>(S) -> None</code></td>
        <td><code>CanSetstate[-S]</code></td>
    </tr>
    <tr>
        <td>
            <code>__getnewargs__</code><br>
            <code>__new__</code>
        </td>
        <td>
            <code>() -> tuple[V, ...]</code><br>
            <code>(V) -> Self</code><br>
        </td>
        <td><code>CanGetnewargs[+V]</code></td>
    </tr>
    <tr>
        <td>
            <code>__getnewargs_ex__</code><br>
            <code>__new__</code>
        </td>
        <td>
            <code>() -> tuple[tuple[V, ...], dict[str, KV]]</code><br>
            <code>(*tuple[V, ...], **dict[str, KV]) -> Self</code><br>
        </td>
        <td><code>CanGetnewargsEx[+V, ~KV]</code></td>
    </tr>
</table>

<!--
# `optype.rick`

I'm a pickle.
-->
