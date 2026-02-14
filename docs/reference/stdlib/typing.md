# optype.typing

#### `Any*` type aliases

Type aliases for anything that can *always* be passed to
`int`, `float`, `complex`, `iter`, or `typing.Literal`

<table>
    <tr>
        <th>Python constructor</th>
        <th><code>optype.typing</code> alias</th>
    </tr>
    <tr>
        <td><code>int(_)</code></td>
        <td><code>AnyInt</code></td>
    </tr>
    <tr>
        <td><code>float(_)</code></td>
        <td><code>AnyFloat</code></td>
    </tr>
    <tr>
        <td><code>complex(_)</code></td>
        <td><code>AnyComplex</code></td>
    </tr>
    <tr>
        <td><code>iter(_)</code></td>
        <td><code>AnyIterable</code></td>
    </tr>
    <tr>
        <td><code>typing.Literal[_]</code></td>
        <td><code>AnyLiteral</code></td>
    </tr>
</table>

!!!note

    Even though *some* `str` and `bytes` can be converted to `int`, `float`,
    `complex`, most of them can't, and are therefore not included in these
    type aliases.

#### `Empty*` type aliases

These are builtin types or collections that are empty, i.e. have length 0 or
yield no elements.

<table>
    <tr>
        <th>instance</th>
        <th><code>optype.typing</code> type</th>
    </tr>
    <tr>
        <td><code>''</code></td>
        <td><code>EmptyString</code></td>
    </tr>
    <tr>
        <td><code>b''</code></td>
        <td><code>EmptyBytes</code></td>
    </tr>
    <tr>
        <td><code>()</code></td>
        <td><code>EmptyTuple</code></td>
    </tr>
    <tr>
        <td><code>[]</code></td>
        <td><code>EmptyList</code></td>
    </tr>
    <tr>
        <td><code>{}</code></td>
        <td><code>EmptyDict</code></td>
    </tr>
    <tr>
        <td><code>set()</code></td>
        <td><code>EmptySet</code></td>
    </tr>
    <tr>
        <td><code>(i for i in range(0))</code></td>
        <td><code>EmptyIterable</code></td>
    </tr>
</table>

#### Literal types

<table>
    <tr>
        <th>Literal values</th>
        <th><code>optype.typing</code> type</th>
        <th>Notes</th>
    </tr>
    <tr>
        <td><code>{False, True}</code></td>
        <td><code>LiteralFalse</code></td>
        <td>
            Similar to <code>typing.LiteralString</code>, but for
            <code>bool</code>.
        </td>
    </tr>
    <tr>
        <td><code>{0, 1, ..., 255}</code></td>
        <td><code>LiteralByte</code></td>
        <td>
            Integers in the range 0-255, that make up a <code>bytes</code>
            or <code>bytearray</code> objects.
        </td>
    </tr>
</table>
