# optype.json

Type aliases for the `json` standard library:

<table>
    <tr>
        <td><code>Value</code></td>
        <td><code>AnyValue</code></td>
    </tr>
    <tr>
        <th><code>json.load(s)</code> return type</th>
        <th><code>json.dumps(s)</code> input type</th>
    </tr>
    <tr>
        <td><code>Array[V: Value = Value]</code></td>
        <td><code>AnyArray[~V: AnyValue = AnyValue]</code></td>
    </tr>
    <tr>
        <td><code>Object[V: Value = Value]</code></td>
        <td><code>AnyObject[~V: AnyValue = AnyValue]</code></td>
    </tr>
</table>

The `(Any)Value` can be any json input, i.e. `Value | Array | Object` is
equivalent to `Value`.
It's also worth noting that `Value` is a subtype of `AnyValue`, which means
that `AnyValue | Value` is equivalent to `AnyValue`.
