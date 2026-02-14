# Rich Relations

The "rich" comparison special methods often return a `bool`.
However, instances of any type can be returned (e.g. a numpy array).
This is why the corresponding `optype.Can*` interfaces accept a second type
argument for the return type, that defaults to `bool` when omitted.
The first type parameter matches the passed method argument, i.e. the
right-hand side operand, denoted here as `x`.

<table>
    <tr>
        <th colspan="4" align="center">operator</th>
        <th colspan="2" align="center">operand</th>
    </tr>
    <tr>
        <td>expression</td>
        <td>reflected</td>
        <th>function</th>
        <th>type</th>
        <td>method</td>
        <th>type</th>
    </tr>
    <tr>
        <td><code>_ == x</code></td>
        <td><code>x == _</code></td>
        <td><code>do_eq</code></td>
        <td><code>DoesEq</code></td>
        <td><code>__eq__</code></td>
        <td><code>CanEq[-T = object, +R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ != x</code></td>
        <td><code>x != _</code></td>
        <td><code>do_ne</code></td>
        <td><code>DoesNe</code></td>
        <td><code>__ne__</code></td>
        <td><code>CanNe[-T = object, +R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ < x</code></td>
        <td><code>x > _</code></td>
        <td><code>do_lt</code></td>
        <td><code>DoesLt</code></td>
        <td><code>__lt__</code></td>
        <td><code>CanLt[-T, +R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ <= x</code></td>
        <td><code>x >= _</code></td>
        <td><code>do_le</code></td>
        <td><code>DoesLe</code></td>
        <td><code>__le__</code></td>
        <td><code>CanLe[-T, +R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ > x</code></td>
        <td><code>x < _</code></td>
        <td><code>do_gt</code></td>
        <td><code>DoesGt</code></td>
        <td><code>__gt__</code></td>
        <td><code>CanGt[-T, +R = bool]</code></td>
    </tr>
    <tr>
        <td><code>_ >= x</code></td>
        <td><code>x <= _</code></td>
        <td><code>do_ge</code></td>
        <td><code>DoesGe</code></td>
        <td><code>__ge__</code></td>
        <td><code>CanGe[-T, +R = bool]</code></td>
    </tr>
</table>
