# Metaclasses

Interfaces for the metaclass hooks behind
[`isinstance()`](https://docs.python.org/3/library/functions.html#isinstance) and
[`issubclass()`](https://docs.python.org/3/library/functions.html#issubclass).
These special methods are looked up on the type (i.e. the metaclass) of the operand.

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
        <td><code>isinstance(x, _)</code></td>
        <td><code>__instancecheck__</code></td>
        <td><code>CanInstancecheck</code></td>
    </tr>
    <tr>
        <td><code>issubclass(c, _)</code></td>
        <td><code>__subclasscheck__</code></td>
        <td><code>CanSubclasscheck</code></td>
    </tr>
</table>

!!! note

    Unlike every other `Can*` protocol, `CanSubclasscheck` is **not**
    `@runtime_checkable`, so it cannot be used with `isinstance()` or
    `issubclass()`. Its `__subclasscheck__` member shadows the method that
    `ABCMeta` invokes internally during a runtime protocol check, which would
    otherwise crash. It remains fully usable for static typing.
