# Descriptors

Interfaces for [descriptors](https://docs.python.org/3/howto/descriptor.html).

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
        <td>
            <code>v: V = T().d</code><br>
            <code>vt: VT = T.d</code>
        </td>
        <td><code>__get__</code></td>
        <td><code>CanGet[-T, +V, +VT=V]</code></td>
    </tr>
    <tr>
        <td>
            <code>v: V = T().d</code><br>
            <code>vt: Self = T.d</code>
        </td>
        <td><code>__get__</code></td>
        <td><code>CanGetSelf[-T, +V]</code></td>
    </tr>
    <tr>
        <td><code>T().k = v</code></td>
        <td><code>__set__</code></td>
        <td><code>CanSet[-T, -V]</code></td>
    </tr>
    <tr>
        <td><code>del T().k</code></td>
        <td><code>__delete__</code></td>
        <td><code>CanDelete[-T]</code></td>
    </tr>
    <tr>
        <td><code>class T: d = _</code></td>
        <td><code>__set_name__</code></td>
        <td><code>CanSetName[-T]</code></td>
    </tr>
</table>
