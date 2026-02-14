# Containers

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
        <td><code>len(_)</code></td>
        <td><code>do_len</code></td>
        <td><code>DoesLen</code></td>
        <td><code>__len__</code></td>
        <td><code>CanLen</code></td>
    </tr>
    <tr>
        <td>
            <code>_.__length_hint__()</code>
            (<a href="https://docs.python.org/3/reference/datamodel.html#object.__length_hint__">docs</a>)
        </td>
        <td><code>do_length_hint</code></td>
        <td><code>DoesLengthHint</code></td>
        <td><code>__length_hint__</code></td>
        <td><code>CanLengthHint</code></td>
    </tr>
    <tr>
        <td><code>_[k]</code></td>
        <td><code>do_getitem</code></td>
        <td><code>DoesGetitem</code></td>
        <td><code>__getitem__</code></td>
        <td><code>CanGetitem[-K, +V]</code></td>
    </tr>
    <tr>
        <td>
            <code>_.__missing__()</code>
            (<a href="https://docs.python.org/3/reference/datamodel.html#object.__missing__">docs</a>)
        </td>
        <td><code>do_missing</code></td>
        <td><code>DoesMissing</code></td>
        <td><code>__missing__</code></td>
        <td><code>CanMissing[-K, +D]</code></td>
    </tr>
    <tr>
        <td><code>_[k] = v</code></td>
        <td><code>do_setitem</code></td>
        <td><code>DoesSetitem</code></td>
        <td><code>__setitem__</code></td>
        <td><code>CanSetitem[-K, -V]</code></td>
    </tr>
    <tr>
        <td><code>del _[k]</code></td>
        <td><code>do_delitem</code></td>
        <td><code>DoesDelitem</code></td>
        <td><code>__delitem__</code></td>
        <td><code>CanDelitem[-K]</code></td>
    </tr>
    <tr>
        <td><code>k in _</code></td>
        <td><code>do_contains</code></td>
        <td><code>DoesContains</code></td>
        <td><code>__contains__</code></td>
        <td><code>CanContains[-K=object]</code></td>
    </tr>
    <tr>
        <td><code>reversed(_)</code></td>
        <td><code>do_reversed</code></td>
        <td><code>DoesReversed</code></td>
        <td><code>__reversed__</code></td>
        <td>
            <code>CanReversed[+R]</code>, or<br>
            <code>CanSequence[-I, +V]</code>
        </td>
    </tr>
</table>

Because `CanMissing[K, D]` generally doesn't show itself without
`CanGetitem[K, V]` there to hold its hand, `optype` conveniently stitched them
together as `optype.CanGetMissing[K, V, D=V]`.

Similarly, there is `optype.CanSequence[K: CanIndex | slice, V]`, which is the
combination of both `CanLen` and `CanItem[I, V]`, and serves as a more
specific and flexible `collections.abc.Sequence[V]`.
