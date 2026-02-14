# Iteration

The operand `x` of `iter(_)` is within Python known as an *iterable*, which is
what `collections.abc.Iterable[V]` is often used for (e.g. as base class, or
for instance checking).

The `optype` analogue is `CanIter[R]`, which as the name suggests,
also implements `__iter__`. But unlike `Iterable[V]`, its type parameter `R`
binds to the return type of `iter(_) -> R`. This makes it possible to annotate
the specific type of the *iterable* that `iter(_)` returns. `Iterable[V]` is
only able to annotate the type of the iterated value. To see why that isn't
possible, see [python/typing#548](https://github.com/python/typing/issues/548).

The `collections.abc.Iterator[V]` is even more awkward; it is a subtype of
`Iterable[V]`. For those familiar with `collections.abc` this might come as a
surprise, but an iterator only needs to implement `__next__`, `__iter__` isn't
needed. This means that the `Iterator[V]` is unnecessarily restrictive.
Apart from that being theoretically "ugly", it has significant performance
implications, because the time-complexity of `isinstance` on a
`typing.Protocol` is $O(n)$, with the $n$ referring to the amount of members.
So even if the overhead of the inheritance and the `abc.ABC` usage is ignored,
`collections.abc.Iterator` is twice as slow as it needs to be.

That's one of the (many) reasons that `optype.CanNext[V]` and
`optype.CanIter[R]` are the better alternatives to `Iterable` and `Iterator`
from the abracadabra collections. This is how they are defined:

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
        <td><code>next(_)</code></td>
        <td><code>do_next</code></td>
        <td><code>DoesNext</code></td>
        <td><code>__next__</code></td>
        <td><code>CanNext[+V]</code></td>
    </tr>
    <tr>
        <td><code>iter(_)</code></td>
        <td><code>do_iter</code></td>
        <td><code>DoesIter</code></td>
        <td><code>__iter__</code></td>
        <td><code>CanIter[+R: CanNext[object]]</code></td>
    </tr>
</table>

For the sake of compatibility with `collections.abc`, there is
`optype.CanIterSelf[V]`, which is a protocol whose `__iter__` returns
`typing.Self`, as well as a `__next__` method that returns `T`.
I.e. it is equivalent to `collections.abc.Iterator[V]`, but without the `abc`
nonsense.
