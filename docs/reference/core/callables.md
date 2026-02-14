# Callables

Unlike `operator`, `optype` provides an operator for callable objects:
`optype.do_call(f, *args. **kwargs)`.

`CanCall` is similar to `collections.abc.Callable`, but is runtime-checkable,
and doesn't use esoteric hacks.

<table>
  <thead>
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
    </thead>
    <tbody>
    <tr>
        <td><code>_(*args, **kwargs)</code></td>
        <td><code>do_call</code></td>
        <td><code>DoesCall</code></td>
        <td><code>__call__</code></td>
        <td><code>CanCall[**Tss, +R]</code></td>
    </tr>
  </tbody>
</table>

!!! note

    Pyright (and probably other typecheckers) tend to accept `collections.abc.Callable`
    in more places than `optype.CanCall`. This could be related to the lack of
    co-/contra-variance specification for `typing.ParamSpec` (they should almost always
    be contravariant, but currently they can only be invariant).

    In case you encounter such a situation, please open an issue about it, so we can
    investigate further.
