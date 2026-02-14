# optype.copy

For the [`copy`][PY-COPY] standard library, `optype.copy` provides the following
runtime-checkable interfaces:

<table>
    <tr>
        <th align="center"><code>copy</code> standard library</th>
        <th colspan="2" align="center"><code>optype.copy</code></th>
    </tr>
    <tr>
        <td>function</td>
        <th>type</th>
        <th>method</th>
    </tr>
    <tr>
        <td><code>copy.copy(_) -> R</code></td>
        <td><code>__copy__() -> R</code></td>
        <td><code>CanCopy[+R]</code></td>
    </tr>
    <tr>
        <td><code>copy.deepcopy(_, memo={}) -> R</code></td>
        <td><code>__deepcopy__(memo, /) -> R</code></td>
        <td><code>CanDeepcopy[+R]</code></td>
    </tr>
    <tr>
        <td><code>copy.replace(_, /, **changes) -> R</code></td>
        <td><code>__replace__(**changes) -> R</code></td>
        <td><code>CanReplace[+R]</code></td>
    </tr>
</table>

!!! note

    [`copy.replace`](PY-COPY-REPLACE) requires `python>=3.13`, but
    `optype.copy.CanReplace` is available in all versions of Python.

In practice, it makes sense that a copy of an instance is the same type as the
original.
But because `typing.Self` cannot be used as a type argument, this difficult
to properly type.
Instead, you can use the `optype.copy.Can{}Self` types, which are the
runtime-checkable equivalents of the following (non-expressible) aliases:

```python
type CanCopySelf = CanCopy[Self]
type CanDeepcopySelf = CanDeepcopy[Self]
type CanReplaceSelf = CanReplace[Self]
```

[PY-COPY]: https://docs.python.org/3/library/copy.html
[PY-COPY-REPLACE]: https://docs.python.org/3/library/copy.html#copy.replace
