# Shape Typing

## Array aliases

Optype provides the generic `onp.Array` type alias for `np.ndarray`.
It is similar to `npt.NDArray`, but includes two (optional) type parameters:
one that matches the *shape type* (`ND: tuple[int, ...]`),
and one that matches the *scalar type* (`ST: np.generic`).

When we put the definitions of `npt.NDArray` and `onp.Array` side-by-side,
their differences become clear:

<table>
    <tr>
        <th markdown="1"><code>numpy.typing.NDArray</code></th>
        <th><code>optype.numpy.Array</code></th>
        <th><code>optype.numpy.ArrayND</code></th>
    </tr>
    <tr>
        <td>
            ```{ .py .no-copy .no-select }
            type NDArray[
                # no shape type
                SCT: generic,  # no default
            ] = ndarray[Any, dtype[SCT]]
            ```
        </td>
        <td>
            ```{ .py .no-copy .no-select }
            type Array[
                NDT: (int, ...) = (int, ...),
                SCT: generic = generic,
            ] = ndarray[NDT, dtype[SCT]]
            ```
        </td>
        <td>
            ```{ .py .no-copy .no-select }
            type ArrayND[
                SCT: generic = generic,
                NDT: (int, ...) = (int, ...),
            ] = ndarray[NDT, dtype[SCT]]
            ```
        </td>
    </tr>
</table>

Additionally, there are the four `Array{0,1,2,3}D` aliases, which are
equivalent to `Array` with `tuple[()]`, `tuple[int]`, `tuple[int, int]` and
`tuple[int, int, int]` as shape-type, respectively.

!!! info

    Since `numpy>=2.2` the `NDArray` alias uses `tuple[int, ...]` as shape-type
    instead of `Any`.

!!! tip

    Before NumPy 2.1, the shape type parameter of `ndarray` (i.e. the type of
    `ndarray.shape`) was invariant. It is therefore recommended to not use `Literal`
    within shape types on `numpy<2.1`. So with `numpy>=2.1` you can use
    `tuple[Literal[3], Literal[3]]` without problem, but with `numpy<2.1` you should use
    `tuple[int, int]` instead.

    See [numpy/numpy#25729](https://github.com/numpy/numpy/issues/25729) and
    [numpy/numpy#26081](https://github.com/numpy/numpy/pull/26081) for details.

In the same way as `ArrayND` for `ndarray` (shown for reference), its subtypes
`np.ma.MaskedArray` and `np.matrix` are also aliased:

<table>
    <tr>
        <th><code>ArrayND</code> (<code>np.ndarray</code>)</th>
        <th><code>MArray</code> (<code>np.ma.MaskedArray</code>)</th>
        <th><code>Matrix</code> (<code>np.matrix</code>)</th>
    </tr>
    <tr>
        <td>
            ```{ .py .no-copy .no-select }
            type ArrayND[
                SCT: generic = generic,
                NDT: (int, ...) = (int, ...),
            ] = ndarray[NDT, dtype[SCT]]
            ```
        </td>
        <td>
            ```{ .py .no-copy .no-select }
            type MArray[
                SCT: generic = generic,
                NDT: (int, ...) = (int, ...),
            ] = ma.MaskedArray[NDT, dtype[SCT]]
            ```
        </td>
        <td>
            ```{ .py .no-copy .no-select }
            type Matrix[
                SCT: generic = generic,
                M: int = int,
                N: int = M,
            ] = matrix[(M, N), dtype[SCT]]
            ```
        </td>
    </tr>
</table>

For masked arrays with specific `ndim`, you could also use one of the four
`MArray{0,1,2,3}D` aliases.

## Array typeguards

To check whether a given object is an instance of `Array{0,1,2,3,N}D`, in a way that
static type-checkers also understand it, the following [PEP 742][PEP742] typeguards can
be used:

<table>
    <tr>
        <th>typeguard</th>
        <th>narrows to</th>
        <th>shape type</th>
    </tr>
    <tr>
        <th colspan="2"><code>optype.numpy._</code></th>
        <th><code>builtins._</code></th>
    </tr>
    <tr>
        <td><code>is_array_nd</code></td>
        <td><code>ArrayND[ST]</code></td>
        <td><code>tuple[int, ...]</code></td>
    </tr>
    <tr>
        <td><code>is_array_0d</code></td>
        <td><code>Array0D[ST]</code></td>
        <td><code>tuple[()]</code></td>
    </tr>
    <tr>
        <td><code>is_array_1d</code></td>
        <td><code>Array1D[ST]</code></td>
        <td><code>tuple[int]</code></td>
    </tr>
    <tr>
        <td><code>is_array_2d</code></td>
        <td><code>Array2D[ST]</code></td>
        <td><code>tuple[int, int]</code></td>
    </tr>
    <tr>
        <td><code>is_array_3d</code></td>
        <td><code>Array3D[ST]</code></td>
        <td><code>tuple[int, int, int]</code></td>
    </tr>
</table>

These functions additionally accept an optional `dtype` argument, that can either be
a `np.dtype[ST]` instance, a `type[ST]`, or something that has a `dtype: np.dtype[ST]`
attribute.
The signatures are almost identical to each other, and in the `0d` case it roughly
looks like this:

```py
_T = TypeVar("_T", bound=np.generic, default=Any)
_ToDType: TypeAlias = type[_T] | np.dtype[_T] | HasDType[np.dtype[_T]]


def is_array_0d(a, /, dtype: _ToDType[_T] | None = None) -> TypeIs[Array0D[_T]]: ...
```

## Shape aliases

A *shape* is nothing more than a tuple of (non-negative) integers, i.e.
an instance of `tuple[int, ...]` such as `(42,)`, `(480, 720, 3)` or `()`.
The length of a shape is often referred to as the *number of dimensions*
or the *dimensionality* of the array or scalar.
For arrays this is accessible through the `np.ndarray.ndim`, which is
an alias for `len(np.ndarray.shape)`.

!!! info

    Before NumPy 2, the maximum number of dimensions was `32`, but has since
    been increased to `ndim <= 64`.

To make typing the shape of an array easier, optype provides two families of
shape type aliases: `AtLeast{N}D` and `AtMost{N}D`.
The `{N}` should be replaced by the number of dimensions, which currently
is limited to `0`, `1`, `2`, and `3`.

Both of these families are generic, and their (optional) type parameters must
be either `int` (default), or a literal (non-negative) integer, i.e. like
`typing.Literal[N: int]`.

The names `AtLeast{N}D` and `AtMost{N}D` are pretty much as self-explanatory:

- `AtLeast{N}D` is a `tuple[int, ...]` with `ndim >= N`
- `AtMost{N}D` is a `tuple[int, ...]` with `ndim <= N`

The shape aliases are roughly defined as:

<table>
<tr><th>
    <code>N</code>
</th><th>
    <code>ndim >= N</code>
</th><th>
    <code>ndim <= N</code>
</th></tr>
<tr><td>
0
</td><td>

```{ .py .no-copy .no-select }
type AtLeast0D = (int, ...)
```

</td><td>

```{ .py .no-copy .no-select }
type AtMost0D = ()
```

</td></tr>
<tr><td colspan="4"></td></tr>
<tr><td>
1
</td><td>

```{ .py .no-copy .no-select }
type AtLeast1D = (int, *AtLeast0D)
```

</td><td>

```{ .py .no-copy .no-select }
type AtMost1D = AtMost0D | (int,)
```

</td></tr>
<tr><td colspan="4"></td></tr>
<tr><td>
2
</td><td>

```{ .py .no-copy .no-select }
type AtLeast2D = (
    tuple[int, int]
    | AtLeast3D[int]
)
```

</td><td>

```{ .py .no-copy .no-select }
type AtMost2D = AtMost1D | (int, int)
```

</td></tr>
<tr><td colspan="4"></td></tr>
<tr><td>
3
</td><td>

```{ .py .no-copy .no-select }
type AtLeast3D = (
    tuple[int, int, int]
    | tuple[int, int, int, int]
    | tuple[int, int, int, int, int]
    # etc...
)
```

</td><td>

```{ .py .no-copy .no-select }
type AtMost3D = AtMost2D | (int, int, int)
```

</td></tr>
</table>

The `AtLeast{}D` optionally accepts a type argument that can either be `int` (default),
or `Any`. Passing `Any` turns it from a *gradual tuple type*, so that they can also be
assigned to compatible bounded shape-types. So `AtLeast1D[Any]` is assignable to
`tuple[int]`, whereas `AtLeast1D` (equiv. `AtLeast1D[int]`) is not.

However, mypy currently has a [bug](https://github.com/python/mypy/issues/19109),
causing it to falsely reject such gradual shape-type assignment for N=1 or up.

[PEP742]: https://peps.python.org/pep-0742/
