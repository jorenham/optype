# Low-level Interfaces

Within `optype.numpy` there are several `Can*` (single-method) and `Has*`
(single-attribute) protocols, related to the `__array_*__` dunders of the
NumPy Python API.
These typing protocols are, just like the `optype.Can*` and `optype.Has*` ones,
runtime-checkable and extensible (i.e. not `@final`).

!!! tip

    All type parameters of these protocols can be omitted, which is equivalent
    to passing its upper type bound.

<table>
    <tr>
        <th>Protocol type signature</th>
        <th>Implements</th>
        <th>NumPy docs</th>
    </tr>
    <tr>
<td>

```{ .py .no-copy .no-select }
class CanArray[
    ND: tuple[int, ...] = ...,
    ST: np.generic = ...,
]: ...
```

</td>
<td>

<!-- blacken-docs:off -->

```{ .py .no-copy .no-select }
def __array__[RT = ST](
    _,
    dtype: DType[RT] | None = ...,
) -> Array[ND, RT]
```

<!-- blacken-docs:on -->

</td>
<td>
<a href="https://numpy.org/doc/stable/user/basics.interoperability.html#the-array-method">User Guide: Interoperability with NumPy</a>
</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```{ .py .no-copy .no-select }
class CanArrayUFunc[
    U: UFunc = ...,
    R: object = ...,
]: ...
```

</td>
<td>

```{ .py .no-copy .no-select }
def __array_ufunc__(
    _,
    ufunc: U,
    method: LiteralString,
    *args: object,
    **kwargs: object,
) -> R: ...
```

</td>
<td>
<a href="https://numpy.org/neps/nep-0013-ufunc-overrides.html">NEP 13</a>
</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```{ .py .no-copy .no-select }
class CanArrayFunction[
    F: CanCall[..., object] = ...,
    R = object,
]: ...
```

</td>
<td>

```{ .py .no-copy .no-select }
def __array_function__(
    _,
    func: F,
    types: CanIterSelf[type[CanArrayFunction]],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> R: ...
```

</td>
<td>
<a href="https://numpy.org/neps/nep-0018-array-function-protocol.html">NEP 18</a>
</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```{ .py .no-copy .no-select }
class CanArrayFinalize[
    T: object = ...,
]: ...
```

</td>
<td>

```{ .py .no-copy .no-select }
def __array_finalize__(_, obj: T): ...
```

</td>
<td>
<a href="https://numpy.org/doc/stable/user/basics.subclassing.html#the-role-of-array-finalize">User Guide: Subclassing ndarray</a>
</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```{ .py .no-copy .no-select }
class CanArrayWrap: ...
```

</td>
<td>

<!-- blacken-docs:off -->

```{ .py .no-copy .no-select }
def __array_wrap__[ND, ST](
    _,
    array: Array[ND, ST],
    context: (...) | None = ...,
    return_scalar: bool = ...,
) -> Self | Array[ND, ST]
```

<!-- blacken-docs:on -->

</td>
<td>
<a href="https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_wrap__">API: Standard array subclasses</a>
</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```{ .py .no-copy .no-select }
class HasArrayInterface[
    V: Mapping[str, object] = ...,
]: ...
```

</td>
<td>

```{ .py .no-copy .no-select }
__array_interface__: V
```

</td>
<td>
<a href="https://numpy.org/doc/stable/reference/arrays.interface.html#array-interface-protocol">API: The array interface protocol</a>
</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```{ .py .no-copy .no-select }
class HasArrayPriority: ...
```

</td>
<td>

```{ .py .no-copy .no-select }
__array_priority__: float
```

</td>
<td>

<a href="https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_priority__">API: Standard array subclasses</a>

</td>
    </tr>
    <tr><td colspan="3"></td></tr>
    <tr>
<td>

```{ .py .no-copy .no-select }
class HasDType[
    DT: DType = ...,
]: ...
```

</td>
<td>

```{ .py .no-copy .no-select }
dtype: DT
```

</td>
<td>
<a href="https://numpy.org/doc/stable/reference/arrays.dtypes.html#specifying-and-constructing-data-types">API: Specifying and constructing data types</a>
</td>
    </tr>
</table>
