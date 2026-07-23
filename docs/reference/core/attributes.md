# Attributes

## Attribute access

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
        <td>
            <code>v = _.k</code> or<br/>
            <code>v = getattr(_, k)</code>
        </td>
        <td><code>do_getattr</code></td>
        <td><code>DoesGetattr</code></td>
        <td><code>__getattr__</code></td>
        <td><code>CanGetattr[+V=object]</code></td>
    </tr>
    <tr>
        <td>
            <code>_.k = v</code> or<br/>
            <code>setattr(_, k, v)</code>
        </td>
        <td><code>do_setattr</code></td>
        <td><code>DoesSetattr</code></td>
        <td><code>__setattr__</code></td>
        <td><code>CanSetattr[-V=object]</code></td>
    </tr>
    <tr>
        <td>
            <code>del _.k</code> or<br/>
            <code>delattr(_, k)</code>
        </td>
        <td><code>do_delattr</code></td>
        <td><code>DoesDelattr</code></td>
        <td><code>__delattr__</code></td>
        <td><code>CanDelattr</code></td>
    </tr>
    <tr>
        <td><code>dir(_)</code></td>
        <td><code>do_dir</code></td>
        <td><code>DoesDir</code></td>
        <td><code>__dir__</code></td>
        <td><code>CanDir[+R: Iterable[str]]</code></td>
    </tr>
</table>

## Instance attributes

<table>
    <tr>
        <th>attribute</th>
        <th>attribute type</th>
        <th>protocol</th>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align: middle"><code>__annotations__</code></td>
        <td><code>dict[str, Any]</code></td>
        <td><code>HasAnnotations</code></td>
    </tr>
    <tr>
        <td><code>D <: Mapping[str, object]</code></td>
        <td><code>HasAnnotations[+D]</code></td>
    </tr>
    <tr>
        <td><code>__class__</code></td>
        <td><code>T <: type</code></td>
        <td><code>HasClass[~T]</code></td>
    </tr>
    <tr>
        <td><code>__code__</code></td>
        <td><code>types.CodeType</code></td>
        <td><code>HasCode</code></td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align: middle"><code>__dict__</code></td>
        <td><code>dict[str, Any]</code></td>
        <td><code>HasDict</code></td>
    </tr>
    <tr>
        <td><code>D <: Mapping[str, object]</code></td>
        <td><code>HasDict[~D]</code></td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align: middle"><code>__doc__</code></td>
        <td><code>str | None</code></td>
        <td><code>HasDoc</code></td>
    </tr>
    <tr>
        <td><code>(S <: str) | None</code></td>
        <td><code>HasDoc[+S]</code></td>
    </tr>
    <tr>
        <td><code>__func__</code></td>
        <td><code>F <: Callable[..., object]</code></td>
        <td><code>HasFunc[+F]</code></td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align: middle"><code>__module__</code></td>
        <td><code>str</code></td>
        <td><code>HasModule</code></td>
    </tr>
    <tr>
        <td><code>S <: str</code></td>
        <td><code>HasModule[+S]</code></td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align: middle"><code>__name__</code></td>
        <td><code>str</code></td>
        <td><code>HasName</code></td>
    </tr>
    <tr>
        <td><code>S <: str</code></td>
        <td><code>HasName[~S]</code></td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align: middle"><code>__objclass__</code></td>
        <td><code>type</code></td>
        <td><code>HasObjclass</code></td>
    </tr>
    <tr>
        <td><code>T <: type</code></td>
        <td><code>HasObjclass[+T]</code></td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align: middle"><code>__qualname__</code></td>
        <td><code>str</code></td>
        <td><code>HasQualname</code></td>
    </tr>
    <tr>
        <td><code>S <: str</code></td>
        <td><code>HasQualname[~S]</code></td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align: middle"><code>__self__</code></td>
        <td><code>object</code></td>
        <td><code>HasSelf</code></td>
    </tr>
    <tr>
        <td><code>T</code></td>
        <td><code>HasSelf[+T]</code></td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align: middle"><code>__type_params__</code></td>
        <td><code>TypeParams</code></td>
        <td><code>HasTypeParams</code></td>
    </tr>
    <tr>
        <td><code>Ts <: TypeParams</code></td>
        <td><code>HasTypeParams[~Ts]</code></td>
    </tr>
    <tr>
        <td><code>__wrapped__</code></td>
        <td><code>F <: Callable[..., object]</code></td>
        <td><code>HasWrapped[+F]</code></td>
    </tr>
</table>

Here, `#!python type TypeParams = tuple[TypeVar | ParamSpec | TypeVarTuple, ...]`.

### `HasNames`

```python
type HasNames[N: str = str, Q: str = N] = HasName[N] & HasQualname[Q]
```

## Class attributes

<table>
    <tr>
        <th>attribute</th>
        <th>attribute type</th>
        <th>protocol</th>
    </tr>
    <tr>
        <td><code>__match_args__</code></td>
        <td><code>tuple[LiteralString, ...] | list[LiteralString]</code></td>
        <td><code>HasMatchArgs</code></td>
    </tr>
    <tr>
        <td><code>__slots__</code></td>
        <td><code>LiteralString | Iterable[LiteralString]</code></td>
        <td><code>HasSlots</code></td>
    </tr>
</table>
