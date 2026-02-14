# optype.io

A collection of protocols and type-aliases that, unlike their analogues in `_typeshed`,
are accessible at runtime, and use a consistent naming scheme.

<table>
    <tr>
        <th><code>optype.io</code> protocol</th>
        <th>implements</th>
        <th>replaces</th>
    </tr>
    <tr>
        <td><code>CanFSPath[+T: str | bytes =]</code></td>
        <td><code>__fspath__: () -> T</code></td>
        <td><code>os.PathLike[AnyStr: (str, bytes)]</code></td>
    </tr>
    <tr>
        <td><code>CanRead[+T]</code></td>
        <td><code>read: () -> T</code></td>
        <td></td>
    </tr>
    <tr>
        <td><code>CanReadN[+T]</code></td>
        <td><code>read: (int) -> T</code></td>
        <td><code>_typeshed.SupportsRead[+T]</code></td>
    </tr>
    <tr>
        <td><code>CanReadline[+T]</code></td>
        <td><code>readline: () -> T</code></td>
        <td><code>_typeshed.SupportsNoArgReadline[+T]</code></td>
    </tr>
    <tr>
        <td><code>CanReadlineN[+T]</code></td>
        <td><code>readline: (int) -> T</code></td>
        <td><code>_typeshed.SupportsReadline[+T]</code></td>
    </tr>
    <tr>
        <td><code>CanWrite[-T, +RT = object]</code></td>
        <td><code>write: (T) -> RT</code></td>
        <td><code>_typeshed.SupportsWrite[-T]</code></td>
    </tr>
    <tr>
        <td><code>CanFlush[+RT = object]</code></td>
        <td><code>flush: () -> RT</code></td>
        <td><code>_typeshed.SupportsFlush</code></td>
    </tr>
    <tr>
        <td><code>CanFileno</code></td>
        <td><code>fileno: () -> int</code></td>
        <td><code>_typeshed.HasFileno</code></td>
    </tr>
</table>

<table>
    <tr>
        <th><code>optype.io</code> type alias</th>
        <th>expression</th>
        <th>replaces</th>
    </tr>
    <tr>
        <td><code>ToPath[+T: str | bytes =]</code></td>
        <td><code>T | CanFSPath[T]</code></td>
        <td>
            <code>_typeshed.StrPath</code><br>
            <code>_typeshed.BytesPath</code><br>
            <code>_typeshed.StrOrBytesPath</code><br>
            <code>_typeshed.GenericPath[AnyStr]</code><br>
        </td>
    </tr>
    <tr>
        <td><code>ToFileno</code></td>
        <td><code>int | CanFileno</code></td>
        <td><code>_typeshed.FileDescriptorLike</code></td>
    </tr>
</table>
