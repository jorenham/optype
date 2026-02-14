# optype.dlpack

A collection of low-level types for working [DLPack](DOC-DLPACK).

## Protocols

<table>
    <tr>
        <th>type signature</th>
        <th>bound method</th>
    </tr>
    <tr>
        <td markdown="block">
        ```{ .plain .no-copy .no-select }
        CanDLPack[
            +T = int,
            +D: int = int,
        ]
        ```
        </td>
        <td markdown="block">
        ```{ .py .no-copy .no-select }
        def __dlpack__(
            *,
            stream: int | None = ...,
            max_version: tuple[int, int] | None = ...,
            dl_device: tuple[T, D] | None = ...,
            copy: bool | None = ...,
        ) -> types.CapsuleType: ...
        ```
        </td>
    </tr>
    <tr></tr>
    <tr>
        <td markdown="block">
            ```{ .plain .no-copy .no-select }
            CanDLPackDevice[
                +T = int,
                +D: int = int,
            ]
            ```
        </td>
        <td markdown="block">
            ```{ .py .no-copy .no-select }
            def __dlpack_device__() -> tuple[T, D]: ...
            ```
        </td>
    </tr>
</table>

The `+` prefix indicates that the type parameter is *co*variant.

## Enums

There are also two convenient
[`IntEnum`](https://docs.python.org/3/library/enum.html#enum.IntEnum)s
in `optype.dlpack`:

- `DLDeviceType` for the device types, and
- `DLDataTypeCode` for the internal type-codes of the `DLPack` data types.
