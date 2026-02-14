# `DType`

In NumPy, a *dtype* (data type) object, is an instance of the
`numpy.dtype[ST: np.generic]` type.
It's commonly used to convey metadata of a scalar type, e.g. within arrays.

Because the type parameter of `np.dtype` isn't optional, it could be more
convenient to use the alias `optype.numpy.DType`, which is defined as:

```py
type DType[ST: np.generic = np.generic] = np.dtype[ST]
```

Apart from the "CamelCase" name, the only difference with `np.dtype` is that
the type parameter can be omitted, in which case it's equivalent to
`np.dtype[np.generic]`, but shorter.
