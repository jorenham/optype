# `UFunc`

A large portion of NumPy's public API consists of *universal functions*, often
denoted as [ufuncs][DOC-UFUNC], which are (callable) instances of
[`np.ufunc`][REF_UFUNC].

!!! tip

    Custom ufuncs can be created using [`np.frompyfunc`][REF_FROMPY], but also
    through a user-defined class that implements the required attributes and
    methods (i.e., duck typing).

But `np.ufunc` has a big issue; it accepts no type parameters.
This makes it very difficult to properly annotate its callable signature and
its literal attributes (e.g. `.nin` and `.identity`).

This is where `optype.numpy.UFunc` comes into play:
It's a runtime-checkable generic typing protocol, that has been thoroughly
type- and unit-tested to ensure compatibility with all of numpy's ufunc
definitions.
Its generic type signature looks roughly like:

```python
type UFunc[
    # The type of the (bound) `__call__` method.
    Fn: CanCall = CanCall,
    # The types of the `nin` and `nout` (readonly) attributes.
    # Within numpy these match either `Literal[1]` or `Literal[2]`.
    Nin: int = int,
    Nout: int = int,
    # The type of the `signature` (readonly) attribute;
    # Must be `None` unless this is a generalized ufunc (gufunc), e.g.
    # `np.matmul`.
    Sig: str | None = str | None,
    # The type of the `identity` (readonly) attribute (used in `.reduce`).
    # Unless `Nin: Literal[2]`, `Nout: Literal[1]`, and `Sig: None`,
    # this should always be `None`.
    # Note that `complex` also includes `bool | int | float`.
    Id: complex | bytes | str | None = float | None,
] = ...
```

!!! note

    On older NumPy versions the extra callable methods of `np.ufunc` (`at`, `reduce`,
    `reduceat`, `accumulate`, and `outer`), are incorrectly annotated (as `None`
    *attributes*, even though at runtime they're methods that raise a
    `ValueError` when called).
    Until optype drops support for these older NumPy versions, it won't be possible to
    properly type these in `optype.numpy.UFunc`; doing so would make it incompatible with
    NumPy's ufuncs.

[DOC-UFUNC]: https://numpy.org/doc/stable/reference/ufuncs.html
[REF_UFUNC]: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.html
[REF_FROMPY]: https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html
