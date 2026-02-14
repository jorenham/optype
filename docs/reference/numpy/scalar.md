# `Scalar`

The `optype.numpy.Scalar` interface is a generic runtime-checkable protocol,
that can be seen as a "more specific" `np.generic`, both in name, and from
a typing perspective.

Its type signature looks roughly like this:

```py
type Scalar[
    # The "Python type", so that `Scalar.item() -> PT`.
    PT: object,
    # The "N-bits" type (without having to deal with `npt.NBitBase`).
    # It matches the `itemsize: NB` property.
    NB: int = int,
] = ...
```

It can be used as e.g.

```py
are_birds_real: Scalar[bool, Literal[1]] = np.bool_(True)
the_answer: Scalar[int, Literal[2]] = np.uint16(42)
alpha: Scalar[float, Literal[8]] = np.float64(1 / 137)
```

!!!note

    The second type argument for `itemsize` can be omitted, which is equivalent
    to setting it to `int`, so `Scalar[PT]` and `Scalar[PT, int]` are equivalent.
