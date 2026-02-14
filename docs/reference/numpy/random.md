# `random` submodule

[SPEC 7](https://scientific-python.org/specs/spec-0007/) -compatible type aliases.
The `optype.numpy.random` module provides three type aliases: `RNG`, `ToRNG`, and
`ToSeed`.

In general, the most useful one is `ToRNG`, which describes what can be
passed to `numpy.random.default_rng`. It is defined as the union of `RNG`, `ToSeed`,
and `numpy.random.BitGenerator`.

The `RNG` is the union type of `numpy.random.Generator` and its legacy dual type,
`numpy.random.RandomState`.

`ToSeed` accepts integer-like scalars, sequences, and arrays, as well as instances of
`numpy.random.SeedSequence`.
