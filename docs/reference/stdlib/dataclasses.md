# optype.dataclasses

For the [`dataclasses`][DC] standard library, `optype.dataclasses` provides the
`HasDataclassFields` interface.
It can conveniently be used to check whether a type or instance is a dataclass,
i.e. `isinstance(obj, HasDataclassFields)`.

!!! warning Breaking change

    Starting in v0.17 `HasDataclassFields` is no longer generic. Previously it was generic  -- `HasDataclassFields[V: Mapping[str, Field]]`.

<!-- TODO(@jorenham): Examples -->

[DC]: https://docs.python.org/3/library/dataclasses.html
