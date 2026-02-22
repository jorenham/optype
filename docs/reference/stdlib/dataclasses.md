# optype.dataclasses

For the [`dataclasses`][DC] standard library, `optype.dataclasses` provides the
`HasDataclassFields` interface.
It can conveniently be used to check whether a type or instance is a dataclass,
i.e. `isinstance(obj, HasDataclassFields)`.

!!! warning "Breaking change"
`HasDataclassFields` was previously generic (`HasDataclassFields[V: Mapping[str, Field]]`),
but has been deparametrized and is no longer generic.

<!-- TODO(@jorenham): Examples -->

[DC]: https://docs.python.org/3/library/dataclasses.html
