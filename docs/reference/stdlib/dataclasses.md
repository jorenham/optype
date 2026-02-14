# optype.dataclasses

For the [`dataclasses`][DC] standard library, `optype.dataclasses` provides the
`HasDataclassFields[V: Mapping[str, Field]]` interface.
It can conveniently be used to check whether a type or instance is a dataclass,
i.e. `isinstance(obj, HasDataclassFields)`.

<!-- TODO(@jorenham): Examples -->

[DC]: https://docs.python.org/3/library/dataclasses.html
