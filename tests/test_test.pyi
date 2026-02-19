from optype.test import assert_subtype

assert_subtype[int](True)
assert_subtype[int](1)
assert_subtype[int](1.0)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
assert_subtype[int](object())  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
