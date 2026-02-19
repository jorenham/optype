---
status: new
tags:
  - experimental
  - type-check only
  - v0.16+
---

# optype.test

!!! warning "Experimental"

    The `optype.test` module is experimental and its API may change without notice.
    Use with caution.

The type-check only `optype.test` module provides utilities for testing static types.
It can only be used in `if TYPE_CHECKING:` blocks in `.py` files, or in `.pyi` files
&mdash; it doesn't exist at runtime.

### `assert_subtype`

The `optype.test` module currently only contains the `assert_subtype` function, which
checks that a value is a subtype of (or assignable to) a given type. It can be seen
as a flexible alternative to `typing.assert_type`, that allows for writing type-tests
that are compatible with multiple type checkers.

For example, different type checkers may infer the literal `1` as `int` or `Literal[1]`,
causing `assert_type(1, int)` to be accepted by one type checker but not the other.
In contrast, `assert_subtype[int](1)` will be accepted by both type checkers.

!!! example

    ```pyi title="assert_subtype_example.pyi"
    import optype as op

    op.test.assert_subtype[int](True) # (1)!
    op.test.assert_subtype[int](1) # (2)!
    op.test.assert_subtype[int](1.0) # (3)!
    ```

    1.  ✔️ `True` is a `bool`, which is a subtype of `int`, so type-checkers will accept this.
    2.  ✔️ `1` is an `int`, so type-checkers will accept this.
    3.  ❌ `1.0` is a `float`, which is not a subtype of `int`, so type-checkers will reject this.
