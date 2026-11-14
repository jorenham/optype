---
status: new
tags:
  - experimental
  - v0.16+
---

# optype.test

!!! warning "Experimental"

    The `optype.test` module is experimental and its API may change without notice.
    Use with caution.

The `optype.test` module provides assertions for testing static types. If
[beartype](https://github.com/beartype/beartype) is installed, they are also checked at
runtime.

### Runtime checking

With beartype installed, a failed assertion raises `AssertionError`, and the PEP 484
numeric tower applies. Without beartype, the assertions are no-ops, and the first one
emits an `UncheckedWarning`.

!!! note

    Runtime checking, `assert_type`, and `UncheckedWarning` were added in `optype 0.20.0`.

### `assert_subtype`

The `assert_subtype` function checks that a value is a subtype of (or assignable to) a
given type. It can be seen as a flexible alternative to `typing.assert_type`, that
allows for writing type-tests that are compatible with multiple type checkers.

For example, different type checkers may infer the literal `1` as `int` or `Literal[1]`,
causing `assert_type(1, int)` to be accepted by one type checker but not the other.
In contrast, `assert_subtype[int](1)` will be accepted by both type checkers.

!!! example

    ```pyi title="assert_subtype_example.pyi"
    import optype as op

    op.test.assert_subtype[int](True)  # (1)!
    op.test.assert_subtype[int](1)  # (2)!
    op.test.assert_subtype[int](1.0)  # (3)!
    ```

    1.  ✔️ `True` is a `bool`, which is a subtype of `int`, so type-checkers will accept this.
    2.  ✔️ `1` is an `int`, so type-checkers will accept this.
    3.  ❌ `1.0` is a `float`, which is not a subtype of `int`, so type-checkers will reject this.

### `assert_type`

Statically, this is `typing.assert_type`. At runtime, it performs the same check as
`assert_subtype`.

!!! example

    ```py title="assert_type_example.py"
    from optype.test import assert_type

    assert_type(1.5, float)  # (1)!
    assert_type(1.5, int)  # (2)!
    assert_type(True, int)  # (3)!
    ```

    1.  ✔️ Accepted statically and at runtime.
    2.  ❌ Rejected statically and at runtime.
    3.  ⚠️ Rejected statically, accepted at runtime.
