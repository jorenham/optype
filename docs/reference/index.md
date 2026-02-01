# Reference

## Overview

The API of `optype` is flat; a single `import optype as opt` is all you need (except for `optype.numpy`).

All typing protocols in `optype` are [runtime-checkable](https://typing.readthedocs.io/en/latest/spec/protocol.html#runtime-checkable-decorator-and-narrowing-types-by-isinstance), which means you can use `isinstance()` to check whether an object implements a protocol:

```python
import optype as op

isinstance('hello', op.CanAdd)  # True
isinstance(42, op.CanAbs)       # True
isinstance([1], op.CanGetitem)  # True
```

Unlike `collections.abc`, `optype`'s protocols aren't abstract base classes (they don't extend `abc.ABC`, only `typing.Protocol`). This allows the protocols to be used as building blocks for `.pyi` type stubs.

## The Five Flavors

There are five categories of types in `optype`:

### 1. `Just[T]` Types

The `optype.Just[T]` type and its subtypes (`JustInt`, `JustFloat`, `JustComplex`, etc.) only accept instances of the type itself, rejecting instances of strict subtypes.

This can be used to:

- Work around `float` and `complex` [type promotions](https://typing.readthedocs.io/en/latest/spec/special-types.html#special-cases-for-float-and-complex)
- Annotate `object()` sentinels with `Just[object]`
- Reject `bool` in functions that accept `int`

**See:** [Just Types Documentation](core/just.md)

### 2. `Can*` Protocols

`optype.Can*` types describe *what can be done* with an object.

For instance, any `CanAbs[T]` type can be used as an argument to the `abs()` builtin function with return type `T`. Most `Can*` protocols implement a single special method whose name directly matches that of the type:

- `CanAbs` implements `__abs__`
- `CanAdd` implements `__add__`
- `CanGetitem` implements `__getitem__`

**See:** Core Types sections for different operation categories

### 3. `Has*` Protocols

`optype.Has*` is the analogue of `Can*`, but for special *attributes*:

- `HasName` has a `__name__` attribute
- `HasDict` has a `__dict__` attribute
- `HasDoc` has a `__doc__` attribute

**See:** [Attributes Documentation](core/attributes.md)

### 4. `Does*` Types

`optype.Does*` types describe the *type of operators*.

- `DoesAbs` is the type of the `abs()` builtin function
- `DoesPos` is the type of the `+` unary prefix operator
- `DoesAdd` is the type of the `+` binary infix operator

**See:** Individual operation documentation pages

### 5. `do_*` Functions

`optype.do_*` are correctly-typed implementations of `Does*`. For each `do_*` there is a `Does*`, and vice-versa:

- `do_abs: DoesAbs` is a typed alias of `abs()`
- `do_pos: DoesPos` is a typed version of `operator.pos`

The `optype.do_*` operators are:

- More complete than `operator` module
- Have runtime-accessible type annotations
- Have names you don't need to know by heart

**See:** Individual operation documentation pages

## Core Types

These are the fundamental protocols for Python's builtin operations:

- **[Just](core/just.md)**: Exact type matching
- **[Type Conversion](core/conversion.md)**: `int()`, `float()`, `str()`, `bool()`, etc.
- **[Rich Relations](core/relations.md)**: `==`, `!=`, `<`, `<=`, `>`, `>=`
- **[Binary Operations](core/binary.md)**: `+`, `-`, `*`, `/`, `@`, `%`, `**`, etc.
- **[Reflected Operations](core/reflected.md)**: `__radd__`, `__rmul__`, etc.
- **[Inplace Operations](core/inplace.md)**: `+=`, `-=`, `*=`, etc.
- **[Unary Operations](core/unary.md)**: `+`, `-`, `~`, `abs()`, etc.
- **[Rounding](core/rounding.md)**: `round()`, `trunc()`, `floor()`, `ceil()`
- **[Callables](core/callables.md)**: `__call__`, function protocols
- **[Iteration](core/iteration.md)**: `iter()`, `next()`, `__iter__`, `__next__`
- **[Awaitables](core/awaitables.md)**: `await`, `__await__`
- **[Async Iteration](core/async-iteration.md)**: `async for`, `__aiter__`, `__anext__`
- **[Containers](core/containers.md)**: `len()`, `[]`, `in`, `reversed()`, etc.
- **[Attributes](core/attributes.md)**: `__name__`, `__dict__`, `__doc__`, etc.
- **[Context Managers](core/context.md)**: `with`, `async with`
- **[Descriptors](core/descriptors.md)**: `__get__`, `__set__`, `__delete__`
- **[Buffer Types](core/buffer.md)**: Memory views and buffer protocol

## Standard Library Modules

These modules provide protocols for Python's standard library:

- **[optype.copy](stdlib/copy.md)**: Shallow and deep copying
- **[optype.dataclasses](stdlib/dataclasses.md)**: Dataclass protocols
- **[optype.inspect](stdlib/inspect.md)**: Introspection protocols
- **[optype.io](stdlib/io.md)**: File I/O protocols
- **[optype.json](stdlib/json.md)**: JSON serialization
- **[optype.pickle](stdlib/pickle.md)**: Pickling protocols
- **[optype.string](stdlib/string.md)**: String formatting
- **[optype.typing](stdlib/typing.md)**: Typing utilities and aliases
- **[optype.dlpack](stdlib/dlpack.md)**: DLPack protocol for array interchange

## NumPy Support

NumPy-specific typing utilities (requires NumPy):

- **[Introduction](numpy/index.md)**: Overview of NumPy support
- **[Shape Typing](numpy/shape.md)**: Array shapes and dimensions
- **[Array-likes](numpy/array-likes.md)**: Array-like protocols
- **[Literals](numpy/literals.md)**: Literal types for NumPy
- **[Compatibility](numpy/compat.md)**: Cross-version compatibility
- **[Random](numpy/random.md)**: Random number generators
- **[DType](numpy/dtype.md)**: Data type objects
- **[Scalar](numpy/scalar.md)**: NumPy scalar types
- **[UFunc](numpy/ufunc.md)**: Universal functions
- **[Type Aliases](numpy/aliases.md)**: Common type aliases
- **[Low-level](numpy/low-level.md)**: Low-level NumPy interfaces

## Type Variance Notation

In the reference docs we use a fictional notation to describe the variance of generic type parameters:

- `~T`: invariant
- `+T`: covariant
- `-T`: contravariant

See the [typing spec](https://typing.python.org/en/latest/spec/generics.html#variance)
for an explanation
