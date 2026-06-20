---
status: new
tags:
  - experimental
  - v0.18+
---

# optype.infer

!!! warning "Experimental"

    The `optype.infer` module is experimental and its API may change without notice.

The `optype.infer` module works out which `optype` protocols a function requires of its
parameters. It runs the function against recording proxies that trace every operation,
then renders the result as a [PEP 695](https://peps.python.org/pep-0695/) signature.

This page documents the Python API. The same inference is available from the command
line as `optype infer`; see [Type Inference](../../infer.md) for a full tour of the
behavior, with worked examples of overloads, intersections, branches, variadics, and
more.

## `infer`

```python
def infer(func, /, *params: str | int) -> str: ...
```

Infer the `optype` protocol(s) required of `func`'s parameters, returned as the inferred
[PEP 695](https://peps.python.org/pep-0695/) signature string:

```pycon
>>> from optype.infer import infer
>>> infer(lambda x: x + 1)
'[R](x: CanAdd[Literal[1], R]) -> R'
```

Pass parameter names or positions to report only those parameters:

```pycon
>>> infer(lambda x, y: x[y], "x")
'[T, R](x: CanGetitem[T, R]) -> R'
```

Raises [`InferError`](#infererror) when `func` is not supported.

## `InferError`

A subclass of `NotImplementedError`, raised when `infer` does not support the given
function, such as a non-callable, an operation without a matching protocol, or a
parameter that requires a value no placeholder can provide.

## `InferWarning`

A subclass of `RuntimeWarning`, emitted when `infer` could not explore the function
exhaustively, such as when the branch budget runs out. The returned signature then only
covers the explored forms.
