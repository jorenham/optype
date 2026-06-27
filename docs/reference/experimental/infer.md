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
def infer(
    func, /, *params: str | int, strict: bool = False, backend: Backend = TERSE
) -> str: ...
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

Raises [`InferError`](#infererror) when `func` is not supported. When exploration is
incomplete it emits an [`InferWarning`](#inferwarning) and still returns a provisional
signature; pass `strict=True` to raise an `InferError` instead.

Pass a [`backend`](#backend) to control how the signature is rendered.

## `Backend`

A `Protocol` with one method, `render(sig: Signature) -> str`. The default `TERSE`
backend emits the compact form shown above; any object with a matching `render` works:

```pycon
>>> from optype.infer import TERSE, infer
>>> class Upper:
...     def render(self, sig, /):
...         return TERSE.render(sig).upper()
>>> infer(lambda x: x + 1, backend=Upper())
'[R](X: CANADD[LITERAL[1], R]) -> R'
```

## `InferError`

A subclass of `NotImplementedError`, raised when `infer` does not support the given
function, such as a non-callable, an operation without a matching protocol, or a
parameter that requires a value no placeholder can provide.

## `InferWarning`

A subclass of `RuntimeWarning`, emitted when `infer` could not explore the function
exhaustively, such as when the branch or run budget runs out. The message categorizes
each coverage gap and names the affected call form, and the returned signature then only
covers the explored paths. The `optype infer` command prints the warning to standard
error, leaving the signature alone on standard output. Pass `strict=True` to `infer` to
raise an [`InferError`](#infererror) instead.
