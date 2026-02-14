<h1 align="center">optype</h1>

<p align="center">Building blocks for precise & flexible type hints</p>

<p align="center">
<a href="https://github.com/jorenham/optype"><img alt="GitHub License" src="https://img.shields.io/github/license/jorenham/optype?style=flat-square&color=121d2f&labelColor=3d444d"></a>
<a href="https://pypi.org/project/optype"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/optype?style=flat-square&color=121d2f&labelColor=3d444d"></a>
<a href="https://anaconda.org/conda-forge/optype"><img alt="Conda Version" src="https://img.shields.io/conda/vn/conda-forge/optype?style=flat-square&color=121d2f&labelColor=3d444d"></a>
<a href="https://github.com/jorenham/optype"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/optype?style=flat-square&color=121d2f&labelColor=3d444d"></a>
<a href="https://pypi.org/project/optype"><img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/optype?style=flat-square&color=121d2f&labelColor=3d444d&cacheSeconds=86400"></a>
</p>

<p align="center">
<a href="https://github.com/numpy/numpy"><img alt="numpy" src="https://img.shields.io/badge/numpy-262c36?style=flat-square&logo=numpy"></a>
<a href="https://detachhead.github.io/basedpyright"><img alt="basedpyright" src="https://img.shields.io/endpoint?url=https://docs.basedpyright.com/latest/badge.json&style=flat-square&color=262c36&labelColor=262c36"></a>
<a href="https://github.com/python/mypy"><img alt="mypy" src="https://img.shields.io/badge/mypy-262c36?style=flat-square&logo=python"></a>
<a href="https://github.com/facebook/pyrefly"><img alt="pyrefly" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/facebook/pyrefly/refs/heads/main/website/static/badge.json&style=flat-square&color=262c36&labelColor=262c36"></a>
<a href="https://github.com/astral-sh/ty"><img alt="ty" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json&style=flat-square&color=262c36&labelColor=262c36"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square&color=262c36&labelColor=262c36"></a>
</p>

---

## What is optype?

`optype` is a Python library that provides building blocks for precise and flexible type
hints. It focuses on typing utilities and has optional elaborate NumPy support.
The library is compatible with all modern static type-checkers (mypy, basedpyright,
pyright, pyrefly, ty) and also works with runtime type-checkers like beartype.

## Key Features

- **Comprehensive Protocol Collection**: Single-method protocols for all builtin special methods (`Can*` protocols)
- **Precise Type Control**: `Just[T]` types that reject strict subtypes (e.g., reject `bool` when you want `int`)
- **Standard Library Support**: Type protocols for `copy`, `pickle`, `json`, `io`, `inspect`, and more
- **NumPy Integration**: Extensive shape-typing, array protocols, and dtype utilities
- **No Dependencies**: Core library has zero dependencies (NumPy support is optional)

## Quick Example

Let's say you're writing a `twice(x)` function that evaluates `2 * x`.
Implementing it is trivial, but what about the type annotations?

Because `twice(2) == 4`, `twice(3.14) == 6.28`, and `twice('I') == 'II'`, it
might seem like a good idea to type it as `twice[T](x: T) -> T: ...`.
However, that wouldn't include cases such as `twice(True) == 2` or
`twice((42, True)) == (42, True, 42, True)`, where the input and output types
differ.

Moreover, `twice` should accept *any* type with a custom `__rmul__` method
that accepts `2` as an argument.

This is where `optype` comes in handy. Use `optype.CanRMul[T, R]`, which is a
protocol with (only) the `__rmul__(self, lhs: T) -> R` method:

=== "Python 3.12+"

    ```python
    from typing import Literal
    import optype as op

    type Two = Literal[2]
    type RMul2[R] = op.CanRMul[Two, R]


    def twice[R](x: RMul2[R]) -> R:
        return 2 * x
    ```

=== "Python 3.11"

    ```python
    from typing import Literal, TypeAlias, TypeVar
    import optype as op

    R = TypeVar("R")
    Two: TypeAlias = Literal[2]
    RMul2: TypeAlias = op.CanRMul[Two, R]


    def twice(x: RMul2[R]) -> R:
        return 2 * x
    ```

See the [Getting Started](getting-started.md) guide for more detailed examples.

## Next Steps

- [Installation](installation.md): How to install optype with pip or conda
- [Getting Started](getting-started.md): Learn how to use optype with complete examples
- [Reference](reference/index.md): Comprehensive API documentation
- [GitHub Repository](https://github.com/jorenham/optype): Source code and issue tracker
