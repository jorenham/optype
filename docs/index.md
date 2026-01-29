# optype

**Building blocks for precise & flexible type hints**

[![GitHub License](https://img.shields.io/github/license/jorenham/optype?style=flat-square&color=121d2f&labelColor=3d444d)](https://github.com/jorenham/optype)
[![PyPI Version](https://img.shields.io/pypi/v/optype?style=flat-square&color=121d2f&labelColor=3d444d)](https://pypi.org/project/optype)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/optype?style=flat-square&color=121d2f&labelColor=3d444d)](https://anaconda.org/conda-forge/optype)
[![Python Versions](https://img.shields.io/pypi/pyversions/optype?style=flat-square&color=121d2f&labelColor=3d444d)](https://github.com/jorenham/optype)
[![PyPI Downloads](https://img.shields.io/pypi/dm/optype?style=flat-square&color=121d2f&labelColor=3d444d&cacheSeconds=86400)](https://pypi.org/project/optype)

[![numpy](https://img.shields.io/badge/numpy-262c36?style=flat-square&logo=numpy)](https://github.com/numpy/numpy)
[![basedpyright](https://img.shields.io/badge/basedpyright-262c36?style=flat-square&logoColor=fdc204)](https://detachhead.github.io/basedpyright)
[![mypy](https://img.shields.io/badge/mypy-262c36?style=flat-square&logo=python)](https://github.com/python/mypy)
[![pyrefly](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/facebook/pyrefly/refs/heads/main/website/static/badge.json&style=flat-square&color=262c36&labelColor=262c36)](https://github.com/facebook/pyrefly)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square&color=262c36&labelColor=262c36)](https://github.com/astral-sh/ruff)

---

## What is optype?

`optype` is a Python library that provides building blocks for precise and flexible type hints. It focuses on typing utilities and has optional elaborate NumPy support. The library is compatible with all modern static type-checkers (mypy, basedpyright, pyright, pyrefly, etc.) and also works with runtime type-checkers like beartype.

## Key Features

- **Comprehensive Protocol Collection**: Single-method protocols for all builtin special methods (`Can*` protocols)
- **Precise Type Control**: `Just[T]` types that reject strict subtypes (e.g., reject `bool` when you want `int`)
- **Operator Types**: Complete typing for operators (`Does*` protocols and `do_*` implementations)
- **Attribute Protocols**: Type-safe attribute access (`Has*` protocols)
- **Standard Library Support**: Type protocols for `copy`, `pickle`, `json`, `io`, `inspect`, and more
- **NumPy Integration**: Extensive shape-typing, array protocols, and dtype utilities
- **Runtime Checkable**: All protocols support `isinstance()` checks
- **No Dependencies**: Core library has zero dependencies (NumPy support is optional)

## Installation

### PyPI

Optype is available as [`optype`](https://pypi.org/project/optype/) on PyPI:

```shell
pip install optype
```

For optional [NumPy](https://github.com/numpy/numpy) support, use the `optype[numpy]` extra.
This ensures that the installed `numpy` and the required [`numpy-typing-compat`](https://github.com/jorenham/numpy-typing-compat)
versions are compatible with each other:

```shell
pip install "optype[numpy]"
```

See the [NumPy reference](reference/numpy/index.md) for more info.

### Conda

Optype can also be installed with `conda` from the [`conda-forge`](https://anaconda.org/conda-forge/optype) channel:

```shell
conda install conda-forge::optype
```

If you want to use `optype.numpy`, you should instead install
[`optype-numpy`](https://anaconda.org/conda-forge/optype-numpy):

```shell
conda install conda-forge::optype-numpy
```

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
    from optype import CanRMul

    type Two = Literal[2]
    type RMul2[R] = CanRMul[Two, R]


    def twice[R](x: RMul2[R]) -> R:
        return 2 * x
    ```

=== "Python 3.11"

    ```python
    from typing import Literal, TypeAlias, TypeVar
    from optype import CanRMul

    R = TypeVar("R")
    Two: TypeAlias = Literal[2]
    RMul2: TypeAlias = CanRMul[Two, R]


    def twice(x: RMul2[R]) -> R:
        return 2 * x
    ```

See the [Getting Started](getting-started.md) guide for more detailed examples.

## Next Steps

- [Getting Started](getting-started.md): Learn how to use optype with complete examples
- [Reference](reference/index.md): Comprehensive API documentation
- [GitHub Repository](https://github.com/jorenham/optype): Source code and issue tracker
