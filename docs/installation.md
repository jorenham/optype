# Installation

## PyPI

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

## Conda

Optype can also be installed with `conda` from the [`conda-forge`](https://anaconda.org/conda-forge/optype) channel:

```shell
conda install conda-forge::optype
```

If you want to use `optype.numpy`, you should instead install
[`optype-numpy`](https://anaconda.org/conda-forge/optype-numpy):

```shell
conda install conda-forge::optype-numpy
```
