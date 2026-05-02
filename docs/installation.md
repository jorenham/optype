# Installation

## PyPI

Optype is available as [`optype`][PYPI] on PyPI:

```shell
pip install optype
```

For optional [NumPy][NUMPY] support, ensure that you use the `optype[numpy]` extra.
This ensures that the installed `numpy` and the required [`numpy-typing-compat`][NPTC]
versions are compatible with each other.

```shell
pip install "optype[numpy]"
```

See the [`optype.numpy` docs](reference/numpy/index.md) for more info.

## Conda

Optype can also be installed with `conda` from the [`conda-forge`][CONDA] channel:

```shell
conda install conda-forge::optype
```

If you want to use [`optype.numpy`](reference/numpy/index.md), you should instead install
[`optype-numpy`][CONDA-NP]:

```shell
conda install conda-forge::optype-numpy
```

[PYPI]: https://pypi.org/project/optype/
[CONDA]: https://anaconda.org/conda-forge/optype
[CONDA-NP]: https://anaconda.org/conda-forge/optype-numpy
[NUMPY]: https://github.com/numpy/numpy
[NPTC]: https://github.com/jorenham/numpy-typing-compat
