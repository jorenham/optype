# optype.numpy

Optional NumPy support for `optype`.

Optype supports both NumPy 1 and 2. The current minimum supported version is `1.26`,
following [NEP 29][NEP29] and [SPEC 0][SPEC0].

`optype.numpy` uses [`numpy-typing-compat`][NPTC] package to ensure compatibility for
older versions of NumPy. To ensure that the correct versions of `numpy` and
`numpy-typing-compat` are installed, you should install `optype` with the `numpy` extra:

```shell
pip install "optype[numpy]"
```

If you're using `conda`, the [`optype-numpy`][CONDA-NP] package can be used, which
will also install the required `numpy` and `numpy-typing-compat` versions:

```shell
conda install conda-forge::optype-numpy
```

!!! note
For the remainder of the `optype.numpy` docs, assume that the following
import aliases are available.

    ```python
    from typing import Any, Literal
    import numpy as np
    import numpy.typing as npt
    import optype.numpy as onp
    ```

    For the sake of brevity and readability, the [PEP 695][PEP695] and
    [PEP 696][PEP696] type parameter syntax will be used, which is supported
    since Python 3.13.

[CONDA-NP]: https://anaconda.org/conda-forge/optype-numpy
[NPTC]: https://github.com/jorenham/numpy-typing-compat
[PEP695]: https://peps.python.org/pep-0695/
[PEP696]: https://peps.python.org/pep-0696/
[NEP29]: https://numpy.org/neps/nep-0029-deprecation_policy.html
[SPEC0]: https://scientific-python.org/specs/spec-0000/
