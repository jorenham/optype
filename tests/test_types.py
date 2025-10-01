import sys

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

import optype.types as opts


@deprecated("Use f_new instead")
def f_old() -> None: ...
def f_new() -> None: ...


@deprecated("Use New instead")
class Old: ...


class New: ...


def test_deprecated_callable() -> None:
    # pyrefly: ignore[deprecated]
    assert isinstance(f_old, opts.Deprecated)  # pyright: ignore[reportDeprecated]
    assert not isinstance(f_new, opts.Deprecated)


def test_deprecated_type() -> None:
    assert isinstance(Old, opts.Deprecated)  # pyright: ignore[reportDeprecated]
    assert not isinstance(New, opts.Deprecated)
