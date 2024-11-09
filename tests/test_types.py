from typing_extensions import deprecated

import optype.types as opt


@deprecated("Use f_new instead")
def f_old() -> None: ...
def f_new() -> None: ...


@deprecated("Use New instead")
class Old: ...


class New: ...


def test_deprecated_callable() -> None:
    assert isinstance(f_old, opt.Deprecated)  # pyright: ignore[reportDeprecated]
    assert not isinstance(f_new, opt.Deprecated)


def test_deprecated_type() -> None:
    assert isinstance(Old, opt.Deprecated)  # pyright: ignore[reportDeprecated]
    assert not isinstance(New, opt.Deprecated)
