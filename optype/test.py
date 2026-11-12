import warnings
from collections.abc import Callable
from typing import Any, Self, final

try:
    from beartype import BeartypeConf
    from beartype.door import die_if_unbearable
except ImportError:
    die_if_unbearable = None  # type: ignore[assignment]
else:
    _CONF = BeartypeConf(is_pep484_tower=True, violation_door_type=AssertionError)


__all__ = ("UncheckedWarning", "assert_subtype", "assert_type")


class UncheckedWarning(UserWarning):
    """Emitted once when `optype.test` is used without beartype installed."""


_warned = False


def _check(val: object, typ: Any, name: str, /) -> None:
    global _warned  # ruff: ignore[global-statement]

    __tracebackhide__ = True

    if die_if_unbearable is None:
        if not _warned:
            warnings.warn(
                f"{name}() is not checked at runtime because beartype is not installed",
                UncheckedWarning,
                stacklevel=3,
            )
            _warned = True
        return

    try:
        die_if_unbearable(val, typ, conf=_CONF, exception_prefix=f"{name}() ")
    except AssertionError as e:
        raise e.with_traceback(None) from None


def assert_type[T](val: T, typ: Any, /) -> T:
    __tracebackhide__ = True

    _check(val, typ, "assert_type")
    return val


@final
class assert_subtype:  # ruff: ignore[invalid-class-name]
    def __class_getitem__(cls, typ: Any, /) -> Callable[[object], Self]:
        def assert_subtype(val: object, /) -> Self:
            __tracebackhide__ = True

            _check(val, typ, "assert_subtype")
            return cls()

        return assert_subtype
