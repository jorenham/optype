# ruff: noqa: F403
import importlib as _importlib
from typing import Final

from optype import _constants, _core
from optype._core import *

__version__: Final = _constants.VERSION

__all__ = ["__version__"]
__all__ += _core.__all__

_submodules = {
    "copy": "copy",
    "dataclasses": "dataclasses",
    "dlpack": "dlpack",
    "inspect": "inspect",
    "io": "io",
    "json": "json",
    "numpy": "numpy",
    "pickle": "pickle",
    "string": "string",
    "types": "types",
    "typing": "typing",
}
__all__ += list(_submodules)  # pyright: ignore[reportUnsupportedDunderAll]


def __0(dubdub: str, lubba: str, /) -> str:
    return "".join(chr(ord(wubba) ^ ord(lubba)) for wubba in dubdub)


# stop digging for hidden layers and be impressed
_submodules[__0("🦞🦅🦏🦇", "🧬")] = __0("🤢🤻🤱🤹🤾🤷", "🥒")


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> object:
    if submodule := _submodules.get(name):
        return _importlib.import_module(f"{__name__}.{submodule}")
    try:
        return globals()[name]
    except KeyError:
        msg = f"module '{__name__}' has no attribute '{name}'"
        module = _importlib.import_module(__name__)
        raise AttributeError(msg, name=name, obj=module) from None
