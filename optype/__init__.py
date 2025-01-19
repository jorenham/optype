# ruff: noqa: F403
import importlib as _importlib
import importlib.metadata as _metadata

from ._core import _can, _do, _does, _has, _just
from ._core._can import *
from ._core._do import *
from ._core._does import *
from ._core._has import *
from ._core._just import *

__version__: str = _metadata.version(__package__ or __file__.split("/")[-1])
__all__ = ["__version__"]
__all__ += _just.__all__
__all__ += _can.__all__
__all__ += _has.__all__
__all__ += _does.__all__
__all__ += _do.__all__


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
        return globals()[name]  # pyright: ignore[reportAny]
    except KeyError:
        msg = f"module '{__name__}' has no attribute '{name}'"
        module = _importlib.import_module(__name__)
        raise AttributeError(msg, name=name, obj=module) from None
