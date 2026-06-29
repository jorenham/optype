"""The catalog mapping recorded dunder traces to the `optype` protocols."""

from typing import NamedTuple

# `from . import` would import the package itself, which imports this module
import optype.infer._numpy as _numpy
from ._errors import InferError
from ._spy import _Args, _Kwargs, _Marker, _Spy, _TraceItem
from optype._core import _can, _has
from optype.inspect import get_protocol_members

_DUNDER_ATTR_WRITE = frozenset({"__delattr__", "__setattr__"})
DUNDER_ATTR = frozenset({"__getattr__", "__getattribute__"}) | _DUNDER_ATTR_WRITE
DUNDER_CLASS_ATTR = frozenset({
    _Marker.CLASS_DELATTR,
    _Marker.CLASS_GETATTR,
    _Marker.CLASS_SETATTR,
})


def _get_dunder_can_map() -> dict[str, str]:
    return {
        dunder: name
        for name in _can.__all__
        if not name.endswith(("Self", "Same"))
        if len(members := get_protocol_members(getattr(_can, name))) == 1
        if (dunder := next(iter(members))) not in DUNDER_ATTR
        # CanPow2, CanRound1, ... share their dunder; keep the canonical protocol
        if dunder.replace("_", "") == name.removeprefix("Can").lower()
    } | _numpy.DUNDER_CAN_MAP


_DUNDER_CAN_MAP = _get_dunder_can_map()
DUNDER_CAN_R = frozenset(
    dunder
    for dunder, proto in _DUNDER_CAN_MAP.items()
    if "CanR" + proto.removeprefix("Can") in _DUNDER_CAN_MAP.values()
)


def _get_dunder_has_map() -> dict[str, str]:
    return {
        next(iter(members)): name
        for name in _has.__all__
        if len(members := get_protocol_members(getattr(_has, name))) == 1
    }


_DUNDER_HAS_MAP = _get_dunder_has_map()

_COERCION_FALLBACK = {
    "__float__": ("__index__",),
    "__int__": ("__index__",),
    "__complex__": ("__float__", "__index__"),
}
_COERCION_PROTOS = {
    dunder: tuple(map(_DUNDER_CAN_MAP.__getitem__, (dunder, *fallback)))
    for dunder, fallback in _COERCION_FALLBACK.items()
}

type Proto = str | tuple[str, ...]  # a tuple is rendered as a union of protocols


class Op(NamedTuple):
    proto: Proto
    args: _Args
    kwargs: _Kwargs
    ret: object
    attr: str | None = None  # the subject of a synthesized `Has[...]` form
    classvar: bool = False  # a class-level attribute, i.e. a `ClassVar` member


def resolve(trace: _TraceItem) -> Op:
    if trace.attr in DUNDER_ATTR or trace.attr in DUNDER_CLASS_ATTR:
        name = trace.args[0]
        if not isinstance(name, str) or isinstance(name, _Spy):
            msg = "no protocol for a dynamic attribute name"
            raise InferError(msg)

        # a class-level attribute mirrors a `ClassVar` protocol member, which no
        # shipped instance-member `Has*` protocol declares
        if trace.attr in DUNDER_CLASS_ATTR:
            return Op("Has", trace.args[1:], {}, trace.return_, name, classvar=True)

        # a read of an attribute with a shipped single-member `Has*` protocol
        if name in _DUNDER_HAS_MAP and trace.attr not in _DUNDER_ATTR_WRITE:
            return Op(_DUNDER_HAS_MAP[name], (), {}, trace.return_)

        # everything else synthesizes the inline `Has['name', T]` form; a write
        # binds the assigned value's type, which a bounded `Has*` could reject
        return Op("Has", trace.args[1:], {}, trace.return_, name)

    # checked before _DUNDER_CAN_MAP, which also contains the coercion dunders
    if trace.attr in _COERCION_PROTOS:
        return Op(_COERCION_PROTOS[trace.attr], (), {}, trace.return_)

    if trace.attr in _DUNDER_CAN_MAP:
        proto = _DUNDER_CAN_MAP[trace.attr]
        # a modulo arg makes `__rpow__` the 3-parameter `CanRPow3`, not `CanRPow`
        if trace.attr == "__rpow__" and len(trace.args) > 1:
            proto = "CanRPow3"
        return Op(proto, trace.args, trace.kwargs, trace.return_)

    msg = f"no protocol for {trace.attr!r}"
    raise InferError(msg)
