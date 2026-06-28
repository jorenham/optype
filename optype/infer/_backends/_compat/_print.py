"""Emit valid Python text from the lowered model, collecting imports as it goes.

Printers thread a `used` set and record each name they reference, so the import block
needs no second traversal. `_type` doubles as the lowerer's canonical serializer; with
`used=None` it just returns text.
"""

import builtins
import types
import typing
from collections.abc import Sequence
from typing import cast

# `from optype.infer import _ir` would re-enter the package, which imports this module
import optype.infer._ir as _ir  # noqa: PLR0402
from ._model import _Alias, _Attr, _Func, _Member, _Method, _Protocol, _value
from optype._core import _can, _has, _just
from optype.infer._backends._base import default_text

__all__ = (
    "_OPTYPE",
    "_alias_text",
    "_func_text",
    "_import_block",
    "_import_of",
    "_member_key",
    "_protocol_text",
    "_type",
)

_ABC = frozenset({
    "Callable",
    "Generator",
    "AsyncGenerator",
    "Iterator",
    "Iterable",
    "Awaitable",
    "Coroutine",
    "Mapping",
    "Sequence",
})
_TYPING = frozenset({"Any", "ClassVar", "Literal", "Never", "Protocol", "overload"})
_TYPING_EXT = frozenset({"TypeForm", "deprecated"})
_OPTYPE = frozenset(_can.__all__) | frozenset(_has.__all__) | frozenset(_just.__all__)

_numpy_names: frozenset[str] | None = None


def _optype_numpy() -> frozenset[str]:
    global _numpy_names  # noqa: PLW0603
    if _numpy_names is None:
        try:
            import optype.numpy as onp  # noqa: PLC0415

            _numpy_names = frozenset(cast("Sequence[str]", onp.__all__))
        except Exception:  # noqa: BLE001 - numpy is optional and may be unimportable
            _numpy_names = frozenset()
    return _numpy_names


def _import_of(name: str) -> tuple[str, str | None] | None:  # noqa: PLR0911
    """The `(module, member)` an importable `name` comes from, or `None` for none.

    A `None` member means the module is imported whole, as in `import numpy as np`.
    """
    if "." in name:
        module = "numpy" if name.partition(".")[0] == "np" else name.rpartition(".")[0]
        return module, None
    if hasattr(builtins, name):
        return None
    if name in _ABC:
        return "collections.abc", name
    if name in _TYPING:
        return "typing", name
    if name in _TYPING_EXT:
        return "typing_extensions", name
    if name in _OPTYPE:
        return "optype", name
    if name in _optype_numpy():
        return "optype.numpy", name
    if hasattr(types, name):
        return "types", name
    if hasattr(typing, name):
        return "typing", name
    return None


def _noop(_name: str) -> None: ...


def _type(node: _ir.Node, used: set[str] | None = None) -> str:  # noqa: C901, PLR0911
    # `rec` records each referenced name when `used` is given
    rec = _noop if used is None else used.add
    match node:
        case _ir.Lit(values):
            rec("Literal")
            return f"Literal[{', '.join(map(repr, values))}]"
        case _ir.Type(cls):
            rec(name := _ir.type_name(cls))
            return name
        case _ir.Name(name):
            rec(name)
            return name
        case _ir.Dots():
            return "..."
        case _ir.App("tuple", ()):
            rec("tuple")
            return "tuple[()]"
        case _ir.App(base, ()):
            rec(base)
            return base
        case _ir.App(base, args):
            rec(base)
            return f"{base}[{', '.join(_type(_value(a), used) for a in args)}]"
        case _ir.Fn(params, ret):
            rec("Callable")
            inner = (
                "..."
                if params == (_ir.Dots(),)
                else f"[{', '.join(_type(_value(p), used) for p in params)}]"
            )
            return f"Callable[{inner}, {_type(ret, used)}]"
        case _ir.Union(parts):
            return " | ".join(_type(part, used) for part in parts)
        case _ir.Unpack(part):
            return f"*{_type(part, used)}"
        case _:  # an Inter/Not/Variance survived lowering, which is a bug
            msg = f"cannot render {node!r} as valid Python"
            raise AssertionError(msg)


def _type_params(tps: Sequence[_ir.TypeParam], used: set[str] | None = None) -> str:
    return f"[{', '.join(_type_param(tp, used) for tp in tps)}]" if tps else ""


def _type_param(tp: _ir.TypeParam, used: set[str] | None = None) -> str:
    if tp.unpack:
        return f"*{tp.name}"
    decl = f"{tp.name}: {_type(tp.bound, used)}" if tp.bound is not None else tp.name
    return f"{decl} = {_type(tp.default, used)}" if tp.default is not None else decl


def _join_params(items: Sequence[tuple[str, bool]]) -> str:
    """Join rendered params, inserting `/` after a leading positional-only run."""
    parts: list[str] = []
    slash = 0
    for decl, positional_only in items:
        if positional_only:
            slash = len(parts) + 1
        parts.append(decl)
    if slash:
        parts.insert(slash, "/")
    return ", ".join(parts)


def _params(params: Sequence[_ir.Param], used: set[str]) -> str:
    show = _default_mask(params)
    auto = 0
    items: list[tuple[str, bool]] = []
    for i, p in enumerate(params):
        if p.nameless:
            decl = f"_{auto}: {_type(p.node, used)}"
            auto += 1
        else:
            decl = f"{p.prefix}{p.name}: {_type(p.node, used)}"
        if p.default is not None and show[i]:
            decl += f" = {default_text(p.default[0])}"
        items.append((decl, p.nameless))
    return _join_params(items)


def _default_mask(params: Sequence[_ir.Param]) -> list[bool]:
    """Which params may show their default: only a suffix of the positional run may.

    A non-default positional parameter forces every earlier one to drop its default,
    since `def f(x=1, y)` is a syntax error; the type already pins the value anyway.
    """
    end = next((i for i, p in enumerate(params) if p.prefix), len(params))
    show = [p.default is not None for p in params]
    required = False
    for i in range(end - 1, -1, -1):
        if params[i].default is None:
            required = True
        elif required:
            show[i] = False
    return show


def _call_params(
    params: Sequence[_ir.Node | _ir.Arg],
    used: set[str] | None = None,
) -> str:
    auto = 0
    items: list[tuple[str, bool]] = []
    for p in params:
        if isinstance(p, _ir.Arg) and p.key:
            decl = f"{p.key}: {_type(p.value, used)}"
            if p.default is not None:
                decl += f" = {default_text(p.default[0])}"
            items.append((decl, False))
        else:
            items.append((f"_{auto}: {_type(_value(p), used)}", True))
            auto += 1
    return _join_params(items)


def _member_text(member: _Member, *, overload: bool, used: set[str]) -> str:
    if isinstance(member, _Attr):
        if member.classvar:
            used.add("ClassVar")
            return f"    {member.name}: ClassVar[{_type(member.type, used)}]"
        if member.readonly:
            ret = _type(member.type, used)
            return f"    @property\n    def {member.name}(self) -> {ret}: ..."
        return f"    {member.name}: {_type(member.type, used)}"
    sig = _call_params(member.params, used)
    head = ""
    if overload:
        used.add("overload")
        head = "    @overload\n"
    self_sig = f"self, {sig}" if sig else "self"
    return f"{head}    def {member.name}({self_sig}) -> {_type(member.ret, used)}: ..."


def _protocol_text(proto: _Protocol, used: set[str]) -> str:
    used.add("Protocol")
    bases = ", ".join([*(_type(b, used) for b in proto.bases), "Protocol"])
    head = f"class {proto.name}{_type_params(proto.type_params, used)}({bases}):"
    if not proto.members:
        return f"{head} ..."
    counts = {m.name: 0 for m in proto.members}
    for m in proto.members:
        counts[m.name] += 1
    body = "\n".join(
        _member_text(
            m,
            overload=isinstance(m, _Method) and counts[m.name] > 1,
            used=used,
        )
        for m in proto.members
    )
    return f"{head}\n{body}"


def _alias_text(alias: _Alias, used: set[str]) -> str:
    value = _type(alias.value, used)
    return f"type {alias.name}{_type_params(alias.type_params, used)} = {value}"


def _func_text(func: _Func, used: set[str]) -> str:
    head = ""
    if func.deprecated is not None:
        used.add("deprecated")
        head = f"@deprecated({func.deprecated!r})\n"
    sig = f"def f{_type_params(func.type_params, used)}({_params(func.params, used)})"
    return f"{head}{sig} -> {_type(func.ret, used)}: ..."


def _member_key(member: _Member) -> str:
    if isinstance(member, _Attr):
        flags = f"{member.classvar}{member.readonly}"
        return f"{member.name}:{flags}:{_type(member.type)}"
    return f"{member.name}:{_call_params(member.params)}->{_type(member.ret)}"


def _import_block(used: set[str], locals_: set[str], typevars: set[str]) -> str:
    """The import lines for every referenced name that is neither helper nor typevar."""
    groups: dict[str, set[str]] = {}
    whole: set[str] = set()
    for name in used - locals_ - typevars:
        if (found := _import_of(name)) is None:
            continue
        module_name, member = found
        if member is None:
            whole.add(module_name)
        else:
            groups.setdefault(module_name, set()).add(member)

    lines = [
        f"import {name} as np" if name == "numpy" else f"import {name}"
        for name in sorted(whole)
    ]
    order = {"collections.abc": 0, "types": 1, "typing": 2, "typing_extensions": 3}
    for module_name in sorted(groups, key=lambda m: (order.get(m, 9), m)):
        members = ", ".join(sorted(groups[module_name]))
        lines.append(f"from {module_name} import {members}")
    return "\n".join(lines)
