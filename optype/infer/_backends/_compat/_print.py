"""Emit valid Python text from the lowered model, collecting imports as it goes.

A `_Printer` records every name it references, so the import block needs no second
traversal. The module-level `_type` and `_member_key` discard that record: the lowerer
uses them to build canonical dedup keys, not output.
"""

import builtins
import functools
import types
import typing
from collections.abc import Sequence, Set as AbstractSet
from typing import Literal, final

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]
from ._model import _Alias, _Attr, _Func, _Member, _Method, _Protocol
from optype._core import _can, _has, _just
from optype.infer._backends._base import qualified_default_text, qualified_value_text

__all__ = "_OPTYPE", "_Printer", "_import_of", "_member_key", "_type"

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

type _Prefix = Literal["", "*"]


@functools.cache
def _optype_numpy() -> frozenset[str]:
    try:
        import optype.numpy as onp  # ruff: ignore[import-outside-top-level]
    except ImportError:  # numpy is optional
        return frozenset()
    return frozenset(onp.__all__)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


def _import_of(name: str) -> tuple[str, str | None] | None:  # ruff: ignore[too-many-return-statements]
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


@final
class _Printer:
    """Render the lowered model as Python text, recording the names it references."""

    _used: set[str]  # every referenced name, for the import block

    def __init__(self) -> None:
        self._used = set()

    def record(self, name: str) -> None:
        self._used.add(name)

    def render_node(self, node: _ir.Node) -> str:  # ruff: ignore[complex-structure, too-many-return-statements]
        rec = self._used.add
        match node:
            case _ir.Lit(values):
                rec("Literal")
                joined = ", ".join(qualified_value_text(v, rec) for v in values)
                return f"Literal[{joined}]"
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
            case _ir.App(origin, ()):
                rec(origin)
                return origin
            case _ir.App(origin, args):
                rec(origin)
                return f"{origin}[{', '.join(self._arg_types(args))}]"
            case _ir.Fn(params, ret):
                rec("Callable")
                inner = (
                    "..."
                    if params == (_ir.Dots(),)
                    else f"[{', '.join(self._arg_types(params))}]"
                )
                return f"Callable[{inner}, {self.render_node(ret)}]"
            case _ir.Union(parts):
                return " | ".join(self.render_node(part) for part in parts)
            case _ir.Unpack(part):
                return f"*{self.render_node(part)}"
            case _:  # an Intersection/Not/Variance survived lowering, which is a bug
                msg = f"cannot render {node!r} as valid Python"
                raise AssertionError(msg)

    def _arg_types(self, args: Sequence[_ir.Term]) -> list[str]:
        return [self.render_node(_ir.term_node(a)) for a in args]

    def type_params(self, typars: Sequence[_ir.TypeParam]) -> str:
        if not typars:
            return ""
        return f"[{', '.join(self._type_param(typar) for typar in typars)}]"

    def _type_param(self, typar: _ir.TypeParam) -> str:
        if typar.unpack:
            return f"*{typar.name}"

        decl = (
            f"{typar.name}: {self.render_node(typar.bound)}"
            if typar.bound is not None
            else typar.name
        )
        if typar.default is None:
            return decl
        return f"{decl} = {self.render_node(typar.default)}"

    def params(self, params: Sequence[_ir.Param]) -> str:
        show = _default_mask(params)
        auto = 0
        items: list[tuple[str, bool]] = []
        for i, p in enumerate(params):
            if p.nameless:
                decl = f"_{auto}: {self.render_node(p.node)}"
                auto += 1
            else:
                decl = f"{p.prefix}{p.name}: {self.render_node(p.node)}"
            if p.default is not None and show[i]:
                decl += f" = {qualified_default_text(p.default[0], self._used.add)}"
            items.append((decl, p.nameless))
        return _join_params(items)

    def _prefixed_type(self, value: _ir.Node) -> tuple[_Prefix, str]:
        """A parameter's `(prefix, annotation)`; an unpack becomes a `*` parameter."""
        match value:
            case _ir.Unpack(_ir.App("tuple", (elem, _ir.Dots()))):
                prefix, value = "*", _ir.term_node(elem)
            case _ir.Unpack():
                prefix = "*"
            case _:
                prefix = ""
        return prefix, self.render_node(value)  # type:ignore[return-value]  # mypy fail

    def call_params(self, params: Sequence[_ir.Term]) -> str:
        auto = 0
        items: list[tuple[str, bool]] = []
        for p in params:
            if isinstance(p, _ir.Arg) and p.key:
                decl = f"{p.key}: {self.render_node(p.value)}"
                if p.default is not None:
                    default = qualified_default_text(p.default[0], self._used.add)
                    decl += f" = {default}"
                items.append((decl, False))
                continue

            # a `*` parameter is not positional-only, so the `/` lands before it
            prefix, ann = self._prefixed_type(_ir.term_node(p))
            items.append((f"{prefix}_{auto}: {ann}", not prefix))
            auto += 1

        return _join_params(items)

    def _member_text(self, member: _Member, *, overload: bool) -> str:
        if isinstance(member, _Attr):
            if member.classvar:
                self.record("ClassVar")
                return f"    {member.name}: ClassVar[{self.render_node(member.type)}]"
            if member.readonly:
                ret = self.render_node(member.type)
                return f"    @property\n    def {member.name}(self) -> {ret}: ..."
            return f"    {member.name}: {self.render_node(member.type)}"
        sig = self.call_params(member.params)
        head = ""
        if overload:
            self.record("overload")
            head = "    @overload\n"
        self_sig = f"self, {sig}" if sig else "self"
        ret = self.render_node(member.ret)
        return f"{head}    def {member.name}({self_sig}) -> {ret}: ..."

    def protocol_text(self, proto: _Protocol) -> str:
        self.record("Protocol")
        bases = ", ".join([*(self.render_node(b) for b in proto.bases), "Protocol"])
        head = f"class {proto.name}{self.type_params(proto.type_params)}({bases}):"
        if not proto.members:
            return f"{head} ..."
        counts = {m.name: 0 for m in proto.members}
        for m in proto.members:
            counts[m.name] += 1
        body = "\n".join(
            self._member_text(m, overload=isinstance(m, _Method) and counts[m.name] > 1)
            for m in proto.members
        )
        return f"{head}\n{body}"

    def alias_text(self, alias: _Alias) -> str:
        value = self.render_node(alias.value)
        return f"type {alias.name}{self.type_params(alias.type_params)} = {value}"

    def func_text(self, func: _Func) -> str:
        head = ""
        if func.deprecated is not None:
            self.record("deprecated")
            head = f"@deprecated({func.deprecated!r})\n"
        sig = f"def f{self.type_params(func.type_params)}({self.params(func.params)})"
        return f"{head}{sig} -> {self.render_node(func.ret)}: ..."

    def import_block(
        self,
        locals_: AbstractSet[str],
        typevars: AbstractSet[str],
    ) -> str:
        """The import lines for referenced names that are neither helper nor typevar."""
        groups: dict[str, set[str]] = {}
        whole: set[str] = set()
        for name in self._used - locals_ - typevars:
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


def _type(node: _ir.Node) -> str:
    """A node's canonical text, for dedup keys; the recorded names are discarded."""
    return _Printer().render_node(node)


def _member_key(member: _Member) -> str:
    if isinstance(member, _Attr):
        flags = f"{member.classvar}{member.readonly}"
        return f"{member.name}:{flags}:{_type(member.type)}"
    params = _Printer().call_params(member.params)
    return f"{member.name}:{params}->{_type(member.ret)}"
