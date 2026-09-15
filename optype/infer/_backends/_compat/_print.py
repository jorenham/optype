"""Emit valid Python text from the lowered model, collecting imports as it goes.

A `Printer` records every name it references, so the import block needs no second
traversal. The module-level `type_text` and `member_key` discard that record: the
lowerer uses them to build canonical dedup keys, not output.
"""

import builtins
import functools
import types
import typing
from collections.abc import Sequence, Set as AbstractSet
from typing import Literal, final

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]
from ._model import Alias, Attr, Member, Method, ProtocolDef
from optype._core import _can, _has, _just
from optype.infer._backends._base import default_text, value_text

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
OPTYPE = frozenset(_can.__all__) | frozenset(_has.__all__) | frozenset(_just.__all__)

type _Prefix = Literal["", "*"]


@functools.cache
def _optype_numpy() -> frozenset[str]:
    try:
        import optype.numpy as onp  # ruff: ignore[import-outside-top-level]
    except ImportError:  # numpy is optional
        return frozenset()
    return frozenset(onp.__all__)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


def import_of(name: str) -> tuple[str, str | None] | None:  # ruff: ignore[too-many-return-statements]
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
    if name in OPTYPE:
        return "optype", name
    if name in _optype_numpy():
        return "optype.numpy", name
    if hasattr(types, name):
        return "types", name
    if hasattr(typing, name):
        return "typing", name
    return None


def _qualified(name: str) -> str:
    """`name` behind its module, or as is when it has none."""
    if hasattr(builtins, name):
        return f"builtins.{name}"
    found = import_of(name)
    return name if found is None or found[1] is None else f"{found[0]}.{name}"


def _join_params(items: Sequence[tuple[str, bool]]) -> str:
    """Join rendered params, inserting `/` after a leading positional-only run."""
    parts: list[str] = []
    slash = 0
    for decl, pos_only in items:
        if pos_only:
            slash = len(parts) + 1
        parts.append(decl)
    if slash:
        parts.insert(slash, "/")
    return ", ".join(parts)


def _default_mask(defaults: Sequence[bool], end: int) -> list[bool]:
    """Which params may show their default: only a suffix of the positional run may.

    A non-default positional parameter forces every earlier one to drop its default,
    since `def f(x=1, y)` is a syntax error; the type already pins the value anyway.
    `end` is where the positional run stops.
    """
    show = list(defaults)
    required = False
    for i in range(end - 1, -1, -1):
        if not defaults[i]:
            required = True
        elif required:
            show[i] = False
    return show


@final
class Printer:
    """Render the lowered model as Python text, recording the names it references."""

    used: set[str]  # every referenced name, for the import block
    _shadowed: frozenset[str]  # names the current class body binds
    _aliases: dict[str, str]  # modules a class body shadows, by their import alias

    def __init__(self) -> None:
        self.used = set()
        self._shadowed = frozenset()
        self._aliases = {}

    def _literal(self, value: object) -> str:
        paths: list[str] = []
        text = value_text(value, paths.append)
        for path in paths:  # an enum member's `module.Class`
            module = path.rpartition(".")[0]
            if module.partition(".")[0] not in self._shadowed:
                self.used.add(path)
                continue
            if (alias := self._aliases.get(module)) is None:
                # a leading double underscore would be mangled in the class body
                alias = "_" + module.replace(".", "_").lstrip("_")
                while alias in self._aliases.values():
                    alias += "_"
                self._aliases[module] = alias
            text = alias + text.removeprefix(module)
        return text

    def _name(self, name: str) -> str:
        if name in self._shadowed:
            name = _qualified(name)
        self.used.add(name)
        return name

    def render_node(self, node: _ir.Node) -> str:  # ruff: ignore[complex-structure, too-many-return-statements]
        match node:
            case _ir.Lit(values):
                joined = ", ".join(map(self._literal, values))
                return f"{self._name('Literal')}[{joined}]"
            case _ir.Type(cls):
                return self._name(_ir.type_name(cls))
            case _ir.Name(name):
                return self._name(name)
            case _ir.Dots():
                return "..."
            case _ir.App("tuple", ()):
                return f"{self._name('tuple')}[()]"
            case _ir.App(origin, ()):
                return self._name(origin)
            case _ir.App(origin, args):
                return f"{self._name(origin)}[{', '.join(self._arg_types(args))}]"
            case _ir.Fn(params, ret):
                inner = (
                    "..."
                    if params == (_ir.Dots(),)
                    else f"[{', '.join(self._arg_types(params))}]"
                )
                return f"{self._name('Callable')}[{inner}, {self.render_node(ret)}]"
            case _ir.Union(parts):
                return " | ".join(self.render_node(part) for part in parts)
            case _ir.Unpack(part):
                return f"*{self.render_node(part)}"
            case _:  # an unlowered Has, Intersection, Not, or Has marker: a bug
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
            decl = f"*{typar.name}"
        elif typar.bound is not None:
            decl = f"{typar.name}: {self.render_node(typar.bound)}"
        else:
            decl = typar.name
        if typar.default is None:
            return decl
        return f"{decl} = {self.render_node(typar.default)}"

    def params(self, params: Sequence[_ir.Param]) -> str:
        end = next((i for i, p in enumerate(params) if p.prefix), len(params))
        show = _default_mask([p.default is not None for p in params], end)
        auto = 0
        items: list[tuple[str, bool]] = []
        for i, p in enumerate(params):
            if p.pos_only:
                decl = f"_{auto}: {self.render_node(p.node)}"
                auto += 1
            else:
                decl = f"{p.prefix}{p.name}: {self.render_node(p.node)}"
            if p.default is not None and show[i]:
                decl += f" = {default_text(p.default[0], self.used.add)}"
            items.append((decl, p.pos_only))
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
        end = next(
            (
                i
                for i, p in enumerate(params)
                if isinstance(_ir.term_node(p), _ir.Unpack)
            ),
            len(params),
        )
        defaults = [isinstance(p, _ir.Arg) and p.default is not None for p in params]
        show = _default_mask(defaults, end)
        auto = 0
        items: list[tuple[str, bool]] = []
        for i, p in enumerate(params):
            if isinstance(p, _ir.Arg) and p.key:
                decl = f"{p.key}: {self.render_node(p.value)}"
                pos_only = False
            else:
                # a `*` parameter is not positional-only, so the `/` lands before it
                prefix, ann = self._prefixed_type(_ir.term_node(p))
                decl = f"{prefix}_{auto}: {ann}"
                pos_only = not prefix
                auto += 1
            if isinstance(p, _ir.Arg) and p.default is not None and show[i]:
                decl += f" = {default_text(p.default[0], self.used.add)}"
            items.append((decl, pos_only))
        return _join_params(items)

    def _member_text(self, member: Member, *, overload: bool) -> str:
        if isinstance(member, Attr):
            if member.classvar:
                self.used.add("ClassVar")
                return f"    {member.name}: ClassVar[{self.render_node(member.type)}]"
            if member.readonly or member.setter is not None:
                ret = self.render_node(member.type)
                text = f"    @property\n    def {member.name}(self) -> {ret}: ..."
                if member.setter is None:
                    return text
                # the getter binds the name before the setter's annotation is read
                self._shadowed = frozenset({member.name})
                value = self.render_node(member.setter)
                self._shadowed = frozenset()
                return (
                    f"{text}\n    @{member.name}.setter\n"
                    f"    def {member.name}(self, value: {value}, /) -> None: ..."
                )
            return f"    {member.name}: {self.render_node(member.type)}"
        sig = self.call_params(member.params)
        head = ""
        if overload:
            self.used.add("overload")
            head = "    @overload\n"
        self_sig = f"self, {sig}" if sig else "self"
        ret = self.render_node(member.ret)
        return f"{head}    def {member.name}({self_sig}) -> {ret}: ..."

    def protocol_text(self, proto: ProtocolDef) -> str:
        self.used.add("Protocol")
        bases = ", ".join([*(self.render_node(b) for b in proto.bases), "Protocol"])
        head = f"class {proto.name}{self.type_params(proto.type_params)}({bases}):"
        if not proto.members:
            return f"{head} ..."
        counts = {m.name: 0 for m in proto.members}
        for m in proto.members:
            counts[m.name] += 1
        body = "\n".join(
            self._member_text(m, overload=isinstance(m, Method) and counts[m.name] > 1)
            for m in proto.members
        )
        return f"{head}\n{body}"

    def alias_text(self, alias: Alias) -> str:
        value = self.render_node(alias.value)
        return f"type {alias.name}{self.type_params(alias.type_params)} = {value}"

    def func_text(self, func: _ir.Signature) -> str:
        head = ""
        if func.deprecated is not None:
            self.used.add("deprecated")
            head = f"@deprecated({func.deprecated!r})\n"
        sig = f"def f{self.type_params(func.type_params)}({self.params(func.params)})"
        return f"{head}{sig} -> {self.render_node(func.ret)}: ..."

    def import_block(
        self,
        locals_: AbstractSet[str],
        tyvars: AbstractSet[str],
    ) -> str:
        """The import lines for referenced names that are neither helper nor typevar."""
        groups: dict[str, set[str]] = {}
        whole: set[str] = set()
        for name in self.used - locals_ - tyvars:
            if (found := import_of(name)) is None:
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
        lines += [
            f"import {m} as {alias}" for m, alias in sorted(self._aliases.items())
        ]
        order = {"collections.abc": 0, "types": 1, "typing": 2, "typing_extensions": 3}
        for module_name in sorted(groups, key=lambda m: (order.get(m, 9), m)):
            members = ", ".join(sorted(groups[module_name]))
            lines.append(f"from {module_name} import {members}")
        return "\n".join(lines)


def type_text(node: _ir.Node) -> str:
    """A node's canonical text, for dedup keys; the recorded names are discarded."""
    return Printer().render_node(node)


def member_key(member: Member) -> str:
    if isinstance(member, Attr):
        flags = f"{member.classvar}{member.readonly}"
        setter = "" if member.setter is None else f"={type_text(member.setter)}"
        return f"{member.name}:{flags}:{type_text(member.type)}{setter}"
    params = Printer().call_params(member.params)
    return f"{member.name}:{params}->{type_text(member.ret)}"
