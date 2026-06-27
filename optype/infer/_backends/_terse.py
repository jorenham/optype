"""The default terse renderer: compact, but not valid Python."""

from collections.abc import Sequence
from typing import Final, assert_never, final

# `from optype.infer import _ir` would re-enter the package, which imports this module
import optype.infer._ir as _ir  # noqa: PLR0402
from ._base import Backend, default_text

_NOT = "~"  # the type complement prefix
_OR = "|"  # the union separator
_AND = "&"  # the intersection separator
_DOTS = "..."  # the `...` ellipsis
_STAR = "*"  # the unpack and typevar-tuple prefix


@final
class TerseBackend:
    """The historical terse renderer; not valid Python, but compact."""

    def render(self, sigs: Sequence[_ir.Signature], /) -> str:
        return "\n".join(dict.fromkeys(map(self._line, sigs)))

    def _line(self, sig: _ir.Signature, /) -> str:
        type_params = ", ".join(map(self._type_param, sig.type_params))
        type_params = f"[{type_params}]" if sig.type_params else ""
        params = ", ".join(map(self._param, sig.params))
        line = f"{type_params}({params}) -> {self._render_node(sig.ret)}"
        if sig.deprecated is not None:
            line = f"@deprecated({sig.deprecated!r})\n{line}"
        return line

    def _render_node(self, node: _ir.Node, /) -> str:
        """Format a type expression, parenthesized where precedence requires."""
        match node:
            case _ir.Lit(values):
                out = f"Literal[{', '.join(map(repr, values))}]"
            case _ir.Type(cls):
                out = _ir.type_name(cls)
            case _ir.Name(name):
                out = name
            case _ir.Dots():
                out = _DOTS
            case _ir.App(base, args):
                out = self._app(base, args)
            case _ir.Fn(params, ret):
                out = self._fn(params, ret)
            case _ir.Not() | _ir.Variance() | _ir.Unpack():
                out = self._prefixed(node)
            case _ir.Union() | _ir.Inter():
                out = self._infixed(node)
        return out

    def _prefix(self, op: str, part: _ir.Node) -> str:
        inner = self._render_node(part)
        if isinstance(part, (_ir.Union, _ir.Inter, _ir.Fn)):
            inner = f"({inner})"
        return f"{op}{inner}"

    def _app(self, base: str, args: tuple[_ir.Node | _ir.Arg, ...]) -> str:
        if base == "tuple" and not args:
            parts = ["()"]
        else:
            parts = [
                f"{arg.key}={self._render_node(arg.value)}"
                if isinstance(arg, _ir.Arg)
                else self._render_node(arg)
                for arg in args
            ]
        return f"{base}[{', '.join(parts)}]" if parts else base

    def _arg(self, param: _ir.Node | _ir.Arg) -> str:
        if not isinstance(param, _ir.Arg):
            return self._render_node(param)
        label = f"{param.key}: " if param.key else ""
        decl = f"{label}{self._render_node(param.value)}"
        if param.default is not None:
            decl += f" = {default_text(param.default[0])}"
        return decl

    def _fn(self, params: tuple[_ir.Node | _ir.Arg, ...], ret: _ir.Node) -> str:
        decls = ", ".join(map(self._arg, params))
        return f"({decls}) -> {self._render_node(ret)}"

    def _prefixed(self, node: _ir.Not | _ir.Variance | _ir.Unpack) -> str:
        match node:
            case _ir.Variance(sign, part):
                op = sign
            case _ir.Not(part):
                op = _NOT
            case _ir.Unpack(part):
                op = _STAR
            case _:
                assert_never(node)

        return self._prefix(op, part)

    def _infixed(self, node: _ir.Union | _ir.Inter) -> str:
        match node:
            case _ir.Union(parts):
                sep, dual = _OR, _ir.Inter
            case _ir.Inter(parts):
                sep, dual = _AND, _ir.Union
            case _:
                assert_never(node)

        return f" {sep} ".join(
            f"({self._render_node(part)})"
            if isinstance(part, (dual, _ir.Fn))
            else self._render_node(part)
            for part in parts
        )

    def _type_param(self, tp: _ir.TypeParam) -> str:
        if tp.unpack:
            return f"{_STAR}{tp.name}"

        decl = (
            f"{tp.name}: {self._render_node(tp.bound)}"
            if tp.bound is not None
            else tp.name
        )
        if tp.default is not None:
            decl += f" = {self._render_node(tp.default)}"
        return decl

    def _param(self, param: _ir.Param) -> str:
        label = "" if param.nameless else f"{param.prefix}{param.name}: "
        decl = f"{label}{self._render_node(param.node)}"
        if param.default is not None:
            decl += f" = {default_text(param.default[0])}"
        return decl


TERSE: Final[Backend] = TerseBackend()
