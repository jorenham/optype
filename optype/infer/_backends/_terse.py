"""The default terse renderer: compact, but not valid Python."""

from collections.abc import Sequence

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]
from ._base import default_text, value_text


def render(sigs: Sequence[_ir.Signature], /) -> str:
    return "\n".join(dict.fromkeys(map(_line, sigs)))


def _line(sig: _ir.Signature, /) -> str:
    type_params = ", ".join(map(_type_param, sig.type_params))
    type_params = f"[{type_params}]" if sig.type_params else ""
    params = ", ".join(map(_param, sig.params))
    line = f"{type_params}({params}) -> {_render_node(sig.ret)}"
    if sig.deprecated is not None:
        line = f"@deprecated({sig.deprecated!r})\n{line}"
    return line


def _render_node(node: _ir.Node, /) -> str:  # ruff: ignore[complex-structure, too-many-branches]
    """Format a type expression, parenthesized where precedence requires."""
    match node:
        case _ir.Lit(values):
            out = f"Literal[{', '.join(map(value_text, values))}]"
        case _ir.Type(cls):
            out = _ir.type_name(cls)
        case _ir.Name(name):
            out = name
        case _ir.Dots():
            out = "..."
        case _ir.App(origin, args):
            out = _app(origin, args)
        case _ir.Has(attr, args):
            parts = [repr(attr), *map(_render_node, args)]
            out = f"Has[{', '.join(parts)}]"
        case _ir.Fn(params, ret):
            out = _fn(params, ret)
        case _ir.Not(part):
            out = _prefix("~", part)
        case _ir.Covariant(part):
            out = _prefix(_ir.COVARIANT, part)
        case _ir.Contravariant(part):
            out = _prefix(_ir.CONTRAVARIANT, part)
        case _ir.Unpack(part):
            out = _prefix("*", part)
        case _ir.Union(parts):
            out = _infix("|", parts, _ir.Intersection)
        case _ir.Intersection(parts):
            out = _infix("&", parts, _ir.Union)
    return out


def _prefix(op: str, part: _ir.Node) -> str:
    inner = _render_node(part)
    if isinstance(part, (_ir.Union, _ir.Intersection, _ir.Fn)):
        inner = f"({inner})"
    return f"{op}{inner}"


def _app(origin: str, args: _ir.Terms) -> str:
    if origin == "tuple" and not args:
        parts = ["()"]
    else:
        parts = [
            f"{arg.key}={_render_node(arg.value)}"
            if isinstance(arg, _ir.Arg)
            else _render_node(arg)
            for arg in args
        ]
    return f"{origin}[{', '.join(parts)}]" if parts else origin


def _arg(param: _ir.Term) -> str:
    if not isinstance(param, _ir.Arg):
        return _render_node(param)
    label = f"{param.key}: " if param.key else ""
    decl = f"{label}{_render_node(param.value)}"
    if param.default is not None:
        decl += f" = {default_text(param.default[0])}"
    return decl


def _fn(params: _ir.Terms, ret: _ir.Node) -> str:
    decls = ", ".join(map(_arg, params))
    return f"({decls}) -> {_render_node(ret)}"


def _infix(
    sep: str,
    parts: tuple[_ir.Node, ...],
    dual: type[_ir.Union | _ir.Intersection],
) -> str:
    return f" {sep} ".join(
        f"({_render_node(part)})"
        if isinstance(part, (dual, _ir.Fn))
        else _render_node(part)
        for part in parts
    )


def _type_param(typar: _ir.TypeParam) -> str:
    if typar.unpack:
        return f"*{typar.name}"

    decl = (
        f"{typar.name}: {_render_node(typar.bound)}"
        if typar.bound is not None
        else typar.name
    )
    if typar.default is not None:
        decl += f" = {_render_node(typar.default)}"
    return decl


def _param(param: _ir.Param) -> str:
    label = "" if param.pos_only else f"{param.prefix}{param.name}: "
    decl = f"{label}{_render_node(param.node)}"
    if param.default is not None:
        decl += f" = {default_text(param.default[0])}"
    return decl
