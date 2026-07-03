"""Recover a signatureless callable's parameters.

Many C builtins have no `inspect.signature`. Those with a legacy `__text_signature__`
(`str.index`, `dict.pop`, ...) declare optional groups (`sub[, start[, end]]`) or
`=<unrepresentable>` defaults that `inspect` rejects; `parse_text_signature` recovers
their parameters directly. The rest (`type`, `iter`, `min`, ...) are probed by trial
and error: calling them with spy placeholders reveals which arities run the body.
"""

import ast
import itertools
import re
from collections.abc import Iterator
from inspect import Parameter
from typing import Final, NamedTuple

from ._spy import _AnyFunc, _Fork, _SpyObject

_MAX_PROBE_ARITY = 8

# at most 2 in practice; 4 caps the expansion at 16 candidates
_MAX_TOGGLES = 4

_NAME = re.compile(r"[A-Za-z_]\w*")
_DEFAULT = re.compile(r"[^,()\[\]{}]+")

# a non-literal (`<unrepresentable>`) default; the parameter becomes a toggle
_UNKNOWN: Final = object()


class _ParseError(ValueError):
    """The text signature does not follow the (legacy) grammar."""


class _RawParam(NamedTuple):
    param: Parameter
    group: int  # innermost enclosing optionality toggle, -1 when required


def _positional_only(raw: list[_RawParam]) -> list[_RawParam]:
    return [
        _RawParam(p.param.replace(kind=Parameter.POSITIONAL_ONLY), p.group)
        if p.param.kind is Parameter.POSITIONAL_OR_KEYWORD
        else p
        for p in raw
    ]


def _scan_group(parents: list[int], group: int, *, opens: bool) -> int:
    if opens:
        parents.append(group)
        return len(parents) - 1
    if group < 0:
        raise _ParseError
    return parents[group]


def _scan_variadic(text: str, i: int, raw: list[_RawParam], group: int) -> int:
    var_keyword = text.startswith("**", i)
    i += 2 if var_keyword else 1
    if match := _NAME.match(text, i):
        var = Parameter.VAR_KEYWORD if var_keyword else Parameter.VAR_POSITIONAL
        raw.append(_RawParam(Parameter(match[0], var), group))
        return match.end()
    if var_keyword:
        raise _ParseError
    return i


def _scan_param(text: str, i: int, *, first: bool) -> tuple[str, object, bool, int]:
    # `$self`/`$module` may only come first, at the top level
    if (dollar := text[i] == "$") and not first:
        raise _ParseError
    i += dollar
    if (match := _NAME.match(text, i)) is None:
        raise _ParseError
    i = match.end()
    default: object = Parameter.empty
    if text.startswith("=", i):
        # defaults are literals, dotted names, or `<unrepresentable>`, never bracketed
        if (expr := _DEFAULT.match(text, i + 1)) is None:
            raise _ParseError
        i = expr.end()
        try:
            default = ast.literal_eval(expr[0])
        except (SyntaxError, ValueError, TypeError):
            default = _UNKNOWN
    return match[0], default, dollar, i


def _scan(text: str) -> tuple[list[_RawParam], list[int], bool]:
    """The raw parameters, the group-parent forest, and whether `$self` was seen.

    Raises:
        _ParseError: If the text signature does not follow the (legacy) grammar.
    """
    text = text.strip()
    if not text.startswith("("):
        raise _ParseError

    raw: list[_RawParam] = []
    parents: list[int] = []  # each group's parent group, -1 at the top level
    group = -1
    kind = Parameter.POSITIONAL_OR_KEYWORD
    dollar = closed = False
    i = 1
    while i < len(text) and not closed:
        c = text[i]
        if c in " ,":
            i += 1
        elif c == ")":
            closed = True
        elif c in "[]":
            group = _scan_group(parents, group, opens=c == "[")
            i += 1
        elif c == "/":
            raw = _positional_only(raw)
            i += 1
        elif c == "*":
            i = _scan_variadic(text, i, raw, group)
            kind = Parameter.KEYWORD_ONLY
        else:
            first = not raw and group == -1
            name, default, seen, i = _scan_param(text, i, first=first)
            param_kind = Parameter.POSITIONAL_ONLY if seen else kind
            raw.append(_RawParam(Parameter(name, param_kind, default=default), group))
            dollar = dollar or seen

    if not closed or group != -1:
        raise _ParseError
    return raw, parents, dollar


def _combinations(parents: list[int]) -> Iterator[tuple[bool, ...]]:
    # the valid toggle activations: a nested toggle requires its parent
    for active in itertools.product((False, True), repeat=len(parents)):
        if all(
            parents[g] < 0 or active[parents[g]] for g, on in enumerate(active) if on
        ):
            yield active


def _candidate(
    raw: list[_RawParam],
    active: tuple[bool, ...],
    groups: int,
) -> dict[str, Parameter] | None:
    """The parameters of one toggle activation, or `None` if it cannot be called."""
    params: dict[str, Parameter] = {}
    gap = False
    for param, group in raw:
        if group >= 0 and not active[group]:
            # an omitted default forces later parameters to keyword (as in the
            # explorer's `_placeholders`); a dropped group shifts the arity instead
            gap = gap or (group >= groups and param.kind is not Parameter.KEYWORD_ONLY)
            continue
        if gap:
            match param.kind:
                case Parameter.POSITIONAL_OR_KEYWORD:
                    param = param.replace(kind=Parameter.KEYWORD_ONLY)  # noqa: PLW2901
                case Parameter.POSITIONAL_ONLY | Parameter.VAR_POSITIONAL:
                    return None
                case _:
                    pass
        params[param.name] = param
    return params


def parse_text_signature(func: _AnyFunc) -> list[dict[str, Parameter]] | None:
    """Candidate parameter mappings parsed from `__text_signature__`, or `None`.

    Optional groups and unrepresentable defaults expand into one candidate per valid
    combination, mirroring `probe_signatures`' per-arity candidates.
    """
    text: object = getattr(func, "__text_signature__", None)
    if not isinstance(text, str):
        return None
    try:
        raw, parents, dollar = _scan(text)
    except _ParseError:
        return None

    if dollar and getattr(func, "__self__", None) is not None:
        raw = raw[1:]  # the bound argument is not part of the call signature
    if len({p.param.name for p in raw}) != len(raw):
        return None

    groups = len(parents)
    toggles = list(parents)
    for index, p in enumerate(raw):
        if p.param.default is _UNKNOWN:
            toggles.append(p.group)
            raw[index] = _RawParam(
                p.param.replace(default=Parameter.empty),
                len(toggles) - 1,
            )
    if len(toggles) > _MAX_TOGGLES:
        return None

    combos = sorted(_combinations(toggles), key=sum)
    return [c for on in combos if (c := _candidate(raw, on, groups)) is not None]


def _accepts(func: _AnyFunc, n: int) -> bool:
    """Whether `func` runs its body when called with `n` positional placeholders."""
    spies = [_SpyObject() for _ in range(n)]
    try:
        func(*spies)
    except (_Fork, Exception, SystemExit) as exc:  # noqa: BLE001
        # a wrong-arity `TypeError` is raised before the body runs, leaving the spies
        # untouched; a touched spy or any other error (or exit) means the body ran
        if type(exc) is TypeError:
            return any(spy.__optype_trace__ for spy in spies)
    return True


def _params(n: int, *, var_positional: bool) -> dict[str, Parameter]:
    params = {f"_{i}": Parameter(f"_{i}", Parameter.POSITIONAL_ONLY) for i in range(n)}
    if var_positional:
        params["args"] = Parameter("args", Parameter.VAR_POSITIONAL)
    return params


def probe_signatures(func: _AnyFunc) -> list[dict[str, Parameter]] | None:
    """A synthetic parameter mapping per explorable arity, or `None` if none run."""
    if not (arities := {n for n in range(_MAX_PROBE_ARITY + 1) if _accepts(func, n)}):
        return None

    if _MAX_PROBE_ARITY in arities:
        # an unbounded arity is variadic: one `*args` the explorer grows into
        return [_params(min(arities), var_positional=True)]

    return [_params(n, var_positional=False) for n in arities]
