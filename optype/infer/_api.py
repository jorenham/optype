"""The `infer` entry point: explore, probe the defaults, render."""

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from inspect import Parameter

import optype.infer._numpy as _numpy
from ._backends import BACKENDS, BackendName
from ._errors import WARN_SKIP_PREFIX, InferError, InferWarning
from ._explore import explore_lenient, explore_tuple_params
from ._gc import pause_gc
from ._ir import Signature
from ._isolate import isolate
from ._overloads import dispatch_overloads, resolve_defaults
from ._render import Names, signatures
from ._signature import parse_text_signature, probe_signatures, signature
from ._spy import _AnyFunc
from ._values import GapKind

# parameter names, positions, or empty for all
type _Selectors = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class _Gap:
    """A coverage gap at a call form."""

    kind: GapKind
    where: str  # the call form, e.g. "(x, y)"

    def message(self) -> str:
        return f"{self.kind} in {self.where}"


def _select(selectors: Iterable[str | int], names: Names) -> Names:
    selected: list[str] = []
    for p in selectors:
        if isinstance(p, int):
            if not -len(names) <= p < len(names):
                msg = f"no parameter at position {p}"
                raise ValueError(msg)
            selected.append(names[p])
        elif p in names:
            selected.append(p)
        else:
            msg = f"unknown parameter {p!r}"
            raise ValueError(msg)
    return selected or names


def _form_signatures(
    func: _AnyFunc,
    params: Mapping[str, Parameter],
    selectors: _Selectors,
    gaps: set[_Gap],
) -> list[Signature]:
    selected = _select(selectors, list(params))
    where = f"({', '.join(params)})"
    try:
        exploration, fallback = explore_lenient(func, params)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise InferError(str(exc)) from exc

    gaps.update(_Gap(kind, where) for kind in exploration.gaps)

    if fallback:
        # the rejected parameters render from their defaults; skip the probing
        return signatures(exploration, params, selected, fallback)

    exploration = exploration._replace(
        tuple_params=explore_tuple_params(func, params, exploration),
    )
    defaults, negate, overloads = resolve_defaults(func, params, selected, exploration)
    lines = signatures(exploration, params, selected, defaults, negate=negate)
    if not defaults and not overloads:
        # dispatch only when defaults didn't already split the form
        lines = dispatch_overloads(func, params, selected, exploration, lines)
    return [*overloads, *lines]


def _candidate_parameters(func: _AnyFunc) -> list[dict[str, Parameter]]:
    try:
        return [dict(signature(func).parameters)]
    except TypeError as exc:  # not callable
        raise InferError(str(exc)) from exc
    except ValueError as exc:  # callable but no signature (e.g. a C builtin)
        if candidates := parse_text_signature(func) or probe_signatures(func):
            return candidates
        raise InferError(str(exc)) from exc


def _infer(func: _AnyFunc, selectors: _Selectors, gaps: set[_Gap]) -> list[Signature]:
    if nin := _numpy.ufunc_nin(func):
        names = _numpy.ufunc_params(nin)
        return _numpy.infer_ufunc(func, names, _select(selectors, names))

    sigs: list[Signature] = []
    last: Exception | None = None
    for params in _candidate_parameters(func):
        try:
            sigs += _form_signatures(func, params, selectors, gaps)
        except (InferError, ValueError) as exc:
            last = exc  # re-raised below only if no arity produces anything
    if not sigs and last is not None:
        raise last
    return sigs


def _infer_render(
    func: _AnyFunc,
    selectors: _Selectors,
    *,
    strict: bool,
    backend: BackendName,
) -> str:
    gaps: set[_Gap] = set()
    try:
        with pause_gc():
            sigs = _infer(func, selectors, gaps)
    except RecursionError as exc:
        raise InferError("the result is nested too deeply") from exc

    if gaps:
        detail = "; ".join(sorted(g.message() for g in gaps))
        msg = f"incomplete exploration: {detail}"
        if strict:
            raise InferError(msg)
        warnings.warn(msg, InferWarning, skip_file_prefixes=(WARN_SKIP_PREFIX,))

    return BACKENDS[backend].render(sigs)


def infer(
    func: _AnyFunc,
    /,
    *selectors: str | int,
    strict: bool = False,
    backend: BackendName = "terse",
) -> str:
    """Infer the `optype` protocol(s) required of `func`'s parameters.

    Pass parameter names or positions to report only those parameters.

    >>> print(infer(lambda x: x + 1))
    [R](x: CanAdd[Literal[1], R]) -> R

    When exploration is incomplete it emits an `InferWarning` naming the affected
    call form; pass `strict=True` to raise an `InferError` instead.

    Pass `backend="compat"` to render a self-contained, type-checkable `.pyi`-style
    stub instead of the default terse form.

    Raises:
        InferError: If `func` is not supported, such as a non-callable, an
            operation without a matching protocol, or a parameter that requires
            a value that no placeholder can provide; or, with `strict=True`, if
            the function could not be explored exhaustively.
    """  # noqa: DOC502

    def render() -> str:
        return _infer_render(func, selectors, strict=strict, backend=backend)

    # even a pure-python function can reach native code that can segfault (#738, #763)
    return isolate(render)
