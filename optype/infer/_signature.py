"""Recover a signatureless callable's parameters by trial and error.

Many C builtins (`type`, `iter`, `min`, ...) have no `inspect.signature`. Calling them
with spy placeholders at each arity reveals which arities the body accepts: a correct
arity runs the body (records a trace or returns), while a wrong one raises `TypeError`
before the body runs.
"""

from inspect import Parameter

from ._spy import _AnyFunc, _Fork, _SpyObject

_MAX_PROBE_ARITY = 8


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
