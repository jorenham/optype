"""Recovering a signatureless builtin's parameters: the text signature and the probe."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false

import select
from inspect import Signature as PySignature, signature
from types import SimpleNamespace
from typing import Any

import pytest

from optype.infer import InferError, infer
from optype.infer._signature import parse_text_signature


def _skip_if_signature(func: Any) -> None:
    try:
        signature(func)
    except ValueError:
        pass
    else:
        pytest.skip(
            f"this build exposes an `inspect.signature` for `{func.__qualname__}`",
        )


def test_builtin_without_signature() -> None:
    _skip_if_signature(iter)
    # the arity probe recovers a signatureless builtin instead of raising, exploring
    # each accepted arity as a separate overload
    assert infer(iter) == "[R](CanIter[R]) -> R\n[R](() -> R, object) -> Iterator[R]"


def _text_candidates(text: str, *, bound: bool = False) -> list[str] | None:
    func: Any = SimpleNamespace(__text_signature__=text)
    if bound:
        func.__self__ = object()
    parsed = parse_text_signature(func)
    if parsed is None:
        return None
    return [str(PySignature(list(c.values()))) for c in parsed]


@pytest.mark.parametrize(
    ("text", "bound", "expected"),
    [
        # the legacy find-family grammar: each optional group level is a candidate
        # (#646)
        (
            "($self, sub[, start[, end]], /)",
            False,
            ["(self, sub, /)", "(self, sub, start, /)", "(self, sub, start, end, /)"],
        ),
        # a bound callable's `$` parameter is not part of the call signature
        (
            "($module, aiterator, default=<unrepresentable>, /)",
            True,
            ["(aiterator, /)", "(aiterator, default, /)"],
        ),
        # omitting an unrepresentable default forces later parameters to keyword-only
        (
            "($self, /, sep=<unrepresentable>, bytes_per_sep=1)",
            False,
            ["(self, /, *, bytes_per_sep=1)", "(self, /, sep, bytes_per_sep=1)"],
        ),
        # a leading (curses-style) group shifts the positional arity instead
        (
            "([y, x,] ch[, attr])",
            False,
            ["(ch)", "(ch, attr)", "(y, x, ch)", "(y, x, ch, attr)"],
        ),
        # literal defaults, `*args`, keyword-only, and `**kwargs` parse as-is
        (
            "($self, /, x=0, *args, key, **kwargs)",
            False,
            ["(self, /, x=0, *args, key, **kwargs)"],
        ),
    ],
)
def test_text_signature(text: str, bound: bool, expected: list[str]) -> None:
    assert _text_candidates(text, bound=bound) == expected


@pytest.mark.parametrize(
    "text",
    ["no parens", "(unclosed", "(a[, b)", "(a, a)", "(a, $b)", "(=1)"],
)
def test_text_signature_invalid(text: str) -> None:
    assert _text_candidates(text) is None


def test_builtin_dict_pop() -> None:
    _skip_if_signature(dict.pop)
    # the 2-parameter form never completes (`KeyError` on an empty `dict`)
    assert infer(dict.pop) == "[T](dict, object, T) -> T"


def test_builtin_bytes_hex() -> None:
    _skip_if_signature(bytes.hex)
    # the str-typed `sep` rejects placeholders; the sep-less candidate renders
    assert infer(bytes.hex) == "(bytes, bytes_per_sep: CanIndex = 1) -> str"


def test_builtin_str_index() -> None:
    _skip_if_signature(str.index)
    if getattr(str.index, "__text_signature__", None) is None:
        pytest.skip("this build has no `__text_signature__` for `str.index`")
    # the text signature parses (#646), but the str-typed `sub` rejects placeholders
    with pytest.raises(InferError, match="must be str"):
        infer(str.index)


def test_text_signature_wraps() -> None:
    # gh-772: a builtin's text signature can wrap, with a default `inspect` can't eval
    text = "($self, /, fd,\n         eventmask=select.EPOLLIN | select.EPOLLOUT)"
    assert _text_candidates(text) == ["(self, /, fd)", "(self, /, fd, eventmask)"]


@pytest.mark.skipif(not hasattr(select, "epoll"), reason="requires select.epoll")
def test_builtin_epoll_register() -> None:
    # gh-772: `inspect` evaluates the defaults where `select` is the function, not the
    # module; the text signature is parsed instead, and the fd rejects a placeholder
    with pytest.raises(InferError, match="fileno"):
        infer(select.epoll.register)
