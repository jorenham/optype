"""ANSI colorization for the `optype infer` command-line output."""

import keyword
import os
import re
from typing import IO, Literal

type ColorMode = Literal["auto", "always", "never"]

_RESET = "\x1b[0m"
_COLORS = {"str": "\x1b[32m", "num": "\x1b[36m", "kw": "\x1b[34m"}
# `str` first so a keyword inside a literal stays a string; hard keywords only
_HIGHLIGHT = re.compile(
    r"(?P<str>'[^']*'|\"[^\"]*\")"
    r"|(?P<num>\b\d[\w.]*)"
    r"|(?P<kw>\b(?:" + "|".join(keyword.kwlist) + r")\b)",
)


def want_color(stream: IO[str], mode: ColorMode = "auto") -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    if os.environ.get("NO_COLOR"):  # wins over FORCE_COLOR (no-color.org)
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return stream.isatty()


def highlight(signature: str) -> str:
    def _wrap(m: re.Match[str]) -> str:
        name = m.lastgroup
        return f"{_COLORS[name]}{m.group()}{_RESET}" if name else m.group()

    return _HIGHLIGHT.sub(_wrap, signature)
