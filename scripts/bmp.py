#!/usr/bin/env python

"""Run as `uv run scripts/bmp.py` from the repo root."""

import subprocess
import sys
from typing import Final, TypeAlias

from typing_extensions import Generator


_Version: TypeAlias = tuple[int, int]

# TODO: figure these out dynamically, e.g. with `distutils` or from `pyproject.toml`
_NP_MIN: Final = 1, 23
_NP_MAX: Final = 2, 2

_NP_MAX_1: Final = 26


def _np_version() -> _Version:
    import numpy as np  # noqa: PLC0415

    major, minor = map(int, np.__version__.split(".", 3)[:2])
    version = major, minor
    assert _NP_MIN <= version <= _NP_MAX
    return version


def _np_version_range(
    first: _Version = _NP_MIN,
    last: _Version = _NP_MAX,
) -> Generator[_Version]:
    assert _NP_MIN <= first <= _NP_MAX
    assert _NP_MIN <= last <= _NP_MAX

    if first >= last:
        return

    v0, v1 = first

    while v0 < last[0]:
        if v1 > _NP_MAX_1:
            assert v0 == 1
            v0 += 1
            v1 = 0
            break

        yield v0, v1
        v1 += 1

    assert v0 == last[0]
    assert v1 <= last[1]

    for v in range(v1, last[1] + 1):
        yield v0, v


def main(*args: str) -> int:
    v0 = _np_version()

    cmd: list[str] = [
        "mypy",
        "--tb",
        "--hide-error-context",
        "--hide-error-code-links",
    ]
    for vi in _np_version_range():
        const = f"NP{vi[0]}{vi[1]}"
        supported = "true" if v0 >= vi else "false"
        cmd.append(f"--always-{supported}={const}")

    if not args:
        cmd.append(".")
    else:
        cmd.extend(args)

    print(*cmd)  # noqa: T201
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
