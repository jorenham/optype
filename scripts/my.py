#!/usr/bin/env python

"""Run as `uv run scripts/my.py` from the repo root."""

import subprocess
import sys
from collections.abc import Generator
from typing import Final, TypeAlias

_Version: TypeAlias = tuple[int, int]

# TODO: figure these out dynamically, e.g. with `distutils` or from `pyproject.toml`
_NP1_MIN: Final = 1, 25
_NP1_MAX: Final = 1, 26
_NP2_MIN: Final = 2, 0
_NP2_MAX: Final = 2, 3
_NP_SKIP: Final = frozenset({(1, 26)})


def _np_version() -> _Version:
    import numpy as np  # noqa: PLC0415

    major, minor = map(int, np.__version__.split(".", 3)[:2])
    version = major, minor
    assert _NP1_MIN <= version <= _NP2_MAX
    return version


def _np_version_range(
    first: _Version = _NP1_MIN,
    last: _Version = _NP2_MAX,
) -> Generator[_Version]:
    assert _NP1_MIN <= first <= _NP2_MAX
    assert _NP1_MIN <= last <= _NP2_MAX

    if first >= last:
        return

    v0, v1 = first

    while v0 < last[0]:
        if v1 > _NP1_MAX[1]:
            assert v0 == 1
            v0, v1 = _NP2_MIN
            break

        if (v0, v1) not in _NP_SKIP:
            yield v0, v1
        v1 += 1

    assert v0 == last[0]
    assert v1 <= last[1]

    for v in range(v1, last[1] + 1):
        if (v0, v) not in _NP_SKIP:
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
