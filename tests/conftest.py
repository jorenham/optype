"""The fixtures shared across the test modules."""

import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def basedpyright() -> Callable[[Path], subprocess.CompletedProcess[str]]:
    """Type-check a directory of stubs from within it, apart from the project."""

    def check(path: Path) -> subprocess.CompletedProcess[str]:
        # skipped at the call, so a test's own assertions run first
        if shutil.which("basedpyright") is None:
            pytest.skip("basedpyright is not installed")
        return subprocess.run(
            ["basedpyright", "."],  # ruff: ignore[start-process-with-partial-path]
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )

    return check
