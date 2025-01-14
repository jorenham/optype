# ruff: noqa: N802
import io
import os
import pathlib
from typing import IO

import pytest

import optype.io as oio
from optype.inspect import get_args


@pytest.mark.parametrize("pathlike", [pathlib.Path()])
def test_PathLike_to_CanFSPath(pathlike: os.PathLike[str]) -> None:
    p: oio.CanFSPath[str] = pathlike
    assert isinstance(pathlike, os.PathLike)
    assert isinstance(pathlike, oio.CanFSPath)


@pytest.mark.parametrize("can_fspath", [pathlib.Path()])
def test_CanFSPath_to_PathLike(can_fspath: oio.CanFSPath[str]) -> None:
    p: os.PathLike[str] = can_fspath
    assert isinstance(can_fspath, oio.CanFSPath)
    assert isinstance(can_fspath, os.PathLike)


@pytest.mark.parametrize("stream", [io.BytesIO()])
def test_IO_to_CanFileno(stream: IO[bytes] | IO[str]) -> None:
    f: oio.CanFileno = stream
    assert isinstance(stream, oio.CanFileno)


@pytest.mark.parametrize("mode", get_args(oio.Mode_B_), ids=str)
def test_mode_b(mode: str) -> None:
    assert len(mode) in {2, 3}
    assert "b" in mode
    assert "t" not in mode
    assert "+" in mode or len(mode) < 3


@pytest.mark.parametrize("mode", get_args(oio.Mode_T_), ids=str)
def test_mode_t(mode: str) -> None:
    assert len(mode) in {2, 3} or "t" not in mode
    assert "b" not in mode
    assert "t" in mode or len(mode) < (2 + ("+" in mode))
