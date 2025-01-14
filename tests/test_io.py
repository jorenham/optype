# ruff: noqa: N802
import os
import pathlib

import pytest

import optype.io as oio


@pytest.mark.parametrize("pathlike", [pathlib.Path()])
def test_PathLike_to_CanFSPath(pathlike: os.PathLike[str]) -> None:
    p: oio.CanFSPath[str] = pathlike
    assert isinstance(pathlike, os.PathLike)
    assert isinstance(pathlike, oio.CanFSPath)


@pytest.mark.parametrize("pathlike", [pathlib.Path()])
def test_CanFSPath_to_PathLike(pathlike: oio.CanFSPath[str]) -> None:
    p: os.PathLike[str] = pathlike
    assert isinstance(pathlike, oio.CanFSPath)
    assert isinstance(pathlike, os.PathLike)
