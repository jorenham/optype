# ruff: noqa: F841
import sys
from typing import Literal, cast

import optype as opt


if sys.version_info >= (3, 13):
    from typing import assert_type
else:
    from typing_extensions import assert_type


def test_slice_obj():
    s = slice(42)

    s_default_0: opt.types.Slice = s
    s_default_1: opt.types.Slice[None] = s
    s_default_2: opt.types.Slice[None, int] = s
    s_default_3: opt.types.Slice[None, int, None] = s

    s_wrong_1: opt.types.Slice[str] = s
    s_wrong_2: opt.types.Slice[None, str] = s
    s_wrong_3: opt.types.Slice[None, int, str] = s

    assert isinstance(s, opt.types.Slice)


# suppress false positives in `basedpyright==1.15.1`
# pyright: reportInvalidCast=false

def test_slice_indices():
    s0 = cast(opt.types.Slice[None, None], slice(None))
    i0 = s0.indices(32)
    assert_type(i0, tuple[Literal[0], int, Literal[1]])

    s1 = cast(opt.types.Slice[None, Literal[42], None], slice(42))
    i1 = s1.indices(32)
    assert_type(i1, tuple[Literal[0], int, Literal[1]])

    s2 = cast(opt.types.Slice[Literal[6], Literal[42], None], slice(6, 42))
    i2 = s2.indices(32)
    assert_type(i2, tuple[int, int, Literal[1]])

    s3 = cast(
        opt.types.Slice[Literal[6], Literal[42], Literal[7]],
        slice(6, 42, 7),
    )
    i3 = s3.indices(32)
    assert_type(i3, tuple[int, int, int])
