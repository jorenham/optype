"""In NumPy <2.0, there are at most 32 dimensions."""

from typing import TypeAlias


_S00: TypeAlias = tuple[()]
_S01: TypeAlias = tuple[int]
_S02: TypeAlias = tuple[int, int]
_S03: TypeAlias = tuple[int, int, int]
_S04: TypeAlias = tuple[int, int, int, int]
_S05: TypeAlias = tuple[int, int, int, int, int]
_S06: TypeAlias = tuple[int, int, int, int, int, int]
_S07: TypeAlias = tuple[int, int, int, int, int, int, int]
_S08: TypeAlias = tuple[int, int, int, int, int, int, int, int]
_S09: TypeAlias = tuple[int, int, int, int, int, int, int, int, int]
_S10: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int]
_S11: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int,
]
_S12: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int,
]
_S13: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int,
]
_S14: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
]
_S15: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int,
]
_S16: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int,
]
_S17: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int,
]
_S18: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int,
]
_S19: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int,
]
_S20: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int,
]
_S21: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int,
]
_S22: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int,
]
_S23: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int,
]
_S24: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int,
]
_S25: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int,
]
_S26: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int, int,
]
_S27: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int, int, int,
]
_S28: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
]
_S29: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int,
]
_S30: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int,
]
_S31: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int,
]
_S32: TypeAlias = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int,
]

AtLeast3D: TypeAlias = (
    _S03 | _S04 | _S05 | _S06 | _S07 | _S08
    | _S09 | _S10 | _S11 | _S12 | _S13 | _S14 | _S15 | _S16
    | _S17 | _S18 | _S19 | _S20 | _S21 | _S22 | _S23 | _S24
    | _S25 | _S26 | _S27 | _S28 | _S29 | _S30 | _S31 | _S32
)
AtLeast2D: TypeAlias = _S02 | AtLeast3D
AtLeast1D: TypeAlias = _S01 | AtLeast2D
AtLeast0D: TypeAlias = _S00 | AtLeast1D
