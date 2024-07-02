"""Panic in case of unexpected `numpy` sized C-type aliases."""
from __future__ import annotations

import ctypes as ct
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from types import ModuleType


_ALIASES_CT = (
    ('c_uint8', 'c_ubyte'),
    ('c_uint16', 'c_ushort'),
    ('c_uint32', 'c_uint'),
    ('c_uint64', 'c_ulonglong'),

    ('c_int8', 'c_byte'),
    ('c_int16', 'c_short'),
    ('c_int32', 'c_int'),
    ('c_int64', 'c_longlong'),
)
_ALIASES_NP = (
    ('uint8', 'ubyte'),
    ('uint16', 'ushort'),

    ('int8', 'byte'),
    ('int16', 'short'),

    ('float16', 'half'),
    ('float32', 'single'),

    ('complex64', 'csingle'),
    ('complex128', 'cdouble'),
)


class UnexpectedAliasError(OSError): ...


def check_aliases(
    module: ModuleType,
    aliases: tuple[tuple[str, str], ...],
    /,
) -> None:
    for name_a, name_b in aliases:
        t_a, t_b = getattr(module, name_a), getattr(module, name_b)
        if t_a is t_b:
            continue

        # assumes that either `t_a` or `t_b` isn't an alias
        if t_a.__name__ == name_a:
            t_alias, name_alias, name_orig = t_b, name_b, name_a
        else:
            assert t_b.__name__ == name_b
            t_alias, name_alias, name_orig = t_a, name_a, name_b

        msg = (
            f'Expected `numpy.{name_alias}` to be an alias for '
            f'numpy.{name_orig}`, but it is `{t_alias.__name__}` instead.'
        )
        raise UnexpectedAliasError(msg)


check_aliases(ct, _ALIASES_CT)
check_aliases(np, _ALIASES_NP)
