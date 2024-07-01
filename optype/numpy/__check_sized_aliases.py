"""Panic in case of unexpected `numpy` sized C-type aliases."""
import numpy as np


_ALIASES = (
    ('uint8', 'ubyte'),
    ('uint16', 'ushort'),
    ('uint32', 'uintc'),

    ('int8', 'byte'),
    ('int16', 'short'),
    ('int32', 'intc'),

    ('float16', 'half'),
    ('float32', 'single'),
    ('float64', 'double'),

    ('complex64', 'csingle'),
    ('complex128', 'cdouble'),
)
_ERROR_MSG = (
    'Expected: {0} is {1}. got {2.__name__}'
    'Please submit an issue at https://github.com/jorenham/optype/issues.'
)


class UnexpectedAliasError(OSError): ...


def check_aliases() -> None:
    for name_sized, name_c in _ALIASES:
        t_sized = getattr(np, name_sized)
        t_c = getattr(np, name_c)
        if t_sized is t_c:
            continue

        if t_sized.__name__ == name_sized:
            t_alias, name_alias, name_orig = t_c, name_c, name_sized
        else:
            assert t_c.__name__ == name_c
            t_alias, name_alias, name_orig = t_sized, name_sized, name_c

        msg = (
            f'Expected `numpy.{name_alias}` to be an alias for '
            f'numpy.{name_orig}`, but it is `{t_alias.__name__}` instead.'
        )
        raise UnexpectedAliasError(msg)


check_aliases()
