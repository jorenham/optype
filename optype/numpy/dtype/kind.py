# pyright: reportRedeclaration=false
"""
Type aliases for `numpy.dtype.kind`.

See Also:
    - https://numpy.org/devdocs/reference/arrays.scalars.html
    - https://numpy.org/devdocs/reference/generated/numpy.dtype.kind.html

Todo:
    - 'T' for `np.dtypes.StringDType` on numpy>=2
"""

from typing import Literal as _L, TypeAlias as _T  # noqa: N814

from numpy.version import short_version as _np_version


_NP_V1 = _np_version < '2'

SignedInteger: _T = _L['i']
UnsignedInteger: _T = _L['u']
Floating: _T = _L['f']
ComplexFloating: _T = _L['c']

Integer: _T = SignedInteger | UnsignedInteger
Inexact: _T = Floating | ComplexFloating
Number: _T = Integer | Inexact

Str: _T = _L['U']
Bytes: _T = _L['S']
Character: _T = Str | Bytes

Void: _T = _L['V']
Flexible: _T = Void | Character

Bool: _T = _L['b']
Datetime64: _T = _L['M']
Timedelta64: _T = _L['m']
Object: _T = _L['O']

Generic: _T = Bool | Datetime64 | Timedelta64 | Object | Number | Flexible
