
"""
See:
    - https://numpy.org/devdocs/user/basics.types.html
    - https://numpy.org/devdocs/reference/arrays.scalars.html
    - https://numpy.org/devdocs/reference/arrays.dtypes.html
"""
from typing import (
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np


DCharGeneric: TypeAlias = Literal[
    '?',  # bool
    'B',  # ubyte / uint8
    'D',  # cdouble / complex128
    'F',  # csingle / complex64
    'G',  # clongdouble / complex192 or complex256
    'H',  # ushort / uint16
    'I',  # uintc / uint32
    'L',  # ulong / uint64
    'M',  # datetime64
    'N',  # uintp (numpy<2)
    'O',  # object
    'P',  # uintp (numpy>=2)
    'Q',  # ulonglong
    'S',  # bytes_
    'T',  # StringDType128
    'U',  # str
    'V',  # void
    'a',  # bytes (deprecated)
    'b',  # byte / int8
    'c',  # S1 / bytes8
    'd',  # double / float64
    'e',  # half / float16
    'f',  # single / float32
    'g',  # longdouble / float96 or float128
    'h',  # short / int16
    'i',  # intc / int32
    'l',  # long / int64
    'm',  # timedelta64
    'n',  # intp / int32 or int64 (numpy<2)
    'p',  # intp (numpy>=2)
    'q',  # longlong
]
"""https://numpy.org/devdocs/reference/generated/numpy.dtype.char.html"""

DKindGeneric: TypeAlias = Literal[
    'b',  # bool
    'u',  # unsignedinteger
    'i',  # integer
    'f',  # floating
    'c',  # complexfloating
    'm',  # timedelta
    'M',  # datetime
    'O',  # object
    'S',  # bytes
    'U',  # str
    'V',  # void
]
"""https://numpy.org/devdocs/reference/generated/numpy.dtype.kind.html"""

DNameObject: TypeAlias = Literal['object', 'object_']
DNameBool: TypeAlias = Literal['bool8', 'bool', 'bool_']

DNameVoid: TypeAlias = Literal['void', 'void0']
DNameStr: TypeAlias = Literal['str', 'str_', 'str0', 'unicode', 'unicode_']
DNameBytes: TypeAlias = Literal['bytes', 'bytes_', 'bytes0']
DNameCharacter: TypeAlias = DNameStr | DNameBytes
DNameFlexible: TypeAlias = DNameVoid | DNameCharacter

DNameInt8: TypeAlias = Literal['int8']
DNameInt16: TypeAlias = Literal['int16']
DNameInt32: TypeAlias = Literal['int32']
DNameInt64: TypeAlias = Literal['int64']
DNameByte: TypeAlias = Literal['byte']
DNameShort: TypeAlias = Literal['short']
DNameIntC: TypeAlias = Literal['intc']
DNameIntP: TypeAlias = Literal['intp', 'int0']
DNameLong: TypeAlias = Literal['long', 'int', 'int_']
DNameLongLong: TypeAlias = Literal['longlong']
DNameSignedInteger: TypeAlias = (
    DNameInt8
    | DNameInt16
    | DNameInt32
    | DNameInt64
    | DNameByte
    | DNameShort
    | DNameIntC
    | DNameIntP
    | DNameLong
    | DNameLongLong
)

DNameUInt8: TypeAlias = Literal['uint8']
DNameUInt16: TypeAlias = Literal['uint16']
DNameUInt32: TypeAlias = Literal['uint32']
DNameUInt64: TypeAlias = Literal['uint64']
DNameUByte: TypeAlias = Literal['ubyte']
DNameUShort: TypeAlias = Literal['ushort']
DNameUIntC: TypeAlias = Literal['uintc']
DNameUIntP: TypeAlias = Literal['uintp', 'uint0']
DNameULong: TypeAlias = Literal['ulong', 'uint']
DNameULongLong: TypeAlias = Literal['ulonglong']
DNameUnsignedInteger: TypeAlias = (
    DNameUInt8
    | DNameUInt16
    | DNameUInt32
    | DNameUInt64
    | DNameUByte
    | DNameUShort
    | DNameUIntC
    | DNameUIntP
    | DNameULong
    | DNameULongLong
)

DNameFloat16: TypeAlias = Literal['float16']
DNameFloat32: TypeAlias = Literal['float32']
DNameFloat64: TypeAlias = Literal['float64']
DNameHalf: TypeAlias = Literal['half']
DNameSingle: TypeAlias = Literal['single']
DNameDouble: TypeAlias = Literal['double', 'float', 'float_']
DNameLongDouble: TypeAlias = Literal['longdouble', 'longfloat']
DNameFloating: TypeAlias = (
    DNameFloat16
    | DNameFloat32
    | DNameFloat64
    | DNameHalf
    | DNameSingle
    | DNameDouble
    | DNameLongDouble
)
DNameComplex64: TypeAlias = Literal['complex64']
DNameComplex128: TypeAlias = Literal['complex128']
DNameCSingle: TypeAlias = Literal['csingle', 'singlecomplex']
DNameCDouble: TypeAlias = Literal['cdouble', 'complex', 'complex_', 'cfloat']
DNameCLongDouble: TypeAlias = Literal[
    'clongdouble',
    'clongfloat',
    'longcomplex',
]
DNameComplexFloating: TypeAlias = (
    DNameComplex64
    | DNameComplex128
    | DNameCSingle
    | DNameCDouble
    | DNameCLongDouble
)

DNameInteger: TypeAlias = DNameSignedInteger | DNameUnsignedInteger
DNameInexact: TypeAlias = DNameFloating | DNameComplexFloating
DNameNumber: TypeAlias = DNameInteger | DNameInexact
DNameGeneric: TypeAlias = DNameObject | DNameBool | DNameNumber | DNameFlexible
"""https://numpy.org/doc/stable/reference/generated/numpy.dtype.name.html"""


_T = TypeVar('_T', bound=np.generic)
_T_co = TypeVar('_T_co', bound=np.generic, covariant=True)


@runtime_checkable
class HasDType(Protocol[_T_co]):
    @property
    def dtype(self) -> np.dtype[_T_co]: ...


# subset of `npt.DTypeLike`, with type parameter `T: np.generic = np.generic`
# useful for overloaded methods with a `dtype` parameter
SomeDType: TypeAlias = np.dtype[_T] | HasDType[_T] | _T
