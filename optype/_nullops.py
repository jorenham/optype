"""
Interfaces for the "nullary" ops (w.r.t. arity of the type parameter).

Note that this might not seem DRY, since many of the protocols are already in
`typing.Supports`. But the problem with those is, that they are also
metaclasses, and that I (@jorenham) apparently am turing into a typing-purist.
"""
import types as _ts
import typing as _tp

# type conversion


@_tp.runtime_checkable
class CanBool(_tp.Protocol):
    """
    `bool(self)`
    """
    def __bool__(self) -> bool: ...

@_tp.runtime_checkable
class CanInt(_tp.Protocol):
    """
    `int(self)`
    """
    def __int__(self) -> int: ...

@_tp.runtime_checkable
class CanFloat(_tp.Protocol):
    """
    `float(self)`
    """
    def __float__(self) -> float: ...

@_tp.runtime_checkable
class CanComplex(_tp.Protocol):
    """
    `complex(self)`
    """
    def __complex__(self) -> float: ...

@_tp.runtime_checkable
class CanBytes(_tp.Protocol):
    """
    `bytes(self)`
    """
    def __bytes__(self) -> str: ...

@_tp.runtime_checkable
class CanStr(_tp.Protocol):
    """
    `str(self)`
    """
    def __str__(self) -> str: ...


# display methods

@_tp.runtime_checkable
class CanRepr(_tp.Protocol):
    """
    `repr(self)`
    """
    def __repr__(self) -> str: ...


# size methods

@_tp.runtime_checkable
class CanLen(_tp.Protocol):
    """
    `len(self)`

    some notes:
    - must be non-negative.
    - (cpython) must not exceed `sys.maxsize`
    - (cpython) without `__bool__` cpython will use `bool(len(self))` instead
    """
    def __len__(self) -> int: ...

@_tp.runtime_checkable
class CanLenHint(_tp.Protocol):
    """
    - approximation of `len(self)`
    - purely for optimization purposes
    - must be `>=0` or `NotImplemented`
    """
    def __len__(self) -> int | _ts.NotImplementedType: ...


# fingerprinting

@_tp.runtime_checkable
class CanHash(_tp.Protocol):
    """
    `hash(self)`
    """
    def __hash__(self) -> int: ...

@_tp.runtime_checkable
class CanIndex(_tp.Protocol):
    """
    `hash(self)`
    """
    def __index__(self) -> int: ...
