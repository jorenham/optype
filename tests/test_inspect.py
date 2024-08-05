# ruff: noqa: ANN205, ANN206
from __future__ import annotations

import sys
import typing as tp
from inspect import getattr_static

import pytest
import typing_extensions as tpx


if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType


import optype as opt


FalsyBool = tp.Literal[False]
FalsyInt: tp.TypeAlias = tp.Annotated[tp.Literal[0], (int, False)]
FalsyIntCo: tp.TypeAlias = FalsyBool | FalsyInt
# this is equivalent to `type FalsyStr = ...` on `python>=3.12`
FalsyStr = TypeAliasType('FalsyStr', tp.Literal['', b''])
Falsy = TypeAliasType('Falsy', tp.Literal[None, FalsyIntCo] | FalsyStr)

_T = tp.TypeVar('_T')
_T_tpx = tp.TypeVar('_T_tpx')

_Pss = tp.ParamSpec('_Pss')
_T_co = tp.TypeVar('_T_co', covariant=True)


class GenericTP(tp.Generic[_T]): ...


class GenericTPX(tp.Generic[_T_tpx]): ...


@tpx.runtime_checkable
class CanInit(tpx.Protocol[_Pss]):
    # on Python 3.10 this requires `typing_extensions.Protocol` to work
    def __init__(self, *args: _Pss.args, **kwargs: _Pss.kwargs) -> None: ...


@tpx.runtime_checkable
class CanNew(tpx.Protocol[_Pss, _T_co]):
    def __new__(cls, *args: _Pss.args, **kwargs: _Pss.kwargs) -> _T_co: ...


class Proto(tp.Protocol): ...


class ProtoX(tpx.Protocol): ...


@tp.runtime_checkable
class ProtoRuntime(tp.Protocol): ...


@tpx.runtime_checkable
class ProtoRuntimeX(tpx.Protocol): ...


@tp.final
class ProtoFinal(tp.Protocol): ...


@tpx.final
class ProtoFinalX(tpx.Protocol): ...


class FinalMembers:
    @property
    def p(self): pass
    @property
    @tp.final
    def p_final(self): pass
    @tpx.final
    def p_final_x(self): pass

    def f(self): pass
    @tp.final
    def f_final(self): pass
    @tpx.final
    def f_final_x(self): pass

    @classmethod
    def cf(cls): pass
    @classmethod
    @tp.final
    def cf_final1(cls): pass
    @classmethod
    @tpx.final
    def cf_final1_x(cls): pass
    @tp.final
    @classmethod
    def cf_final2(cls): pass
    @tpx.final
    @classmethod
    def cf_final2_x(cls): pass

    @staticmethod
    def sf(): pass
    @staticmethod
    @tp.final
    def sf_final1(): pass
    @staticmethod
    @tpx.final
    def sf_final1_x(): pass
    @tp.final
    @staticmethod
    def sf_final2(): pass
    @tpx.final
    @staticmethod
    def sf_final2_x(): pass


def test_get_args_literals():
    assert opt.inspect.get_args(FalsyBool) == (False,)
    assert opt.inspect.get_args(FalsyInt) == (0,)
    assert opt.inspect.get_args(FalsyIntCo) == (False, 0)
    assert opt.inspect.get_args(FalsyStr) == ('', b'')
    assert opt.inspect.get_args(Falsy) == (None, False, 0, '', b'')


@pytest.mark.parametrize('origin', [type, list, tuple, GenericTP, GenericTPX])
def test_get_args_generic(origin: tp.Any):
    assert opt.inspect.get_args(origin[FalsyBool]) == (FalsyBool,)
    assert opt.inspect.get_args(origin[FalsyInt]) == (FalsyInt,)
    assert opt.inspect.get_args(origin[FalsyIntCo]) == (FalsyIntCo,)
    assert opt.inspect.get_args(origin[FalsyStr]) == (FalsyStr,)
    assert opt.inspect.get_args(origin[Falsy]) == (Falsy,)


def test_get_protocol_members():
    assert opt.inspect.get_protocol_members(opt.CanAdd) == {'__add__'}
    assert opt.inspect.get_protocol_members(opt.CanPow) == {'__pow__'}
    assert opt.inspect.get_protocol_members(opt.CanHash) == {'__hash__'}
    assert opt.inspect.get_protocol_members(opt.CanEq) == {'__eq__'}
    assert opt.inspect.get_protocol_members(opt.CanGetMissing) == {
        '__getitem__',
        '__missing__',
    }
    assert opt.inspect.get_protocol_members(opt.CanWith) == {
        '__enter__',
        '__exit__',
    }

    assert opt.inspect.get_protocol_members(opt.HasName) == {'__name__'}
    assert opt.inspect.get_protocol_members(opt.HasNames) == {
        '__name__',
        '__qualname__',
    }
    assert opt.inspect.get_protocol_members(opt.HasClass) == {'__class__'}
    assert opt.inspect.get_protocol_members(opt.HasDict) == {'__dict__'}
    assert opt.inspect.get_protocol_members(opt.HasSlots) == {'__slots__'}
    assert opt.inspect.get_protocol_members(opt.HasAnnotations) == {
        '__annotations__',
    }

    assert opt.inspect.get_protocol_members(CanInit) == {'__init__'}
    assert opt.inspect.get_protocol_members(CanNew) == {'__new__'}


def test_get_protocols():
    import collections.abc  # noqa: PLC0415
    import types  # noqa: PLC0415

    assert not opt.inspect.get_protocols(collections.abc)
    assert not opt.inspect.get_protocols(types)
    # ... hence optype

    protocols_tp = opt.inspect.get_protocols(tp)
    assert protocols_tp
    assert opt.inspect.get_protocols(tp, private=True) >= protocols_tp

    protocols_tpx = opt.inspect.get_protocols(tpx)
    assert protocols_tpx
    assert opt.inspect.get_protocols(tpx, private=True) >= protocols_tpx


def test_type_is_final():
    assert not opt.inspect.is_final(opt.CanAdd)
    assert opt.inspect.is_final(opt.DoesAdd)

    assert not opt.inspect.is_final(Proto)
    assert not opt.inspect.is_final(ProtoX)
    assert not opt.inspect.is_final(ProtoRuntime)
    if sys.version_info >= (3, 11):
        assert opt.inspect.is_final(ProtoFinal)
    assert opt.inspect.is_final(ProtoFinalX)


def test_property_is_final():
    assert not opt.inspect.is_final(FinalMembers.p)
    if sys.version_info >= (3, 11):
        assert opt.inspect.is_final(FinalMembers.p_final)
    assert opt.inspect.is_final(FinalMembers.p_final_x)


def test_method_is_final():
    assert not opt.inspect.is_final(FinalMembers.f)
    if sys.version_info >= (3, 11):
        assert opt.inspect.is_final(FinalMembers.f_final)
    assert opt.inspect.is_final(FinalMembers.f_final_x)


def test_classmethod_is_final():
    assert not opt.inspect.is_final(FinalMembers.cf)
    if sys.version_info >= (3, 11):
        assert opt.inspect.is_final(getattr_static(FinalMembers, 'cf_final1'))
        assert opt.inspect.is_final(getattr_static(FinalMembers, 'cf_final2'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'cf_final1_x'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'cf_final2_x'))


def test_staticmethod_is_final():
    assert not opt.inspect.is_final(FinalMembers.sf)
    if sys.version_info >= (3, 11):
        assert opt.inspect.is_final(getattr_static(FinalMembers, 'sf_final1'))
        assert opt.inspect.is_final(getattr_static(FinalMembers, 'sf_final2'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'sf_final1_x'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'sf_final2_x'))


@pytest.mark.parametrize('origin', [type, list, tuple, GenericTP, GenericTPX])
def test_is_generic_alias(origin: tp.Any):
    assert not opt.inspect.is_generic_alias(origin)

    assert opt.inspect.is_generic_alias(origin[None])
    Alias = TypeAliasType('Alias', origin[None])  # noqa: N806
    assert opt.inspect.is_generic_alias(Alias)
    assert opt.inspect.is_generic_alias(tp.Annotated[origin[None], None])
    assert opt.inspect.is_generic_alias(tpx.Annotated[origin[None], None])

    assert not opt.inspect.is_generic_alias(origin[None] | None)


def test_is_iterable():
    assert opt.inspect.is_iterable([])
    assert opt.inspect.is_iterable(())
    assert opt.inspect.is_iterable('')
    assert opt.inspect.is_iterable(b'')
    assert opt.inspect.is_iterable(range(2))
    assert opt.inspect.is_iterable(i for i in range(2))


def test_is_runtime_protocol():
    assert opt.inspect.is_runtime_protocol(opt.CanAdd)
    assert not opt.inspect.is_runtime_protocol(opt.DoesAdd)

    assert not opt.inspect.is_runtime_protocol(Proto)
    assert not opt.inspect.is_runtime_protocol(ProtoX)
    assert opt.inspect.is_runtime_protocol(ProtoRuntime)
    assert opt.inspect.is_runtime_protocol(ProtoRuntimeX)
    assert not opt.inspect.is_runtime_protocol(ProtoFinal)
    assert not opt.inspect.is_runtime_protocol(ProtoFinalX)


@pytest.mark.parametrize('origin', [int, tp.Literal[True], Proto, ProtoX])
def test_is_union_type(origin: tp.Any):
    assert opt.inspect.is_union_type(origin | None)
    Alias = TypeAliasType('Alias', origin | None)  # noqa: N806
    assert opt.inspect.is_union_type(Alias)
    assert opt.inspect.is_union_type(tp.Annotated[origin | None, None])
    assert opt.inspect.is_union_type(tp.Annotated[origin, None] | None)

    assert not opt.inspect.is_union_type(origin)
    assert not opt.inspect.is_union_type(origin | origin)
