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

_T_tp = tp.TypeVar('_T_tp')
_T_tpx = tp.TypeVar('_T_tpx')
_Pss = tpx.ParamSpec('_Pss')
_T_co = tpx.TypeVar('_T_co', covariant=True)


class GenericTP(tp.Generic[_T_tp]): ...


class GenericTPX(tp.Generic[_T_tpx]): ...


@tpx.runtime_checkable
class CanInit(tp.Protocol[_Pss]):
    def __init__(self, *args: _Pss.args, **kwargs: _Pss.kwargs) -> None: ...


@tpx.runtime_checkable
class CanNew(tp.Protocol[_Pss, _T_co]):
    def __new__(cls, *args: _Pss.args, **kwargs: _Pss.kwargs) -> _T_co: ...


class PTyp(tp.Protocol): ...


class PExt(tpx.Protocol): ...


@tp.runtime_checkable
class RuntimePTyp(tp.Protocol): ...


@tpx.runtime_checkable
class RuntimePExt(tpx.Protocol): ...


@tp.final
class FinalPTyp(tp.Protocol): ...


@tpx.final
class FinalPExt(tpx.Protocol): ...


class FinalMembers:
    @property
    def p(self): pass
    @property
    @tp.final
    def p_final_typ(self): pass
    @tpx.final
    def p_final_ext(self): pass

    def f(self): pass
    @tp.final
    def f_final_typ(self): pass
    @tpx.final
    def f_final_ext(self): pass

    @classmethod
    def cf(cls): pass
    @classmethod
    @tp.final
    def cf_final1_typ(cls): pass
    @classmethod
    @tpx.final
    def cf_final1_ext(cls): pass
    @tp.final
    @classmethod
    def cf_final2_typ(cls): pass
    @tpx.final
    @classmethod
    def cf_final2_ext(cls): pass

    @staticmethod
    def sf(): pass
    @staticmethod
    @tp.final
    def sf_final1_typ(): pass
    @staticmethod
    @tpx.final
    def sf_final1_ext(): pass
    @tp.final
    @staticmethod
    def sf_final2_typ(): pass
    @tpx.final
    @staticmethod
    def sf_final2_ext(): pass


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

    assert protocols_tp <= protocols_tpx


def test_type_is_final():
    assert not opt.inspect.is_final(opt.CanAdd)
    assert opt.inspect.is_final(opt.DoesAdd)

    assert not opt.inspect.is_final(PTyp)
    assert not opt.inspect.is_final(PExt)
    assert not opt.inspect.is_final(RuntimePTyp)
    assert opt.inspect.is_final(FinalPTyp)
    assert opt.inspect.is_final(FinalPExt)


def test_property_is_final():
    assert not opt.inspect.is_final(FinalMembers.p)
    assert opt.inspect.is_final(FinalMembers.p_final_typ)
    assert opt.inspect.is_final(FinalMembers.p_final_ext)


def test_method_is_final():
    assert not opt.inspect.is_final(FinalMembers.f)
    assert opt.inspect.is_final(FinalMembers.f_final_typ)
    assert opt.inspect.is_final(FinalMembers.f_final_ext)


def test_classmethod_is_final():
    assert not opt.inspect.is_final(FinalMembers.cf)
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'cf_final1_typ'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'cf_final1_ext'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'cf_final2_typ'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'cf_final2_ext'))


def test_staticmethod_is_final():
    assert not opt.inspect.is_final(FinalMembers.sf)
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'sf_final1_typ'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'sf_final1_ext'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'sf_final2_typ'))
    assert opt.inspect.is_final(getattr_static(FinalMembers, 'sf_final2_ext'))


@pytest.mark.parametrize('origin', [type, list, tuple, GenericTP, GenericTPX])
def test_is_generic_alias(origin: tp.Any):
    assert not opt.inspect.is_generic_alias(origin)

    assert opt.inspect.is_generic_alias(origin[None])
    Alias = TypeAliasType('Alias', origin[None])  # noqa: N806
    assert opt.inspect.is_generic_alias(Alias)
    assert opt.inspect.is_generic_alias(tp.Annotated[origin[None], None])

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

    assert not opt.inspect.is_runtime_protocol(PTyp)
    assert not opt.inspect.is_runtime_protocol(PExt)
    assert opt.inspect.is_runtime_protocol(RuntimePTyp)
    assert opt.inspect.is_runtime_protocol(RuntimePExt)
    assert not opt.inspect.is_runtime_protocol(FinalPTyp)
    assert not opt.inspect.is_runtime_protocol(FinalPTyp)


# TODO: test `is_union_type`
