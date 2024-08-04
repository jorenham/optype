# ruff: noqa: ANN205, ANN206
from __future__ import annotations

import sys
import typing as tp


if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

from inspect import getattr_static

import typing_extensions as tpx

import optype as opt


FalsyBool = tp.Literal[False]
FalsyInt: tp.TypeAlias = tp.Annotated[tp.Literal[0], (int, False)]
FalsyIntCo: tp.TypeAlias = FalsyBool | FalsyInt
# this is equivalent to `type FalsyStr = ...` on `python>=3.12`
FalsyStr = TypeAliasType('FalsyStr', tp.Literal['', b''])
Falsy = TypeAliasType('Falsy', tp.Literal[None, FalsyIntCo] | FalsyStr)


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


def test_get_args():
    assert opt.inspect.get_args(FalsyBool) == (False,)
    assert opt.inspect.get_args(FalsyInt) == (0,)
    assert opt.inspect.get_args(FalsyIntCo) == (False, 0)
    assert opt.inspect.get_args(FalsyStr) == ('', b'')
    assert opt.inspect.get_args(Falsy) == (None, False, 0, '', b'')


# TODO: test `get_protocol_members`
# TODO: test `get_protocols`


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


# TODO: test `is_generic_alias`


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
