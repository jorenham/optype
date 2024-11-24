# mypy: disable-error-code="unreachable"
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


import optype as o


FalsyBool = tp.Literal[False]
FalsyInt: tp.TypeAlias = tp.Annotated[tp.Literal[0], (int, False)]
FalsyIntCo: tp.TypeAlias = FalsyBool | FalsyInt
# this is equivalent to `type FalsyStr = ...` on `python>=3.12`
FalsyStr = TypeAliasType("FalsyStr", tp.Literal["", b""])
Falsy = TypeAliasType("Falsy", tp.Literal[None, FalsyIntCo] | FalsyStr)  # noqa: PYI061

_T = tp.TypeVar("_T")
_T_tpx = tp.TypeVar("_T_tpx")

_Pss = tp.ParamSpec("_Pss")
_T_co = tp.TypeVar("_T_co", covariant=True)


class GenericTP(tp.Generic[_T]): ...


class GenericTPX(tp.Generic[_T_tpx]): ...


@tpx.runtime_checkable
class CanInit(tpx.Protocol[_Pss]):
    # on Python 3.10 this requires `typing_extensions.Protocol` to work
    def __init__(self, *args: _Pss.args, **kwargs: _Pss.kwargs) -> None: ...


@tpx.runtime_checkable
class CanNew(tpx.Protocol[_Pss, _T_co]):
    def __new__(cls, *args: _Pss.args, **kwargs: _Pss.kwargs) -> _T_co: ...  # type: ignore[misc]


class ProtoOverload(tpx.Protocol):
    @tpx.overload
    def method(self) -> int: ...
    @tpx.overload
    def method(self, x: object, /) -> str: ...


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
    def p(self) -> object:
        pass

    @property
    @tp.final
    def p_final(self) -> object:
        pass

    @tpx.final
    def p_final_x(self) -> object:
        pass

    def f(self) -> object:
        pass

    @tp.final
    def f_final(self) -> object:
        pass

    @tpx.final
    def f_final_x(self) -> object:
        pass

    @classmethod
    def cf(cls) -> object:
        pass

    @classmethod
    @tp.final
    def cf_final1(cls) -> object:
        pass

    @classmethod
    @tpx.final
    def cf_final1_x(cls) -> object:
        pass

    @tp.final
    @classmethod
    def cf_final2(cls) -> object:
        pass

    @tpx.final
    @classmethod
    def cf_final2_x(cls) -> object:
        pass

    @staticmethod
    def sf() -> object:
        pass

    @staticmethod
    @tp.final
    def sf_final1() -> object:
        pass

    @staticmethod
    @tpx.final
    def sf_final1_x() -> object:
        pass

    @tp.final
    @staticmethod
    def sf_final2() -> object:
        pass

    @tpx.final
    @staticmethod
    def sf_final2_x() -> object:
        pass


def test_get_args_literals() -> None:
    assert o.inspect.get_args(FalsyBool) == (False,)
    assert o.inspect.get_args(FalsyInt) == (0,)
    assert o.inspect.get_args(FalsyIntCo) == (False, 0)
    assert o.inspect.get_args(FalsyStr) == ("", b"")
    assert o.inspect.get_args(Falsy) == (None, False, 0, "", b"")


@pytest.mark.parametrize("origin", [type, list, tuple, GenericTP, GenericTPX])
def test_get_args_generic(origin: o.types.GenericType) -> None:
    assert o.inspect.get_args(origin[FalsyBool]) == (FalsyBool,)
    assert o.inspect.get_args(origin[FalsyInt]) == (FalsyInt,)
    assert o.inspect.get_args(origin[FalsyIntCo]) == (FalsyIntCo,)
    assert o.inspect.get_args(origin[FalsyStr]) == (FalsyStr,)
    assert o.inspect.get_args(origin[Falsy]) == (Falsy,)


def test_get_protocol_members() -> None:
    assert o.inspect.get_protocol_members(o.CanAdd) == {"__add__"}
    assert o.inspect.get_protocol_members(o.CanPow) == {"__pow__"}
    assert o.inspect.get_protocol_members(o.CanHash) == {"__hash__"}
    assert o.inspect.get_protocol_members(o.CanEq) == {"__eq__"}
    assert o.inspect.get_protocol_members(o.CanGetMissing) == {
        "__getitem__",
        "__missing__",
    }
    assert o.inspect.get_protocol_members(o.CanWith) == {"__enter__", "__exit__"}

    assert o.inspect.get_protocol_members(o.HasName) == {"__name__"}
    assert o.inspect.get_protocol_members(o.HasNames) == {
        "__name__",
        "__qualname__",
    }
    assert o.inspect.get_protocol_members(o.HasClass) == {"__class__"}
    assert o.inspect.get_protocol_members(o.HasDict) == {"__dict__"}
    assert o.inspect.get_protocol_members(o.HasSlots) == {"__slots__"}
    assert o.inspect.get_protocol_members(o.HasAnnotations) == {"__annotations__"}

    assert o.inspect.get_protocol_members(CanInit) == {"__init__"}
    assert o.inspect.get_protocol_members(CanNew) == {"__new__"}

    assert o.inspect.get_protocol_members(ProtoOverload) == {"method"}


def test_get_protocols() -> None:
    import collections.abc  # noqa: PLC0415
    import types  # noqa: PLC0415

    assert not o.inspect.get_protocols(collections.abc)
    assert not o.inspect.get_protocols(types)
    # ... hence optype

    protocols_tp = o.inspect.get_protocols(tp)
    assert protocols_tp
    assert o.inspect.get_protocols(tp, private=True) >= protocols_tp

    protocols_tpx = o.inspect.get_protocols(tpx)
    assert protocols_tpx
    assert o.inspect.get_protocols(tpx, private=True) >= protocols_tpx


def test_type_is_final() -> None:
    assert not o.inspect.is_final(Proto)
    assert not o.inspect.is_final(ProtoX)
    assert not o.inspect.is_final(ProtoRuntime)
    if sys.version_info >= (3, 11):
        assert o.inspect.is_final(ProtoFinal)
    assert o.inspect.is_final(ProtoFinalX)


def test_property_is_final() -> None:
    assert not o.inspect.is_final(FinalMembers.p)
    if sys.version_info >= (3, 11):
        assert o.inspect.is_final(FinalMembers.p_final)
    assert o.inspect.is_final(FinalMembers.p_final_x)


def test_method_is_final() -> None:
    assert not o.inspect.is_final(FinalMembers.f)
    if sys.version_info >= (3, 11):
        assert o.inspect.is_final(FinalMembers.f_final)
    assert o.inspect.is_final(FinalMembers.f_final_x)


def test_classmethod_is_final() -> None:
    assert not o.inspect.is_final(FinalMembers.cf)
    if sys.version_info >= (3, 11):
        assert o.inspect.is_final(getattr_static(FinalMembers, "cf_final1"))
        assert o.inspect.is_final(getattr_static(FinalMembers, "cf_final2"))

    assert o.inspect.is_final(
        tp.cast(  # type: ignore[no-any-explicit]
            "classmethod[FinalMembers, ..., object]",
            getattr_static(FinalMembers, "cf_final1_x"),
        ),
    )
    assert o.inspect.is_final(
        tp.cast(  # type: ignore[no-any-explicit]
            "classmethod[FinalMembers, ..., object]",
            getattr_static(FinalMembers, "cf_final2_x"),
        ),
    )


def test_staticmethod_is_final() -> None:
    assert not o.inspect.is_final(FinalMembers.sf)
    if sys.version_info >= (3, 11):
        assert o.inspect.is_final(getattr_static(FinalMembers, "sf_final1"))
        assert o.inspect.is_final(getattr_static(FinalMembers, "sf_final2"))

    assert o.inspect.is_final(
        tp.cast(  # type: ignore[no-any-explicit]
            "staticmethod[..., object]",
            getattr_static(FinalMembers, "sf_final1_x"),
        ),
    )
    assert o.inspect.is_final(
        tp.cast(  # type: ignore[no-any-explicit]
            "staticmethod[..., object]",
            getattr_static(FinalMembers, "sf_final2_x"),
        ),
    )


@pytest.mark.parametrize("origin", [type, list, tuple, GenericTP, GenericTPX])
def test_is_generic_alias(origin: o.types.GenericType) -> None:
    assert not o.inspect.is_generic_alias(origin)

    assert o.inspect.is_generic_alias(origin[None])
    Alias = TypeAliasType("Alias", origin[None])  # type: ignore[valid-type]  # noqa: N806
    assert o.inspect.is_generic_alias(Alias)
    assert o.inspect.is_generic_alias(tp.Annotated[origin[None], None])
    assert o.inspect.is_generic_alias(tpx.Annotated[origin[None], None])

    assert not o.inspect.is_generic_alias(origin[None] | None)


def test_is_iterable() -> None:
    assert o.inspect.is_iterable([])
    assert o.inspect.is_iterable(())
    assert o.inspect.is_iterable("")
    assert o.inspect.is_iterable(b"")
    assert o.inspect.is_iterable(range(2))
    assert o.inspect.is_iterable(i for i in range(2))


def test_is_runtime_protocol() -> None:
    assert o.inspect.is_runtime_protocol(o.CanAdd)
    assert not o.inspect.is_runtime_protocol(o.DoesAdd)

    assert not o.inspect.is_runtime_protocol(Proto)
    assert not o.inspect.is_runtime_protocol(ProtoX)
    assert o.inspect.is_runtime_protocol(ProtoRuntime)
    assert o.inspect.is_runtime_protocol(ProtoRuntimeX)
    assert not o.inspect.is_runtime_protocol(ProtoFinal)
    assert not o.inspect.is_runtime_protocol(ProtoFinalX)


@pytest.mark.parametrize("origin", [int, tp.Literal[True], Proto, ProtoX])
def test_is_union_type(origin: type) -> None:
    assert o.inspect.is_union_type(origin | None)
    Alias: TypeAliasType = TypeAliasType("Alias", origin | None)  # noqa: N806  # pyright: ignore[reportGeneralTypeIssues]
    assert o.inspect.is_union_type(Alias)
    assert o.inspect.is_union_type(tp.Annotated[origin | None, None])
    assert o.inspect.is_union_type(tp.Annotated[origin, None] | None)

    assert not o.inspect.is_union_type(origin)
    assert not o.inspect.is_union_type(origin | origin)
