# mypy: disable-error-code="unreachable"
import sys
import typing as tp
import typing_extensions as tpx
from inspect import getattr_static

if sys.version_info >= (3, 13):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

import pytest

import optype as op

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
    assert op.inspect.get_args(FalsyBool) == (False,)
    assert op.inspect.get_args(FalsyInt) == (0,)
    assert op.inspect.get_args(FalsyIntCo) == (False, 0)
    assert op.inspect.get_args(FalsyStr) == ("", b"")
    assert op.inspect.get_args(Falsy) == (None, False, 0, "", b"")


@pytest.mark.parametrize("origin", [type, list, tuple, GenericTP, GenericTPX])
def test_get_args_generic(origin: op.types.GenericType) -> None:
    assert op.inspect.get_args(origin[FalsyBool]) == (FalsyBool,)
    assert op.inspect.get_args(origin[FalsyInt]) == (FalsyInt,)
    assert op.inspect.get_args(origin[FalsyIntCo]) == (FalsyIntCo,)
    assert op.inspect.get_args(origin[FalsyStr]) == (FalsyStr,)
    assert op.inspect.get_args(origin[Falsy]) == (Falsy,)


def test_get_protocol_members() -> None:
    assert op.inspect.get_protocol_members(op.CanAdd) == {"__add__"}
    assert op.inspect.get_protocol_members(op.CanPow) == {"__pow__"}
    assert op.inspect.get_protocol_members(op.CanHash) == {"__hash__"}
    assert op.inspect.get_protocol_members(op.CanEq) == {"__eq__"}
    assert op.inspect.get_protocol_members(op.CanGetMissing) == {
        "__getitem__",
        "__missing__",
    }
    assert op.inspect.get_protocol_members(op.CanWith) == {"__enter__", "__exit__"}

    assert op.inspect.get_protocol_members(op.HasName) == {"__name__"}
    assert op.inspect.get_protocol_members(op.HasNames) == {
        "__name__",
        "__qualname__",
    }
    assert op.inspect.get_protocol_members(op.HasClass) == {"__class__"}
    assert op.inspect.get_protocol_members(op.HasDict) == {"__dict__"}
    assert op.inspect.get_protocol_members(op.HasSlots) == {"__slots__"}
    assert op.inspect.get_protocol_members(op.HasAnnotations) == {"__annotations__"}

    assert op.inspect.get_protocol_members(CanInit) == {"__init__"}
    assert op.inspect.get_protocol_members(CanNew) == {"__new__"}

    assert op.inspect.get_protocol_members(ProtoOverload) == {"method"}


def test_get_protocols() -> None:
    import collections.abc  # noqa: PLC0415
    import types  # noqa: PLC0415

    assert not op.inspect.get_protocols(collections.abc)
    assert not op.inspect.get_protocols(types)
    # ... hence optype

    protocols_tp = op.inspect.get_protocols(tp)
    assert protocols_tp
    assert op.inspect.get_protocols(tp, private=True) >= protocols_tp

    protocols_tpx = op.inspect.get_protocols(tpx)
    assert protocols_tpx
    assert op.inspect.get_protocols(tpx, private=True) >= protocols_tpx


def test_type_is_final() -> None:
    assert not op.inspect.is_final(Proto)
    assert not op.inspect.is_final(ProtoX)
    assert not op.inspect.is_final(ProtoRuntime)
    assert op.inspect.is_final(ProtoFinal)
    assert op.inspect.is_final(ProtoFinalX)


def test_property_is_final() -> None:
    assert not op.inspect.is_final(FinalMembers.p)
    assert op.inspect.is_final(FinalMembers.p_final)
    assert op.inspect.is_final(FinalMembers.p_final_x)


def test_method_is_final() -> None:
    assert not op.inspect.is_final(FinalMembers.f)
    assert op.inspect.is_final(FinalMembers.f_final)
    assert op.inspect.is_final(FinalMembers.f_final_x)


def test_classmethod_is_final() -> None:
    assert not op.inspect.is_final(FinalMembers.cf)
    assert op.inspect.is_final(getattr_static(FinalMembers, "cf_final1"))
    assert op.inspect.is_final(getattr_static(FinalMembers, "cf_final2"))

    assert op.inspect.is_final(
        tp.cast(
            "classmethod[FinalMembers, ..., object]",
            getattr_static(FinalMembers, "cf_final1_x"),
        ),
    )
    assert op.inspect.is_final(
        tp.cast(
            "classmethod[FinalMembers, ..., object]",
            getattr_static(FinalMembers, "cf_final2_x"),
        ),
    )


def test_staticmethod_is_final() -> None:
    assert not op.inspect.is_final(FinalMembers.sf)
    assert op.inspect.is_final(getattr_static(FinalMembers, "sf_final1"))
    assert op.inspect.is_final(getattr_static(FinalMembers, "sf_final2"))

    assert op.inspect.is_final(
        tp.cast(
            "staticmethod[..., object]",
            getattr_static(FinalMembers, "sf_final1_x"),
        ),
    )
    assert op.inspect.is_final(
        tp.cast(
            "staticmethod[..., object]",
            getattr_static(FinalMembers, "sf_final2_x"),
        ),
    )


@pytest.mark.parametrize("origin", [type, list, tuple, GenericTP, GenericTPX])
def test_is_generic_alias(origin: op.types.GenericType) -> None:
    assert not op.inspect.is_generic_alias(origin)

    assert op.inspect.is_generic_alias(origin[None])
    Alias = TypeAliasType("Alias", origin[None])  # type: ignore[valid-type]  # noqa: N806
    assert op.inspect.is_generic_alias(Alias)
    assert op.inspect.is_generic_alias(tp.Annotated[origin[None], None])
    assert op.inspect.is_generic_alias(tpx.Annotated[origin[None], None])

    assert not op.inspect.is_generic_alias(origin[None] | None)


def test_is_iterable() -> None:
    assert op.inspect.is_iterable([])
    assert op.inspect.is_iterable(())
    assert op.inspect.is_iterable("")
    assert op.inspect.is_iterable(b"")
    assert op.inspect.is_iterable(range(2))
    assert op.inspect.is_iterable(i for i in range(2))


def test_is_runtime_protocol() -> None:
    assert op.inspect.is_runtime_protocol(op.CanAdd)
    assert not op.inspect.is_runtime_protocol(op.DoesAdd)

    assert not op.inspect.is_runtime_protocol(Proto)
    assert not op.inspect.is_runtime_protocol(ProtoX)
    assert op.inspect.is_runtime_protocol(ProtoRuntime)
    assert op.inspect.is_runtime_protocol(ProtoRuntimeX)
    assert not op.inspect.is_runtime_protocol(ProtoFinal)
    assert not op.inspect.is_runtime_protocol(ProtoFinalX)


@pytest.mark.parametrize("origin", [int, tp.Literal[True], Proto, ProtoX])
def test_is_union_type(origin: type) -> None:
    assert op.inspect.is_union_type(origin | None)
    Alias: TypeAliasType = TypeAliasType("Alias", origin | None)  # noqa: N806  # pyright: ignore[reportGeneralTypeIssues]
    assert op.inspect.is_union_type(Alias)
    assert op.inspect.is_union_type(tp.Annotated[origin | None, None])
    assert op.inspect.is_union_type(tp.Annotated[origin, None] | None)

    assert not op.inspect.is_union_type(origin)
    assert not op.inspect.is_union_type(origin | origin)
