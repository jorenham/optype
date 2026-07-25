"""Assign type-parameter names to the spies of one exploration."""

from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass, field, replace

# `from . import _ir` would re-enter this package
import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]
from ._analyze import (
    all_packed,
    analyze,
    group_traces,
    representatives,
    return_spies,
)
from ._spy import _SpyObject, _Traces
from ._values import _Rec, _RecVar, _walk, fn_spies

__all__ = ("TYPEVAR_TUPLE_NAME", "_Naming", "build")

TYPEVAR_TUPLE_NAME = "Ts"  # the PEP 646 typevar-tuple binder, used as `*Ts`


def _result_tyvar(index: int) -> str:
    """The `index`-th return typevar name: `R`, `R2`, `R3`, ..."""
    return "R" if not index else f"R{index + 1}"


@dataclass(frozen=True, slots=True)
class _Naming:
    """Which spies get a type parameter, and under what name."""

    reps: Mapping[int, int]  # spy id -> representative id
    tyvars: Mapping[int, str]  # spy id -> type parameter name
    named: Mapping[int, str]  # representative id -> name
    param_spies: Sequence[_SpyObject]
    declared_spies: Sequence[_SpyObject]
    result_spies: Sequence[_SpyObject]
    rec_tyvars: Mapping[_RecVar, str]
    rec_body: Mapping[_RecVar, object]
    group_traces: _Traces
    vartuple: bool  # whether the `*args` spy renders as a `*Ts` typevar tuple

    def pool(self, vartuple_id: int | None) -> dict[str, int]:
        """The declared `name -> representative` map the inline decision ranges over."""
        return {
            self.tyvars[sid]: self.reps.get(sid, sid)
            for spy in self.declared_spies
            if (sid := id(spy)) != vartuple_id
        }

    def inlined(
        self,
        pool: Mapping[str, int],
        inline: AbstractSet[str],
        traces: _Traces,
    ) -> "_Naming":
        """This naming with the `inline` names dropped and the survivors renumbered."""
        remap = {
            old: _ir.tyvar_name(n)
            for n, old in enumerate(var for var in pool if var not in inline)
        }
        tyvars = {
            sid: remap.get(var, var)
            for sid, var in self.tyvars.items()
            if var not in inline
        }
        return replace(
            self,
            tyvars=tyvars,
            named={
                rep: remap.get(var, var)
                for rep, var in self.named.items()
                if var not in inline
            },
            declared_spies=[s for s in self.declared_spies if id(s) in tyvars],
            group_traces=group_traces(tyvars, self.reps, traces),
        )


@dataclass(slots=True)
class _Assign:
    """The two maps that are filled together as names are handed out."""

    reps: Mapping[int, int]
    tyvars: dict[int, str] = field(default_factory=dict)
    named: dict[int, str] = field(default_factory=dict)  # representative id -> name

    def rep(self, spy: _SpyObject) -> int:
        return self.reps.get(sid := id(spy), sid)

    def name_results(
        self,
        results: Sequence[object],
        param_ids: AbstractSet[int],
    ) -> list[_SpyObject]:
        """Name one type parameter per distinct returned expression."""
        result_spies: list[_SpyObject] = []
        for result in results:
            for spy in return_spies(result):
                sid = id(spy)
                if sid in param_ids or sid in self.tyvars:
                    continue
                # results of one op-shape share a type parameter, even traced or reused
                if (var := self.named.get(rep := self.rep(spy))) is None:
                    var = _result_tyvar(len(result_spies))
                    result_spies.append(spy)
                    self.named[rep] = var
                self.tyvars[sid] = var
        return result_spies

    def declare_typars(
        self,
        param_spies: Sequence[_SpyObject],
        order: Sequence[_SpyObject],
        appear: Mapping[int, int],
    ) -> list[_SpyObject]:
        """Name one type parameter per distinct expression used at least twice.

        Duplicates sharing a representative reuse its name.
        """
        param_ids = {id(spy) for spy in param_spies}
        candidates = [
            spy for spy in param_spies if appear[id(spy)] >= 2 or id(spy) in self.tyvars
        ]
        candidates += [
            spy
            for spy in order
            if appear[id(spy)] >= 2
            and id(spy) not in param_ids
            and id(spy) not in self.tyvars
        ]

        declared: list[_SpyObject] = []
        n = 0
        for spy in candidates:
            if (var := self.named.get(rep := self.rep(spy))) is None:
                var = self.tyvars.get(id(spy))  # a `*Ts` variadic keeps its name
                if var is None:
                    var = _ir.tyvar_name(n)
                    n += 1
                self.named[rep] = var
                declared.append(spy)
            self.tyvars[id(spy)] = var
        return declared


def build(
    results: Sequence[object],
    spies: Mapping[str, _SpyObject],
    traces: _Traces,
    varpos: _SpyObject | None,
    var_count: int,
) -> _Naming:
    """Assign a type parameter to every spy that needs one, in signature order."""
    # a returned function's parameter spies are named like regular parameters
    param_spies = [*spies.values(), *fn_spies(results)]
    order, appear = analyze(param_spies, results, traces)
    reps = representatives(order, traces)

    assign = _Assign(reps)
    vartuple = varpos is not None and all_packed(varpos, results, traces, var_count)
    if vartuple:
        # the `tyvars` entry earns `varpos` a slot and names it; `vartuple` flags it
        assign.tyvars[id(varpos)] = TYPEVAR_TUPLE_NAME

    result_spies = assign.name_results(results, {id(spy) for spy in param_spies})

    rec_body: dict[_RecVar, object] = {
        node.var: node.body
        for result in results
        for node in _walk(result)
        if isinstance(node, _Rec)
    }
    base = len(result_spies)
    rec_tyvars = {var: _result_tyvar(base + i) for i, var in enumerate(rec_body)}

    declared = assign.declare_typars(param_spies, order, appear)
    return _Naming(
        reps,
        assign.tyvars,
        assign.named,
        param_spies,
        declared,
        result_spies,
        rec_tyvars,
        rec_body,
        group_traces(assign.tyvars, reps, traces),
        vartuple,
    )
