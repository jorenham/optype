"""Render the recorded spy traces as a PEP 695 signature."""

import sys
import types
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from inspect import Parameter, _ParameterKind
from typing import Any, cast, final

# `from . import` would import the package itself, which imports this module
import optype.infer._ir as _ir
import optype.infer._numpy as _numpy
from ._analyze import (
    all_packed,
    analyze,
    group_traces,
    reflect,
    representatives,
    return_spies,
    spy_runs,
)
from ._backend import TERSE, Backend
from ._protocols import Op, Proto, resolve
from ._spy import (
    _AnyFunc,
    _class_spy,
    _Marker,
    _Spy,
    _SpyBytes,
    _SpyObject,
    _SpyStr,
    _TraceItem,
    _Traces,
    as_spy,
)
from ._values import (
    COROUTINE,
    Exploration,
    _Fn,
    _Gen,
    _Rec,
    _RecRef,
    _RecVar,
    _walk,
    fn_spies,
)
from optype.inspect import _get_alias, is_generic_alias, is_union_type

_PARAM_PREFIX: dict[_ParameterKind, str] = {
    Parameter.VAR_POSITIONAL: "*",
    Parameter.VAR_KEYWORD: "**",
}

_TYPEVARS = "TUVWXYZ"
_TYPEVAR_TUPLE_NAME = "Ts"  # the PEP 646 typevar-tuple binder, used as `*Ts`

_NEVER = "Never"
_OBJECT = "object"

_TUPLE_LIMIT = 16

# `object`-typed so the `is` check isn't flagged (`Callable` is a special form)
_CALLABLE_ORIGIN: object = Callable

# the attribute variance signs of the fictional inline `Has['name', T]` form
_READ = _ir.COVARIANT  # a read-only property suffices
_WRITE = _ir.CONTRAVARIANT  # the attribute only has to accept the value

type _Vars = dict[int, str]
type Names = Sequence[str]
type Defaults = Mapping[str, object]


def _or_object(node: _ir.Node | None) -> _ir.Node:
    """The node itself, or `object` for an unconstrained (`None`) one."""
    return _ir.Name(_OBJECT) if node is None else node


def _sign_read(ret: _ir.Node) -> _ir.Node:
    """Mark a read covariant; a callable signs its return type instead: a method."""
    if isinstance(ret, _ir.Fn):
        return _ir.Fn(ret.params, _sign_read(ret.ret))
    return ret if ret == _ir.Name(_OBJECT) else _ir.Variance(_READ, ret)


def _default(defaults: Defaults, name: str) -> tuple[object] | None:
    """The boxed default value of `name`, or `None` if it has none."""
    return (defaults[name],) if name in defaults else None


_CLEAN_EXIT = _ir.App("CanExit", (_ir.Name("None"),) * 3)
_CLEAN_AEXIT = _ir.App(
    "CanAExit",
    (*(_ir.Name("None"),) * 3, _ir.App("CanAwait", (_ir.Name(_OBJECT),))),
)
_LEN = _ir.App("CanLen", ())


def _merge_combined(parts: list[_ir.Node]) -> list[_ir.Node]:
    """Merge an intersection pair into the combined protocol that optype ships.

    A `with` statement requires `__enter__` and a clean-exit `__exit__` together,
    which combine as `CanWith`; the unused `__exit__` result is unconstrained.
    `CanAsyncWith` is alike, but its declared parameters are the awaited results, so
    the `CanAwait` wrappers unwrap. `CanGetitem & CanLen` combines as `CanSequence`.
    """
    for app in parts:
        match app:
            case _ir.App("CanEnter", (entered,)):
                partner = _CLEAN_EXIT
                merged = _ir.App("CanWith", (entered, _ir.Name(_OBJECT)))
            case _ir.App("CanAEnter", (_ir.App("CanAwait", (entered,)),)):
                partner = _CLEAN_AEXIT
                merged = _ir.App("CanAsyncWith", (entered, _ir.Name(_OBJECT)))
            case _ir.App("CanGetitem", (key, value)):
                partner = _LEN
                merged = _ir.App("CanSequence", (key, value))
            case _:
                continue
        if partner in parts:
            parts = [merged if p is app else p for p in parts if p != partner]
    return parts


def _result_var(index: int) -> str:
    """The `index`-th return typevar name: `R`, `R2`, `R3`, ..."""
    return "R" if not index else f"R{index + 1}"


def _typevar_name(n: int) -> str:
    """The `n`-th generic parameter name: `T, U, ..., Z`, then `T7, T8, ...`."""
    return _TYPEVARS[n] if n < len(_TYPEVARS) else f"T{n}"


@final
class _Renderer:
    """Render an inferred `def` signature from the recorded spy traces."""

    _backend: Backend
    _spies: Mapping[str, _SpyObject]
    _fixed: Mapping[str, object]
    _results: list[object]
    _traces: _Traces
    _count: int  # the `*args` placeholder count used during exploration
    _tuple_params: frozenset[str]  # params that also accept `tuple[<bound>, ...]`

    _prefix: dict[str, str]
    _nameless: set[str]
    _optional: set[str]
    _varpos: _SpyObject | None
    _vartuple: bool  # whether `_varpos` renders as a `*Ts` typevar tuple

    _reps: dict[int, int]
    _vars: _Vars
    _named: dict[int, str]

    _result_spies: list[_SpyObject]

    _rec_vars: dict[_RecVar, str]
    _rec_body: dict[_RecVar, object]

    _declared_spies: list[_SpyObject]
    _param_spies: list[_SpyObject]
    _group_traces: _Traces

    _typer: "_ResultTyper"  # renders explored result values into type nodes

    def __init__(
        self,
        exploration: Exploration,
        params: Mapping[str, Parameter],
        traces: _Traces,
        backend: Backend = TERSE,
    ) -> None:
        # `traces` is the map to render: the raw exploration traces or the reflected one
        self._backend = backend
        self._spies = exploration.spies
        self._results = exploration.results
        self._traces = traces
        self._count = exploration.var_count
        self._fixed = exploration.fixed
        self._tuple_params = exploration.tuple_params

        self._configure(params)
        self._assign_typevars()
        # the typer shares the live `_vars` map (mutated in place while inlining)
        self._typer = _ResultTyper(
            vars_=self._vars,
            rec_vars=self._rec_vars,
            slot=self._slot,
            varpos=self._varpos,
            vartuple=self._vartuple,
            count=self._count,
        )
        self._inline_single_use()

    def _configure(self, params: Mapping[str, Parameter]) -> None:
        """The per-parameter display facts: prefix, positional-only, optional."""
        self._prefix = {n: _PARAM_PREFIX.get(p.kind, "") for n, p in params.items()}
        self._nameless = {
            n for n, p in params.items() if p.kind is Parameter.POSITIONAL_ONLY
        }
        self._optional = {
            name
            for name, p in params.items()
            if p.default is not Parameter.empty or p.kind in _PARAM_PREFIX
        }
        self._varpos = next(  # ty:ignore[invalid-assignment]
            (
                self._spies[name]
                for name, p in params.items()
                if p.kind is Parameter.VAR_POSITIONAL
            ),
            None,
        )

    def _assign_typevars(self) -> None:
        """Assign a type parameter to every spy that needs one, in signature order.

        Populates the spy->name map (`_vars`), the per-representative bound traces
        (`_group_traces`), the `_declared_spies` pool, and result/recursive typevars.
        """
        results, traces = self._results, self._traces
        # a returned function's parameter spies are named like regular parameters
        param_spies = [*self._spies.values(), *fn_spies(results)]
        order, appear = analyze(param_spies, results, traces)
        param_ids = {id(spy) for spy in param_spies}
        self._reps = reps = representatives(order, traces)

        self._vars = {}
        self._named = {}  # representative id -> name
        varpos = self._varpos
        self._vartuple = varpos is not None and all_packed(
            varpos,
            results,
            traces,
            self._count,
        )
        if self._vartuple:
            # the `_vars` entry earns `varpos` a slot and names it; `_vartuple` flags it
            self._vars[id(varpos)] = _TYPEVAR_TUPLE_NAME

        self._result_spies = []
        self._name_results(param_ids, reps)

        self._rec_body = {
            node.var: node.body
            for result in results
            for node in _walk(result)
            if isinstance(node, _Rec)
        }
        base = len(self._result_spies)
        self._rec_vars = {
            var: _result_var(base + i) for i, var in enumerate(self._rec_body)
        }

        self._declare_typevars(param_spies, order, appear, param_ids, reps)
        self._param_spies = param_spies
        self._group_traces = group_traces(self._vars, reps, traces)

    def _declare_typevars(
        self,
        param_spies: Sequence[_SpyObject],
        order: Sequence[_SpyObject],
        appear: Mapping[int, int],
        param_ids: set[int],
        reps: dict[int, int],
    ) -> None:
        """Name one type parameter per distinct expression used at least twice.

        Duplicates sharing a representative reuse its name.
        """
        candidates = [
            spy for spy in param_spies if appear[id(spy)] >= 2 or id(spy) in self._vars
        ]
        candidates += [
            spy
            for spy in order
            if appear[id(spy)] >= 2
            and id(spy) not in param_ids
            and id(spy) not in self._vars
        ]

        self._declared_spies = []
        n = 0
        for spy in candidates:
            rep = reps.get(id(spy), id(spy))
            if (var := self._named.get(rep)) is None:
                var = self._vars.get(id(spy))  # a `*Ts` variadic keeps its name
                if var is None:
                    var = _typevar_name(n)
                    n += 1
                self._named[rep] = var
                self._declared_spies.append(spy)
            self._vars[id(spy)] = var

    def _inline_single_use(self) -> None:
        # a bounded typevar referenced once and absent from the return carries no more
        # than its bound, so it inlines back into the one spot that uses it

        vars_, reps = self._vars, self._reps

        bounds = {
            rep: self.traces(self._group_traces.get(rep, ())) for rep in self._named
        }
        rendered = [
            *(self._slot(spy) for spy in self._param_spies),
            *(node for node in bounds.values() if node is not None),
            self._typer.type_union(self._results),
        ]
        counts = Counter(name for node in rendered for name in _ir.names(node))

        vartuple_id = id(self._varpos) if self._vartuple else None
        pool_vars = {
            vars_[sid]: reps.get(sid, sid)
            for spy in self._declared_spies
            if (sid := id(spy)) != vartuple_id
        }
        inline = {
            var
            for var, rep in pool_vars.items()
            if counts[var] == 1 and bounds[rep] is not None
        }
        if not inline:
            return

        # renumber the survivors back to a gapless `T, U, V, ...`
        remap = {
            old: _typevar_name(n)
            for n, old in enumerate(var for var in pool_vars if var not in inline)
        }

        # mutate `_vars` in place so the typer's shared reference stays current
        survivors = {
            sid: remap.get(var, var) for sid, var in vars_.items() if var not in inline
        }
        self._vars.clear()
        self._vars.update(survivors)
        self._declared_spies = [
            spy for spy in self._declared_spies if id(spy) in self._vars
        ]
        self._named = {
            rep: remap.get(var, var)
            for rep, var in self._named.items()
            if var not in inline
        }
        self._group_traces = group_traces(self._vars, reps, self._traces)

    def _name_results(self, param_ids: set[int], reps: dict[int, int]) -> None:
        for result in self._results:
            for spy in return_spies(result):
                sid = id(spy)
                if sid in param_ids or sid in self._vars:
                    continue
                # results of one op-shape share a type parameter, even traced or reused
                rep = reps.get(sid, sid)
                if (var := self._named.get(rep)) is not None:
                    self._vars[sid] = var
                    continue
                var = _result_var(len(self._result_spies))
                self._vars[sid] = var
                self._result_spies.append(spy)
                self._named[rep] = var

    def returns(self, members: Iterable[Op]) -> _ir.Node | None:
        named: list[str] = []
        items: list[_TraceItem] = []
        returned = False
        for m in members:
            if not isinstance(m.ret, _SpyObject):
                continue
            returned = True
            if (var := self._vars.get(id(m.ret))) is not None:
                named.append(var)
            else:
                items.extend(self._traces[id(m.ret)])
        parts: list[_ir.Node] = [_ir.Name(var) for var in dict.fromkeys(named)]
        if items and (node := self.traces(items)) is not None:
            parts.append(node)
        if (out := _ir.inter(parts)) is None and returned:
            # an unused result is unconstrained, but omitting it would leave the
            # last argument in the return slot, e.g. the `R` of `(T) -> R`
            out = _ir.Name(_OBJECT)
        return out

    def _collapse_varargs(
        self,
        members: Sequence[Op],
        pos: list[_ir.Node],
    ) -> list[_ir.Node]:
        # the variadic spreads `count` copies of one spy: the placeholder (`f(*args)`)
        # or its iterated element (`map`); collapse the trailing run to `*tuple[T, ...]`
        varpos = self._varpos
        call_args = members[0].args
        if varpos is None or not call_args:
            return pos

        tail = call_args[-1]
        if not isinstance(tail, _SpyObject) or (
            tail is not varpos and tail is not varpos.__optype_element__
        ):
            return pos

        # require every call to star-unpack the same trailing run, else order decides
        count = self._count
        if not all(
            m.args[-1] is tail and spy_runs(m.args, tail)[-1] == count for m in members
        ):
            return pos

        inner = _ir.tuple_node_variadic(self._typer.return_type(tail))
        return [*pos[: len(pos) - count], _ir.Unpack(inner)]

    def group(self, proto: Proto, members: Sequence[Op]) -> _ir.Node:
        if isinstance(proto, tuple):  # coercion protocols, which record no args
            return _ir.Union(tuple(map(_ir.Name, proto)))

        if proto == "CanArrayFunction":
            ret = _or_object(self.returns(members))
            func = cast("_AnyFunc", members[0].args[0])
            return _numpy.array_function_node(func, ret)

        pos = [
            arg
            for i in range(len(members[0].args))
            if (arg := self._typer.value_union(m.args[i] for m in members)) is not None
        ]
        args = (
            *pos,
            *(
                _ir.Arg(key, kw)
                for key in members[0].kwargs
                if (kw := self._typer.value_union(m.kwargs[key] for m in members))
                is not None
            ),
        )
        ret = self.returns(members)

        if proto == "CanCall":
            call_pos = self._collapse_varargs(members, pos)
            return _ir.Fn((*call_pos, *args[len(pos) :]), _or_object(ret))

        if (attr := members[0].attr) is not None:
            # a read is covariant (a property suffices) and a write contravariant
            # (the attribute only has to accept the value); existence renders bare
            signed: tuple[_ir.Node, ...] = tuple(_ir.Variance(_WRITE, a) for a in pos)
            if ret is not None and ret != _ir.Name(_OBJECT):
                signed = *signed, _sign_read(ret)
            if members[0].classvar:
                signed = (_ir.App("ClassVar", signed),)
            return _ir.App(proto, (_ir.Name(repr(attr)), *signed))

        if ret is not None:
            args = *args, ret

        return _ir.App(proto, args)

    def traces(self, items: Iterable[_TraceItem]) -> _ir.Node | None:
        # dedup the re-collected return-chain items so they can't blow up (#734)
        items = list({id(item): item for item in items}.values())
        # an absence marker means the op was optional; an attribute probe keys on its
        # name, so it spares the other reads
        optional = {item.args for item in items if item.attr == _Marker.ABSENT}
        groups: dict[
            tuple[Proto, str | None, bool, int, tuple[str, ...]],
            list[Op],
        ] = {}
        for item in items:
            if item.attr == _Marker.ABSENT:
                continue
            probed = (
                ("__getattr__", item.args[0])
                if item.attr == "__getattr__" and item.args
                else (item.attr,)
            )
            if probed in optional:
                continue
            op = resolve(item)
            key = op.proto, op.attr, op.classvar, len(op.args), tuple(sorted(op.kwargs))
            groups.setdefault(key, []).append(op)
        parts = [self.group(key[0], group) for key, group in groups.items()]
        return _ir.inter(_merge_combined(parts))

    def spy(self, spy: _SpyObject) -> _ir.Node | None:
        return self.traces(self._traces[id(spy)])

    def _slot(self, spy: _SpyObject) -> _ir.Node:
        if self._vartuple and spy is self._varpos:
            return _ir.Unpack(_ir.Name(_TYPEVAR_TUPLE_NAME))
        if (var := self._vars.get(id(spy))) is not None:
            return _ir.Name(var)
        return _or_object(self.spy(spy))

    def typevar(
        self,
        spy: _SpyObject,
        defaulted: Mapping[int, _ir.Node],
        *,
        negate: bool = False,
    ) -> _ir.TypeParam:
        var = self._vars[id(spy)]
        if self._vartuple and spy is self._varpos:
            # a PEP 646 typevar tuple takes no bound or default
            return _ir.TypeParam(var, unpack=True)
        rep = self._reps.get(id(spy), id(spy))
        node = self.traces(self._group_traces.get(rep, ()))
        default = defaulted.get(id(spy))
        if negate and default is not None:
            node = _ir.exclude(node, default)
        return _ir.TypeParam(var, bound=node, default=None if negate else default)

    def signature(
        self,
        selected: Names,
        defaults: Defaults | None = None,
        *,
        negate: bool = False,
        ret: _ir.Node | None = None,
        deprecated: str | None = None,
    ) -> _ir.Signature:
        defaults = defaults or {}
        defaulted = {
            spy_id: node
            for name, value in defaults.items()
            if (spy := self._spies.get(name)) is not None
            if (spy_id := id(spy)) in self._vars
            if (node := self._typer.value_union((value,))) is not None
        }
        ordered = self._declared_spies + self._result_spies
        if not negate:
            # PEP 696 requires defaulted type parameters to come last
            ordered.sort(key=lambda spy: id(spy) in defaulted)
        type_params = [self.typevar(spy, defaulted, negate=negate) for spy in ordered]

        # a recursive typevar carries no default, so it precedes the defaulted tail
        deferred = 0 if negate else sum(id(spy) in defaulted for spy in ordered)
        cut = len(ordered) - deferred
        type_params[cut:cut] = [
            _ir.TypeParam(name, bound=self._typer.return_type(self._rec_body[var]))
            for var, name in self._rec_vars.items()
        ]

        params = [
            param
            for name in selected
            if (param := self._param(name, defaults, negate=negate)) is not None
        ]
        ret_node = self._typer.type_union(self._results) if ret is None else ret
        return _ir.Signature(
            tuple(type_params),
            tuple(params),
            ret_node,
            deprecated,
        )

    def render(
        self,
        selected: Names,
        defaults: Defaults | None = None,
        *,
        negate: bool = False,
        ret: _ir.Node | None = None,
        deprecated: str | None = None,
    ) -> str:
        sig = self.signature(
            selected,
            defaults,
            negate=negate,
            ret=ret,
            deprecated=deprecated,
        )
        return self._backend.render(sig)

    def _param(
        self,
        name: str,
        defaults: Defaults,
        *,
        negate: bool,
    ) -> _ir.Param | None:
        # a positional-only parameter cannot be passed by keyword, so no name shows
        nameless = name in self._nameless
        prefix = self._prefix[name]
        if name in self._fixed and name not in self._optional:
            # a fixed parameter without a default is a method descriptor's `self`
            node = _ir.Type(type(self._fixed[name]))
            return _ir.Param(name, node, prefix, nameless)
        if (spy := self._spies.get(name)) is None:
            # an omitted parameter binds its default, so passing it behaves the same
            node = self._typer.value_type(defaults[name])
            return _ir.Param(name, node, prefix, nameless, _default(defaults, name))
        node = self._slot(spy)
        if negate and name in defaults:
            if id(spy) not in self._vars and (
                mark := self._typer.value_union((defaults[name],))
            ):
                node = _ir.exclude(self.spy(spy), mark)
            return _ir.Param(name, node, prefix, nameless)
        if name in self._tuple_params and id(spy) not in self._vars:
            # a typevar keeps its binding, so only an inlined bound widens to the union
            node = _ir.union([node, _ir.tuple_node_variadic(node)]) or node
        if node == _ir.Name(_OBJECT) and name in self._optional:
            return None
        return _ir.Param(name, node, prefix, nameless, _default(defaults, name))


@final
class _ResultTyper:
    """Render an explored runtime value as a type node, against the renderer's binding.

    Bridges back via the `slot` callback (and the shared live `_vars`), so renderer and
    typer are mutually recursive.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        vars_: _Vars,
        rec_vars: dict[_RecVar, str],
        slot: Callable[[_SpyObject], _ir.Node],
        varpos: _SpyObject | None,
        vartuple: bool,
        count: int,
    ) -> None:
        self._vars = vars_  # the live map the renderer mutates while inlining
        self._rec_vars = rec_vars
        self._slot = slot
        self._varpos = varpos
        self._vartuple = vartuple
        self._count = count

    def value_union(
        self,
        values: Iterable[object],
        *,
        tuples: bool = False,
    ) -> _ir.Node | None:
        """The union of `values` as literals; `type_union` widens them to types."""
        literals: list[object] = []
        parts: list[_ir.Node] = []
        for value in values:
            if isinstance(value, (int, str, bytes)) and not isinstance(value, _Spy):
                literals.append(value)
            else:
                parts.append(self.return_type(value))

        nodes: list[_ir.Node] = []
        if literals:
            nodes.append(_ir.Lit(tuple(literals)))
        nodes.extend(dict.fromkeys(parts))
        return _ir.union(nodes, tuples=tuples)

    def return_type(self, result: object) -> _ir.Node:
        match result:
            case _RecRef() | _Rec():
                node = _ir.Name(self._rec_vars[result.var])
            case _SpyObject() if (spy := as_spy(result)) is not None:
                node = _ir.Name(self._vars.get(id(spy), _OBJECT))
            case _SpyStr():
                node = _ir.Type(str)
            case _SpyBytes():
                node = _ir.Type(bytes)
            case _Gen():
                node = self._generator_type(result)
            case slice():
                node = _ir.App(
                    _ir.type_name(slice),
                    (
                        self.value_type(result.start),
                        self.value_type(result.stop),
                        self.value_type(result.step),
                    ),
                )
            case _Fn():
                node = self._function(result)
            case _ if result is None or _ir.is_sentinel(result):
                node = _ir.Name(repr(result))
            case _:
                node = self._container(result)
        return node

    def _generator_type(self, result: _Gen) -> _ir.Node:
        if result.kind == COROUTINE:
            # an awaitable yields objects and is sent `None`, as `CanAwait`
            out = self.type_union(result.yielded)
            return _ir.App(COROUTINE, (_ir.Name(_OBJECT), _ir.Name("None"), out))
        if not result.yielded and "." in result.kind:
            # a qualified (`itertools`/`functools`) kind drops the misleading `[Never]`
            return _ir.Name(result.kind)
        return _ir.App(result.kind, (self.type_union(result.yielded),))

    def type_union(self, values: Iterable[object]) -> _ir.Node:
        """The deduplicated union of the types of `values`, or `Never` if empty."""
        parts = dict.fromkeys(map(self.return_type, values))
        return _ir.union(parts, tuples=True) or _ir.Name(_NEVER)

    def value_type(self, value: object) -> _ir.Node:
        """The type of a single `value`, or `Never` if unconstrained."""
        return self.value_union((value,)) or _ir.Name(_NEVER)

    def _function(self, fn: _Fn) -> _ir.Node:
        """The signature-syntax type of an explored function result."""
        params = tuple(
            _ir.Arg(
                None if p.kind is Parameter.POSITIONAL_ONLY else name,
                self._fn_param(fn, name),
                _default(fn.defaults, name),
            )
            for name, p in fn.params.items()
        )
        return _ir.Fn(params, self.type_union(fn.results))

    def _fn_param(self, fn: _Fn, name: str) -> _ir.Node:
        if (spy := fn.spies.get(name)) is not None:
            return self._slot(spy)
        value = fn.fixed[name]
        if name in fn.defaults:
            # a pinned default renders as its value, like the outer parameters do
            return self.value_type(value)
        return _ir.Type(type(value))  # a method descriptor's pinned `self`

    def _class_of(self, cls: type[Any]) -> _ir.Node | None:
        """The type of `cls`'s instances, if it is expressible."""
        if (spy := _class_spy(cls)) is not None:
            return self.return_type(spy)
        if issubclass(cls, _SpyStr):
            return _ir.Type(str)
        if issubclass(cls, _SpyBytes):
            return _ir.Type(bytes)
        if cls is type(None):
            return _ir.Name("None")
        if getattr(sys.modules.get(cls.__module__), cls.__name__, None) is cls:
            return _ir.Type(cls)
        return None  # a local class has no nameable type, so it stays a bare `type`

    def _type_expr(self, value: object) -> _ir.Node | None:
        """The type a value *is*, not the type *of* it: `list[int]` -> `list[int]`."""
        value = _get_alias(value)
        if isinstance(value, type):
            return self._class_of(value)
        if is_union_type(value):  # `int | str` and `typing.Union[int, str]` alike
            parts = self._type_args(value.__args__)
            return None if parts is None else _ir.union(parts)
        if is_generic_alias(value):  # builtin `list[int]` and user `Foo[int]` alike
            if value.__origin__ is _CALLABLE_ORIGIN:
                # `Callable`'s `__args__` is a flat `(*params, ret)`, not a type row
                return self._callable_type(value.__args__)
            origin = self._type_expr(value.__origin__)
            args = self._type_args(value.__args__)
            if isinstance(origin, _ir.Type) and args is not None:
                return _ir.App(_ir.type_name(origin.cls), tuple(args))
        return None

    def _callable_type(self, args: tuple[object, ...]) -> _ir.Node | None:
        """A `Callable[...]` value as the `(params) -> ret` form rendered elsewhere."""
        *params, ret = args  # `_type_args` renders a `Callable[..., R]`'s `...` too
        ret_node = self._type_expr(ret)
        param_nodes = self._type_args(params)
        if ret_node is None or param_nodes is None:
            return None
        return _ir.Fn(tuple(param_nodes), ret_node)

    def _type_args(self, values: Sequence[object]) -> list[_ir.Node] | None:
        """Every type argument as an annotation node, or `None` if any is unnameable."""
        args: list[_ir.Node] = []
        for value in values:
            node = _ir.Dots() if value is ... else self._type_expr(value)
            if node is None:
                return None
            args.append(node)
        return args

    def _container(self, result: object) -> _ir.Node:
        result = _get_alias(result)
        if (inner := self._type_expr(result)) is not None:
            # `type[...]` for a class, `TypeForm[...]` for a union or `Callable`
            class_form = isinstance(result, type) or (
                is_generic_alias(result) and result.__origin__ is not _CALLABLE_ORIGIN
            )
            return _ir.App("type" if class_form else "TypeForm", (inner,))

        if is_generic_alias(result):
            # an unnameable origin or argument keeps the honest, if unhelpful, alias
            return _ir.Type(types.GenericAlias)

        cls = type(result)
        match result:
            case Mapping():
                mapping = cast("Mapping[object, object]", result)
                key = self.value_union(mapping, tuples=True) or _ir.Name(_NEVER)
                val = self.value_union(mapping.values(), tuples=True) or _ir.Name(
                    _NEVER,
                )
                return _ir.App(_ir.type_name(cls), (key, val))
            case list() | set() | frozenset():
                inner = self.value_union(
                    cast("Collection[object]", result),
                    tuples=True,
                )
                return _ir.App(_ir.type_name(cls), (inner or _ir.Name(_NEVER),))
            case tuple() if cls is tuple:
                return (
                    self._tuple(cast("tuple[object, ...]", result))
                    if result
                    else _ir.App("tuple", ())
                )
            case _:
                return _ir.Type(cls)

    def _tuple(self, items: tuple[object, ...]) -> _ir.Node:
        spy = self._varpos
        if spy is not None:
            if any(item is spy for item in items) and self._vartuple:
                # every use is packed, so the placeholders unpack into a single `*Ts`
                start = next(i for i, item in enumerate(items) if item is spy)
                parts = [self.value_type(item) for item in items if item is not spy]
                parts.insert(start, _ir.Unpack(_ir.Name(_TYPEVAR_TUPLE_NAME)))
                return _ir.tuple_node(parts)

            # a uniform spread is `tuple[T, ...]`: the placeholder (`(*args,)`) at any
            # length, or its zipped element (`zip(*args)`) only at the full count
            full_count = len(items) == self._count
            for target, exact in ((spy, False), (spy.__optype_element__, True)):
                if (
                    target is not None
                    and (full_count or not exact)
                    and all(item is target for item in items)
                ):
                    return _ir.tuple_node_variadic(self.return_type(target))

        if len(items) > _TUPLE_LIMIT:  # e.g. `random.getstate`
            return _ir.tuple_node_variadic(self.type_union(items))

        return _ir.tuple_node(self.value_type(item) for item in items)


def _renderers(
    exploration: Exploration,
    params: Mapping[str, Parameter],
    *,
    backend: Backend = TERSE,
) -> list[_Renderer]:
    traces, results = exploration.traces, exploration.results
    reflected = reflect([*exploration.spies.values(), *fn_spies(results)], traces)
    return [_Renderer(exploration, params, t, backend) for t in (traces, reflected)]


def signatures(  # noqa: PLR0913
    exploration: Exploration,
    params: Mapping[str, Parameter],
    selected: Names,
    defaults: Defaults | None = None,
    *,
    negate: bool = False,
    backend: Backend = TERSE,
) -> list[str]:
    return [
        r.render(selected, defaults, negate=negate, deprecated=exploration.deprecated)
        for r in _renderers(exploration, params, backend=backend)
    ]


def widened_signature(
    exploration: Exploration,
    params: Mapping[str, Parameter],
    selected: Names,
    *,
    backend: Backend = TERSE,
) -> list[str]:
    """The fallback overload for a value dispatch, deduplicated: the parameter as the
    absent branch leaves it, the return widened to `object` (the only sound supertype of
    an arbitrary present return).

    Empty if a signature still declares a type parameter: a pinned return cannot bind
    one, so a parameter-only typevar would dangle.
    """
    sigs = [
        r.signature(selected, ret=_ir.Name(_OBJECT))
        for r in _renderers(exploration, params, backend=backend)
    ]
    if any(sig.type_params for sig in sigs):
        return []
    return list(dict.fromkeys(backend.render(sig) for sig in sigs))


def union_signature(
    exploration: Exploration,
    variant: Exploration,
    params: Mapping[str, Parameter],
    selected: Names,
    *,
    backend: Backend = TERSE,
) -> list[str]:
    """The single overload for a presence dispatch whose return ignores the attribute's
    value: the parameter (forced absent in `variant`) widens to `object`, and the return
    unions the present and absent branches (so a `hasattr` predicate renders `bool`).
    """
    combined = variant._replace(results=[*exploration.results, *variant.results])
    return signatures(combined, params, selected, backend=backend)
