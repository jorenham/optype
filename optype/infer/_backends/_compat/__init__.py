"""A render backend that emits valid Python (`.pyi` style) from inferred signatures.

`_Lowerer` (`_lower`) rewrites the `Signature` IR into a printable `_Module`; `_print`
then emits the text.
"""

from collections.abc import Sequence
from typing import Final, final

import optype.infer._ir as _ir  # noqa: PLR0402
from ._lower import _Lowerer
from ._model import _Protocol
from ._print import _alias_text, _func_text, _import_block, _protocol_text

__all__ = ("COMPAT", "CompatBackend")


@final
class CompatBackend:
    """Render the signatures as a self-contained, type-checkable `.pyi` stub."""

    def render(self, sigs: Sequence[_ir.Signature], /) -> str:  # noqa: PLR6301
        module = _Lowerer().module(sigs)
        used: set[str] = set()
        bodies = list(dict.fromkeys(_func_text(f, used) for f in module.funcs))
        if len(bodies) > 1:
            used.add("overload")
            bodies = [f"@overload\n{body}" for body in bodies]
        helpers = [
            _protocol_text(h, used)
            if isinstance(h, _Protocol)
            else _alias_text(h, used)
            for h in module.helpers
        ]
        locals_ = {h.name for h in module.helpers}
        typevars = {
            tp.name
            for defn in (*module.helpers, *module.funcs)
            for tp in defn.type_params
        }

        blocks: list[str] = []
        if imports := _import_block(used, locals_, typevars):
            blocks.append(imports)
        if helpers:
            blocks.append("\n".join(helpers))
        blocks.append("\n".join(bodies))
        return "\n\n".join(blocks)


COMPAT: Final[CompatBackend] = CompatBackend()
