"""A render backend that emits valid Python (`.pyi` style) from inferred signatures.

`Lowerer` (`_lower`) rewrites the `Signature` IR into a printable `Module`; `_print`
then emits the text.
"""

from collections.abc import Sequence
from typing import Final, final

import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]
from ._lower import Lowerer
from ._model import ProtocolDef
from ._print import Printer


@final
class CompatBackend:
    """Render the signatures as a self-contained, type-checkable `.pyi` stub."""

    def render(self, sigs: Sequence[_ir.Signature], /) -> str:  # ruff: ignore[no-self-use]
        module = Lowerer().module(sigs)
        printer = Printer()
        bodies = list(dict.fromkeys(printer.func_text(f) for f in module.funcs))
        if len(bodies) > 1:
            printer.record("overload")
            bodies = [f"@overload\n{body}" for body in bodies]
        helpers = [
            printer.protocol_text(h)
            if isinstance(h, ProtocolDef)
            else printer.alias_text(h)
            for h in module.helpers
        ]
        locals_ = {h.name for h in module.helpers}
        typevars = {
            tp.name
            for defn in (*module.helpers, *module.funcs)
            for tp in defn.type_params
        }

        blocks: list[str] = []
        if imports := printer.import_block(locals_, typevars):
            blocks.append(imports)
        if helpers:
            blocks.append("\n".join(helpers))
        blocks.append("\n".join(bodies))
        return "\n\n".join(blocks)


COMPAT: Final[CompatBackend] = CompatBackend()
