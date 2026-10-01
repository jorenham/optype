"""A render backend that emits valid Python (`.pyi` style) from inferred signatures.

`Lowerer` (`_lower`) rewrites the `Signature` IR into a printable `Module`; `_print`
then emits the text.
"""

from collections.abc import Sequence

import optype.infer._ir as _ir  # ruff: ignore[manual-from-import]
from ._lower import Lowerer
from ._print import Printer


def render(sigs: Sequence[_ir.Signature], /) -> str:
    """Export supported signatures as a `.pyi` stub, rejecting known lossy forms."""
    module = Lowerer().module(sigs)
    printer = Printer()
    bodies = list(dict.fromkeys(printer.func_text(f) for f in module.funcs))
    if len(bodies) > 1:
        printer.used.add("overload")
        bodies = [f"@overload\n{body}" for body in bodies]
    helpers = [printer.protocol_text(h) for h in module.helpers]
    locals_ = {h.name for h in module.helpers}
    tyvars = {p.name for h in module.helpers for p in h.type_params}
    tyvars.update(p.name for f in module.funcs for p in f.type_params)

    blocks: list[str] = []
    if imports := printer.import_block(locals_, tyvars):
        blocks.append(imports)
    if helpers:
        blocks.append("\n".join(helpers))
    blocks.append("\n".join(bodies))
    return "\n\n".join(blocks)
