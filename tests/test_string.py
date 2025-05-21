import string
from typing import Final, LiteralString, Protocol, TypeVar

import pytest

import optype as op


@pytest.mark.parametrize(
    ("opt_str", "std_str"),
    [
        (op.string.DIGITS_OCT, string.octdigits),
        (op.string.DIGITS, string.digits),
        (op.string.DIGITS_HEX, string.hexdigits),
        (op.string.LETTERS_LOWER, string.ascii_lowercase),
        (op.string.LETTERS_UPPER, string.ascii_uppercase),
        (op.string.LETTERS, string.ascii_letters),
        (op.string.PUNCTUATION, string.punctuation),
        (op.string.WHITESPACE, string.whitespace),
        (op.string.PRINTABLE, string.printable),
    ],
)
def test_optype_eq_stdlib(
    opt_str: tuple[LiteralString, ...],
    std_str: LiteralString,
) -> None:
    assert opt_str == tuple(std_str)


_ArgT_co = TypeVar("_ArgT_co", bound=op.typing.AnyLiteral, covariant=True)


# a `typing.Literal` stores it's values in `__args__`
class _HasArgs(Protocol[_ArgT_co]):
    __args__: Final[tuple[_ArgT_co, ...]]  # type: ignore[misc]


@pytest.mark.parametrize(
    ("const", "lit"),
    [
        (op.string.DIGITS_BIN, op.string.DigitBin),
        (op.string.DIGITS_OCT, op.string.DigitOct),
        (op.string.DIGITS, op.string.Digit),
        (op.string.DIGITS_HEX, op.string.DigitHex),
        (op.string.LETTERS_LOWER, op.string.LetterLower),
        (op.string.LETTERS_UPPER, op.string.LetterUpper),
        (op.string.LETTERS, op.string.Letter),
        (op.string.PUNCTUATION, op.string.Punctuation),
        (op.string.WHITESPACE, op.string.Whitespace),
        (op.string.PRINTABLE, op.string.Printable),
    ],
)
def test_literal_args_eq_constant(
    const: tuple[LiteralString, ...],
    lit: _HasArgs[LiteralString],
) -> None:
    assert op.inspect.get_args(lit) == const
