import string
from typing import Final

import pytest
from typing_extensions import LiteralString, Protocol, TypeVar

import optype as o


@pytest.mark.parametrize(
    ("opt_str", "std_str"),
    [
        (o.string.DIGITS_OCT, string.octdigits),
        (o.string.DIGITS, string.digits),
        (o.string.DIGITS_HEX, string.hexdigits),
        (o.string.LETTERS_LOWER, string.ascii_lowercase),
        (o.string.LETTERS_UPPER, string.ascii_uppercase),
        (o.string.LETTERS, string.ascii_letters),
        (o.string.PUNCTUATION, string.punctuation),
        (o.string.WHITESPACE, string.whitespace),
        (o.string.PRINTABLE, string.printable),
    ],
)
def test_optype_eq_stdlib(
    opt_str: tuple[LiteralString, ...],
    std_str: LiteralString,
) -> None:
    assert opt_str == tuple(std_str)


_ArgT_co = TypeVar("_ArgT_co", bound=o.typing.AnyLiteral, covariant=True)


# a `typing.Literal` stores it's values in `__args__`
class _HasArgs(Protocol[_ArgT_co]):
    __args__: Final[tuple[_ArgT_co, ...]]  # type: ignore[misc]


@pytest.mark.parametrize(
    ("const", "lit"),
    [
        (o.string.DIGITS_BIN, o.string.DigitBin),
        (o.string.DIGITS_OCT, o.string.DigitOct),
        (o.string.DIGITS, o.string.Digit),
        (o.string.DIGITS_HEX, o.string.DigitHex),
        (o.string.LETTERS_LOWER, o.string.LetterLower),
        (o.string.LETTERS_UPPER, o.string.LetterUpper),
        (o.string.LETTERS, o.string.Letter),
        (o.string.PUNCTUATION, o.string.Punctuation),
        (o.string.WHITESPACE, o.string.Whitespace),
        (o.string.PRINTABLE, o.string.Printable),
    ],
)
def test_literal_args_eq_constant(
    const: tuple[LiteralString, ...],
    lit: _HasArgs[LiteralString],
) -> None:
    assert o.inspect.get_args(lit) == const
