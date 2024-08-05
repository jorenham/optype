import string
from typing import Final

import pytest
from typing_extensions import LiteralString, Protocol, TypeVar

import optype as opt


@pytest.mark.parametrize(
    ('opt_str', 'std_str'),
    [
        (opt.string.DIGITS_OCT, string.octdigits),
        (opt.string.DIGITS, string.digits),
        (opt.string.DIGITS_HEX, string.hexdigits),
        (opt.string.LETTERS_LOWER, string.ascii_lowercase),
        (opt.string.LETTERS_UPPER, string.ascii_uppercase),
        (opt.string.LETTERS, string.ascii_letters),
        (opt.string.PUNCTUATION, string.punctuation),
        (opt.string.WHITESPACE, string.whitespace),
        (opt.string.PRINTABLE, string.printable),
    ],
)
def test_optype_eq_stdlib(
    opt_str: tuple[LiteralString, ...],
    std_str: LiteralString,
):
    assert opt_str == tuple(std_str)


_ArgT_co = TypeVar('_ArgT_co', bound=opt.typing.AnyLiteral, covariant=True)


# a `typing.Literal` stores it's values in `__args__`
class _HasArgs(Protocol[_ArgT_co]):
    __args__: Final[tuple[_ArgT_co, ...]]


@pytest.mark.parametrize(
    ('const', 'lit'),
    [
        (opt.string.DIGITS_BIN, opt.string.DigitBin),
        (opt.string.DIGITS_OCT, opt.string.DigitOct),
        (opt.string.DIGITS, opt.string.Digit),
        (opt.string.DIGITS_HEX, opt.string.DigitHex),
        (opt.string.LETTERS_LOWER, opt.string.LetterLower),
        (opt.string.LETTERS_UPPER, opt.string.LetterUpper),
        (opt.string.LETTERS, opt.string.Letter),
        (opt.string.PUNCTUATION, opt.string.Punctuation),
        (opt.string.WHITESPACE, opt.string.Whitespace),
        (opt.string.PRINTABLE, opt.string.Printable),
    ],
)
def test_literal_args_eq_constant(
    const: tuple[LiteralString, ...],
    lit: _HasArgs[LiteralString],
):
    assert opt.inspect.get_args(lit) == const
