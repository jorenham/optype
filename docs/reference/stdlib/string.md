# optype.string

The [`string`](https://docs.python.org/3/library/string.html) standard
library contains practical constants, but it has two issues:

- The constants contain a collection of characters, but are represented as
  a single string. This makes it practically impossible to type-hint the
  individual characters, so typeshed currently types these constants as a
  `LiteralString`.
- The names of the constants are inconsistent, and doesn't follow
  [PEP 8](https://peps.python.org/pep-0008/#constants).

So instead, `optype.string` provides an alternative interface, that is
compatible with `string`, but with slight differences:

- For each constant, there is a corresponding `Literal` type alias for
  the *individual* characters. Its name matches the name of the constant,
  but is singular instead of plural.
- Instead of a single string, `optype.string` uses a `tuple` of characters,
  so that each character has its own `typing.Literal` annotation.
  Note that this is only tested with (based)pyright / pylance, so it might
  not work with mypy (it has more bugs than it has lines of codes).
- The names of the constant are consistent with PEP 8, and use a postfix
  notation for variants, e.g. `DIGITS_HEX` instead of `hexdigits`.
- Unlike `string`, `optype.string` has a constant (and type alias) for
  binary digits `'0'` and `'1'`; `DIGITS_BIN` (and `DigitBin`). Because
  besides `oct` and `hex` functions in `builtins`, there's also the
  `builtins.bin` function.

<table>
    <tr>
        <th colspan="2"><code>string._</code></th>
        <th colspan="2"><code>optype.string._</code></th>
    </tr>
    <tr>
        <th>constant</th>
        <th>char type</th>
        <th>constant</th>
        <th>char type</th>
    </tr>
    <tr>
        <td colspan="2" align="center"><i>missing</i></td>
        <td><code>DIGITS_BIN</code></td>
        <td><code>DigitBin</code></td>
    </tr>
    <tr>
        <td><code>octdigits</code></td>
        <td rowspan="9"><code>LiteralString</code></td>
        <td><code>DIGITS_OCT</code></td>
        <td><code>DigitOct</code></td>
    </tr>
    <tr>
        <td><code>digits</code></td>
        <td><code>DIGITS</code></td>
        <td><code>Digit</code></td>
    </tr>
    <tr>
        <td><code>hexdigits</code></td>
        <td><code>DIGITS_HEX</code></td>
        <td><code>DigitHex</code></td>
    </tr>
    <tr>
        <td><code>ascii_letters</code></td>
        <td><code>LETTERS</code></td>
        <td><code>Letter</code></td>
    </tr>
    <tr>
        <td><code>ascii_lowercase</code></td>
        <td><code>LETTERS_LOWER</code></td>
        <td><code>LetterLower</code></td>
    </tr>
    <tr>
        <td><code>ascii_uppercase</code></td>
        <td><code>LETTERS_UPPER</code></td>
        <td><code>LetterUpper</code></td>
    </tr>
    <tr>
        <td><code>punctuation</code></td>
        <td><code>PUNCTUATION</code></td>
        <td><code>Punctuation</code></td>
    </tr>
    <tr>
        <td><code>whitespace</code></td>
        <td><code>WHITESPACE</code></td>
        <td><code>Whitespace</code></td>
    </tr>
    <tr>
        <td><code>printable</code></td>
        <td><code>PRINTABLE</code></td>
        <td><code>Printable</code></td>
    </tr>
</table>

Each of the `optype.string` constants is exactly the same as the corresponding
`string` constant (after concatenation / splitting), e.g.

```pycon
>>> import string
>>> import optype as opt
>>> "".join(opt.string.PRINTABLE) == string.printable
True
>>> tuple(string.printable) == opt.string.PRINTABLE
True
```

Similarly, the values within a constant's `Literal` type exactly match the
values of its constant:

```pycon
>>> import optype as opt
>>> from optype.inspect import get_args
>>> get_args(opt.string.Printable) == opt.string.PRINTABLE
True
```

The `optype.inspect.get_args` is a non-broken variant of `typing.get_args`
that correctly flattens nested literals, type-unions, and PEP 695 type aliases,
so that it matches the official typing specs.
*In other words; `typing.get_args` is yet another fundamentally broken
python-typing feature that's useless in the situations where you need it
most.*
