# optype.string

String literal constants with precise typing.

## Overview

PEP 8 compliant alternatives to the [`string`](https://docs.python.org/3/library/string.html) module constants, providing `Literal` types for individual characters.

## String Constants

### Comparison with `string` Module

| `optype.string` | `string` Module   | Characters                                        |
| --------------- | ----------------- | ------------------------------------------------- |
| `DIGITS_BIN`    | —                 | `'01'`                                            |
| `DIGITS_OCT`    | `octdigits`       | `'01234567'`                                      |
| `DIGITS_DEC`    | `digits`          | `'0123456789'`                                    |
| `DIGITS_HEX`    | `hexdigits`       | `'0123456789abcdefABCDEF'`                        |
| `LETTERS_LOWER` | `ascii_lowercase` | `'abcdefghijklmnopqrstuvwxyz'`                    |
| `LETTERS_UPPER` | `ascii_uppercase` | `'ABCDEFGHIJKLMNOPQRSTUVWXYZ'`                    |
| `LETTERS`       | `ascii_letters`   | `LETTERS_LOWER + LETTERS_UPPER`                   |
| `PUNCTUATION`   | `punctuation`     | ``!"#$%&'()*+,-./:;<=>?@[\]^_`{\|}~``             |
| `WHITESPACE`    | `whitespace`      | `' \t\n\r\v\f'`                                   |
| `PRINTABLE`     | `printable`       | `DIGITS_DEC + LETTERS + PUNCTUATION + WHITESPACE` |

## Key Differences

### Tuple-Based Storage

Unlike `str` constants in the `string` module, `optype.string` constants are `tuple[str, ...]`:

```python
import string
import optype.string as ops

# Standard library uses str
type(string.digits)  # <class 'str'>
string.digits[0]     # '0'

# optype uses tuple for better typing
type(ops.DIGITS_DEC)  # <class 'tuple'>
ops.DIGITS_DEC[0]     # '0'
```

### Literal Character Types

Each character has a precise `Literal` type:

```python
from optype.string import DigitDec, LetterLower, Whitespace
from typing import Literal

# Individual character Literal types
DigitDec  # Literal['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LetterLower  # Literal['a', 'b', ..., 'z']
Whitespace  # Literal[' ', '\t', '\n', '\r', '\x0b', '\x0c']
```

## Usage Examples

### Character Validation

```python
from optype.string import DIGITS_DEC, DigitDec
from typing import TypeGuard

def is_digit(char: str) -> TypeGuard[DigitDec]:
    \"\"\"Check if character is a decimal digit.\"\"\"
    return char in DIGITS_DEC

# Type-safe validation
char: str = '5'
if is_digit(char):
    # Type checker knows char is DigitDec here
    print(f"Valid digit: {char}")
```

### Parsing Hexadecimal

```python
from optype.string import DIGITS_HEX, DigitHex

def parse_hex_char(char: DigitHex) -> int:
    \"\"\"Convert hex character to integer.\"\"\"
    if char in '0123456789':
        return int(char)
    elif char in 'abcdef':
        return ord(char) - ord('a') + 10
    else:  # 'ABCDEF'
        return ord(char) - ord('A') + 10

# Works with literal types
parse_hex_char('a')   # 10
parse_hex_char('F')   # 15
parse_hex_char('9')   # 9
```

### String Cleaning

```python
from optype.string import WHITESPACE, PRINTABLE

def clean_string(text: str) -> str:
    \"\"\"Remove non-printable characters.\"\"\"
    return ''.join(c for c in text if c in PRINTABLE)

def normalize_whitespace(text: str) -> str:
    \"\"\"Replace all whitespace with single spaces.\"\"\"
    result = text
    for ws in WHITESPACE:
        result = result.replace(ws, ' ')
    return ' '.join(result.split())

# Usage
clean_string("Hello\x00World")  # "HelloWorld"
normalize_whitespace("Hello\t\n  World")  # "Hello World"
```

### Letter Case Detection

```python
from optype.string import LETTERS_LOWER, LETTERS_UPPER, LetterLower, LetterUpper

def classify_letter(char: str) -> str | None:
    \"\"\"Classify a character as upper, lower, or not a letter.\"\"\"
    if char in LETTERS_LOWER:
        return "lowercase"
    elif char in LETTERS_UPPER:
        return "uppercase"
    else:
        return None

# Type-safe letter operations
def to_upper(char: LetterLower) -> LetterUpper:
    \"\"\"Convert lowercase to uppercase.\"\"\"
    return char.upper()  # type: ignore[return-value]

classify_letter('a')  # "lowercase"
classify_letter('Z')  # "uppercase"
classify_letter('1')  # None
```

### Binary String Parsing

```python
from optype.string import DIGITS_BIN, DigitBin

def parse_binary(bits: str) -> int | None:
    \"\"\"Parse binary string to integer.\"\"\"
    if not all(c in DIGITS_BIN for c in bits):
        return None
    return int(bits, 2)

def is_binary_string(s: str) -> bool:
    \"\"\"Check if string contains only binary digits.\"\"\"
    return all(c in DIGITS_BIN for c in s)

parse_binary('1010')  # 10
parse_binary('1012')  # None
is_binary_string('0101')  # True
```

### Identifier Validation

```python
from optype.string import LETTERS, DIGITS_DEC, Letter, DigitDec

def is_valid_identifier_char(char: str, first: bool = False) -> bool:
    \"\"\"Check if character is valid in Python identifier.\"\"\"
    if first:
        return char in LETTERS or char == '_'
    return char in LETTERS or char in DIGITS_DEC or char == '_'

def validate_identifier(name: str) -> bool:
    \"\"\"Validate Python identifier.\"\"\"
    if not name:
        return False
    if not is_valid_identifier_char(name[0], first=True):
        return False
    return all(is_valid_identifier_char(c) for c in name[1:])

validate_identifier('my_var_123')  # True
validate_identifier('123_var')     # False
validate_identifier('_private')    # True
```

### Token Parsing

```python
from optype.string import WHITESPACE, PUNCTUATION, Punctuation

def tokenize_simple(text: str) -> list[str]:
    \"\"\"Split text on whitespace and punctuation.\"\"\"
    # Replace punctuation with spaces
    for punct in PUNCTUATION:
        text = text.replace(punct, f' {punct} ')
    
    # Split on whitespace
    return [token for token in text.split() if token]

def split_on_punctuation(text: str) -> list[str]:
    \"\"\"Split preserving punctuation as separate tokens.\"\"\"
    tokens: list[str] = []
    current = ''
    
    for char in text:
        if char in PUNCTUATION:
            if current:
                tokens.append(current)
                current = ''
            tokens.append(char)
        elif char not in WHITESPACE:
            current += char
        elif current:
            tokens.append(current)
            current = ''
    
    if current:
        tokens.append(current)
    
    return tokens

tokenize_simple("Hello, World!")  # ['Hello', ',', 'World', '!']
```

## Literal Character Types

Individual character types for precise typing:

### Digit Types

- `DigitBin`: `Literal['0', '1']`
- `DigitOct`: `Literal['0', '1', '2', '3', '4', '5', '6', '7']`
- `DigitDec`: `Literal['0', '1', ..., '9']`
- `DigitHex`: `Literal['0', '1', ..., '9', 'a', ..., 'f', 'A', ..., 'F']`

### Letter Types

- `LetterLower`: `Literal['a', 'b', ..., 'z']`
- `LetterUpper`: `Literal['A', 'B', ..., 'Z']`
- `Letter`: `LetterLower | LetterUpper`

### Other Character Types

- `Punctuation`: All ASCII punctuation marks
- `Whitespace`: `Literal[' ', '\t', '\n', '\r', '\x0b', '\x0c']`
- `Printable`: All printable ASCII characters

## PEP 8 Compliance

The naming follows PEP 8 conventions:

- **UPPER_CASE**: For constants (e.g., `DIGITS_DEC`, `LETTERS_LOWER`)
- **PascalCase**: For type aliases (e.g., `DigitDec`, `LetterLower`)

This differs from the `string` module which uses lowercase for constants.

## Performance Considerations

### Membership Testing

```python
# Tuple membership (optype.string)
char in DIGITS_DEC  # O(n) but small n (10 items)

# String membership (string module)
char in string.digits  # O(n) string search

# For better performance with many checks, use sets
DIGITS_DEC_SET = set(DIGITS_DEC)
char in DIGITS_DEC_SET  # O(1) average case
```

## Related Types

- **[Typing](typing.md)**: Type alias utilities
- **[JSON](json.md)**: String-based serialization
- **[IO](io.md)**: Text I/O protocols
