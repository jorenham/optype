# Rounding

Protocols for rounding operations.

## Overview

Python provides several rounding functions in the built-in namespace and `math` module. The `round()` function has two overloads (with 1 and 2 parameters), while `math.trunc()`, `math.floor()`, and `math.ceil()` each have fixed signatures.

`optype` provides separate protocols for each overload and function, allowing precise type hints for code that implements these special methods.

## Built-in `round()`

The `round()` function has two distinct overloads:

| Expression        | Function   | Protocol                      |
| ----------------- | ---------- | ----------------------------- |
| `round(_)`        | `do_round` | `CanRound1[+R=int]`           |
| `round(_, n)`     | `do_round` | `CanRound2[-T=int, +R=float]` |
| `round(_, n=...)` | `do_round` | `CanRound[-T, +R1, +R2]`      |

### Overload Details

**`CanRound1[+R=int]`** - `round(_)` with no second argument:

- Implements `__round__` with no parameters (or optional `ndigits=None`)
- Returns a value (typically `int`)
- Example: `round(3.14)` → `3`

**`CanRound2[-T=int, +R=float]`** - `round(_, n)` with second argument:

- Implements `__round__` with `ndigits` parameter
- `ndigits` type parameter defaults to `int`
- Returns a value (typically `float` when `ndigits` is provided)
- Example: `round(3.14159, 2)` → `3.14`

**`CanRound[-T, +R1, +R2]`** - Overloaded union:

- Combines both overloads: `CanRound1[R1] & CanRound2[T, R2]`
- Used for typing objects that support both `round()` call styles

## Math Module Functions

| Expression      | Function   | Protocol           |
| --------------- | ---------- | ------------------ |
| `math.trunc(_)` | `do_trunc` | `CanTrunc[+R=int]` |
| `math.floor(_)` | `do_floor` | `CanFloor[+R=int]` |
| `math.ceil(_)`  | `do_ceil`  | `CanCeil[+R=int]`  |

All three functions have fixed signatures taking a single argument and returning an integer (by default).

## Examples

```python
import optype as op
```

### Custom Decimal Type with Both Rounding Overloads

```python
from decimal import Decimal

class Fraction(op.CanRound[int, int, "Fraction"]):
    def __init__(self, numerator: int, denominator: int):
        self.num = numerator
        self.den = denominator
    
    def __round__(self, ndigits: int | None = None) -> int | "Fraction":
        """Support both round(fraction) and round(fraction, n)."""
        if ndigits is None:
            # round(fraction) → int
            return round(self.num / self.den)
        else:
            # round(fraction, n) → Fraction
            # Keep n decimal places by scaling denominator
            scale = 10 ** ndigits
            new_num = round((self.num / self.den) * scale)
            return Fraction(new_num, scale)
    
    def __repr__(self) -> str:
        return f"Fraction({self.num}, {self.den})"


frac = Fraction(22, 7)
print(round(frac))        # 3
print(round(frac, 2))     # Fraction(314, 100)
```

### Custom Numeric Type with Truncation

```python
class Temperature(op.CanTrunc[int]):
    def __init__(self, celsius: float):
        self.celsius = celsius
    
    def __trunc__(self) -> int:
        """Truncate to integer Celsius."""
        return int(self.celsius)
    
    def __repr__(self) -> str:
        return f"Temperature({self.celsius}°C)"


temp = Temperature(98.6)
truncated = int(temp)  # Calls __trunc__()
print(truncated)       # 98
```

### Custom Container with Floor and Ceil

```python
import math

class Vector(op.CanFloor[int], op.CanCeil[int]):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __floor__(self) -> int:
        """Return magnitude (floored)."""
        magnitude = (self.x**2 + self.y**2) ** 0.5
        return math.floor(magnitude)
    
    def __ceil__(self) -> int:
        """Return magnitude (ceiled)."""
        magnitude = (self.x**2 + self.y**2) ** 0.5
        return math.ceil(magnitude)
    
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"


vec = Vector(3.2, 4.1)
print(math.floor(vec))  # 5
print(math.ceil(vec))   # 6
```

### Rational Number with All Rounding Methods

```python
import math

class Rational(op.CanRound[int, int, "Rational"], op.CanTrunc[int],
               op.CanFloor[int], op.CanCeil[int]):
    def __init__(self, numerator: int, denominator: int):
        self.num = numerator
        self.den = denominator
        # Simplify fraction
        from math import gcd
        g = gcd(numerator, denominator)
        self.num //= g
        self.den //= g
    
    def __round__(self, ndigits: int | None = None) -> int | "Rational":
        """Support both round() variants."""
        if ndigits is None:
            return round(self.num / self.den)
        else:
            scale = 10 ** ndigits
            new_num = round((self.num / self.den) * scale)
            return Rational(new_num, scale)
    
    def __trunc__(self) -> int:
        """Truncate towards zero."""
        return self.num // self.den
    
    def __floor__(self) -> int:
        """Floor towards negative infinity."""
        return math.floor(self.num / self.den)
    
    def __ceil__(self) -> int:
        """Ceiling towards positive infinity."""
        return math.ceil(self.num / self.den)
    
    def __repr__(self) -> str:
        return f"Rational({self.num}, {self.den})"


r = Rational(22, 7)
print(round(r))        # 3
print(round(r, 2))     # Rational(314, 100)
print(math.trunc(r))   # 3
print(math.floor(r))   # 3
print(math.ceil(r))    # 4
```

## Type Parameters

### `CanRound` Variants

- **`CanRound1[-R=int]`**:
  - `R`: Covariant - return type of `round(_)`

- **`CanRound2[-T=int, +R=float]`**:
  - `T`: Contravariant - type of `ndigits` parameter
  - `R`: Covariant - return type when `ndigits` is provided

- **`CanRound[-T, +R1, +R2]`**:
  - `T`: Contravariant - type accepted for `ndigits`
  - `R1`: Covariant - return type of `round(_)` (no second arg)
  - `R2`: Covariant - return type of `round(_, ndigits)`

### Math Module Functions

- **`CanTrunc[+R=int]`**, **`CanFloor[+R=int]`**, **`CanCeil[+R=int]`**:
  - `R`: Covariant - return type (defaults to `int`)

## Important Notes

### Default Return Types

While all rounding functions typically return `int`, the protocols allow any return type. This provides flexibility for custom numeric types that want to return different types.

### Built-in `round()` Behavior

The built-in `round()` uses banker's rounding (round half to even):

```python
round(0.5)    # 0
round(1.5)    # 2
round(2.5)    # 2
```

Custom implementations may use different rounding strategies.

### Type Compatibility

Float objects implement all four rounding protocols:

```python
x: float = 3.14
x1: CanRound1[int] = x
x2: CanRound2[int, float] = x
x3: CanRound[int, int, float] = x
x4: CanTrunc[int] = x
x5: CanFloor[int] = x
x6: CanCeil[int] = x
```

## Related Protocols

- **[Binary Operations](binary.md)**: For arithmetic operators
- **[Unary Operations](unary.md)**: For negation and absolute value
- **[Conversion](conversion.md)**: For type conversion protocols
