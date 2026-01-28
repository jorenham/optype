# Unary Operations

Protocols for unary operators.

## Overview

Unary operations apply a single operand and transform it into a new value. Python provides four unary operators:

| Expression | Operator    | Method       | Use Case                  |
| ---------- | ----------- | ------------ | ------------------------- |
| `+x`       | Positive    | `__pos__`    | Explicitly positive value |
| `-x`       | Negative    | `__neg__`    | Negation/inversion        |
| `~x`       | Bitwise NOT | `__invert__` | Bitwise complement        |
| `abs(x)`   | Absolute    | `__abs__`    | Magnitude/distance        |

Each operator has two protocol variants:

- **Standard**: Returns a potentially different type `R`
- **Self-returning**: Returns `Self` (optionally with alternative type union)

| Expression | Function    | Protocol                                |
| ---------- | ----------- | --------------------------------------- |
| `+_`       | `do_pos`    | `CanPos[+R]`<br>`CanPosSelf[+R?]`       |
| `-_`       | `do_neg`    | `CanNeg[+R]`<br>`CanNegSelf[+R?]`       |
| `~_`       | `do_invert` | `CanInvert[+R]`<br>`CanInvertSelf[+R?]` |
| `abs(_)`   | `do_abs`    | `CanAbs[+R]`<br>`CanAbsSelf[+R?]`       |

## Protocol Details

### Standard Protocols: `Can*[+R]`

Return a value of type `R` (can be different from the operand type):

```python
class MyType(CanNeg[int]):
    def __neg__(self) -> int:
        return 42  # Return a different type
```

### Self-Returning Protocols: `Can*Self[+R?]`

Return `Self`. The optional `R` parameter (available since optype 0.12.1) allows specifying an alternative return type:

- **Without `R`**: Returns `-> Self`
- **With `R`**: Returns `-> Self | R`

```python
class Point(CanNegSelf):
    def __neg__(self) -> "Point":
        return Point(-self.x, -self.y)

class FlexibleNegation(CanNegSelf[int]):
    def __neg__(self) -> "FlexibleNegation | int":
        # Can return either Self or int
        return self if self.value >= 0 else -self.value
```

## Examples

### Positive and Negative Operations

```python
from optype import CanPosSelf, CanNegSelf

class Temperature(CanPosSelf, CanNegSelf):
    def __init__(self, celsius: float):
        self.celsius = celsius
    
    def __pos__(self) -> "Temperature":
        """Unary plus - returns copy."""
        return Temperature(self.celsius)
    
    def __neg__(self) -> "Temperature":
        """Unary minus - negates temperature."""
        return Temperature(-self.celsius)
    
    def __repr__(self) -> str:
        return f"Temperature({self.celsius}°C)"


temp = Temperature(25.0)
print(+temp)  # Temperature(25.0°C)
print(-temp)  # Temperature(-25.0°C)
```

### Bitwise Inversion

```python
from optype import CanInvertSelf

class Flags(CanInvertSelf[int]):
    def __init__(self, value: int):
        self.value = value
    
    def __invert__(self) -> "Flags | int":
        """Bitwise NOT - can return Flags or raw int."""
        inverted = ~self.value
        # Return Flags for small values, raw int otherwise
        return Flags(inverted) if abs(inverted) < 256 else inverted
    
    def __repr__(self) -> str:
        return f"Flags({bin(self.value)})"


flags = Flags(0b1010)
print(~flags)  # Flags(0b...11111111111111111111111111110101)
```

### Absolute Value

```python
from optype import CanAbsSelf

class Vector(CanAbsSelf):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __abs__(self) -> "Vector":
        """Return vector with absolute components."""
        return Vector(abs(self.x), abs(self.y))
    
    def magnitude(self) -> float:
        """Calculate actual magnitude."""
        return (self.x**2 + self.y**2) ** 0.5
    
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"


vec = Vector(-3.0, -4.0)
abs_vec = abs(vec)
print(abs_vec)            # Vector(3.0, 4.0)
print(abs_vec.magnitude()) # 5.0
```

### Complex Number Operations

```python
import math
from optype import CanPos, CanNeg, CanAbs

class Complex(CanPos[float], CanNeg["Complex"], CanAbs[float]):
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag
    
    def __pos__(self) -> float:
        """Unary plus - return magnitude."""
        return self.magnitude()
    
    def __neg__(self) -> "Complex":
        """Unary minus - negate both parts."""
        return Complex(-self.real, -self.imag)
    
    def __abs__(self) -> float:
        """Absolute value - magnitude."""
        return self.magnitude()
    
    def magnitude(self) -> float:
        """Calculate magnitude."""
        return math.sqrt(self.real**2 + self.imag**2)
    
    def __repr__(self) -> str:
        sign = "+" if self.imag >= 0 else ""
        return f"({self.real}{sign}{self.imag}j)"


c = Complex(3.0, 4.0)
print(+c)      # 5.0
print(-c)      # (-3.0-4.0j)
print(abs(c))  # 5.0
```

### Numeric Type with Type Transformation

```python
from optype import CanNeg, CanAbs

class Rational(CanNeg["Rational"], CanAbs[int]):
    def __init__(self, numerator: int, denominator: int):
        from math import gcd
        g = gcd(abs(numerator), abs(denominator))
        self.num = numerator // g
        self.den = denominator // g
    
    def __neg__(self) -> "Rational":
        """Negate the rational number."""
        return Rational(-self.num, self.den)
    
    def __abs__(self) -> int:
        """Return numerator (simplified to single value)."""
        return abs(self.num)
    
    def __repr__(self) -> str:
        return f"Rational({self.num}/{self.den})"


r = Rational(-22, 7)
print(-r)     # Rational(22/7)
print(abs(r)) # 22
```

## Type Parameters

### Standard Protocols: `Can*[+R]`

- **`R`**: Covariant - return type of the operation (can be any type)

### Self-Returning Protocols: `Can*Self[+R?]`

- **`R`**: Covariant - optional alternative return type
  - Default: `Never` (returns only `Self`)
  - When provided: Returns `Self | R`

## Important Notes

### Operator Behavior

- **`+x` (Positive)**: By convention, should return a copy or equivalent value. Often used to normalize types.
- **`-x` (Negative)**: Negates or inverts the value. For numeric types, represents the additive inverse.
- **`~x` (Invert)**: Bitwise NOT for integers. Can represent logical negation or bitwise complement.
- **`abs(x)` (Absolute)**: Returns the magnitude or distance from zero. Result should always be non-negative.

### Type Consistency

Many numeric types implement multiple unary operations:

```python
x: float = -3.14
x1: CanPosSelf = x           # Supports +x
x2: CanNegSelf = x           # Supports -x
x3: CanAbsSelf = x           # Supports abs(x)
```

### Self-Returning Convention

Using `CanPosSelf` instead of `CanPos[Self]` is more idiomatic and provides better type inference:

```python
# Preferred
class Point(CanNegSelf):
    def __neg__(self) -> "Point": ...

# Less preferred
class Point(CanNeg["Point"]):
    def __neg__(self) -> "Point": ...
```

## Related Protocols

- **[Binary Operations](binary.md)**: For operations with two operands
- **[Reflected Operations](reflected.md)**: For swapped operand binary operators
- **[Inplace Operations](inplace.md)**: For augmented assignment operators
- **[Rounding](rounding.md)**: For numeric rounding operations
