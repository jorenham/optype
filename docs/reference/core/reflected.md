# Reflected Operations

Protocols for reflected (swapped operand) binary operators.

## Overview

Reflected operations handle the right-hand side of binary operators when the left operand doesn't support the operation. When Python evaluates `x + y` and `x.__add__(y)` returns `NotImplemented`, Python then tries `y.__radd__(x)`.

Each reflected operation is prefixed with `CanR` and has two protocol variants:

- **`CanR*[-T, +R=T]`**: Standard form
- **`CanR*Self[-T]`**: Returns `Self`

| Expression     | Function       | Method          | Protocols                                          |
| -------------- | -------------- | --------------- | -------------------------------------------------- |
| `x + _`        | `do_radd`      | `__radd__`      | `CanRAdd[-T, +R=T]`<br>`CanRAddSelf[-T]`           |
| `x - _`        | `do_rsub`      | `__rsub__`      | `CanRSub[-T, +R=T]`<br>`CanRSubSelf[-T]`           |
| `x * _`        | `do_rmul`      | `__rmul__`      | `CanRMul[-T, +R=T]`<br>`CanRMulSelf[-T]`           |
| `x @ _`        | `do_rmatmul`   | `__rmatmul__`   | `CanRMatmul[-T, +R=T]`<br>`CanRMatmulSelf[-T]`     |
| `x / _`        | `do_rtruediv`  | `__rtruediv__`  | `CanRTruediv[-T, +R=T]`<br>`CanRTruedivSelf[-T]`   |
| `x // _`       | `do_rfloordiv` | `__rfloordiv__` | `CanRFloordiv[-T, +R=T]`<br>`CanRFloordivSelf[-T]` |
| `x % _`        | `do_rmod`      | `__rmod__`      | `CanRMod[-T, +R=T]`<br>`CanRModSelf[-T]`           |
| `divmod(x, _)` | `do_rdivmod`   | `__rdivmod__`   | `CanRDivmod[-T, +R]`                               |
| `x ** _`       | `do_rpow`      | `__rpow__`      | `CanRPow[-T, +R=T]`<br>`CanRPowSelf[-T]`           |
| `x << _`       | `do_rlshift`   | `__rlshift__`   | `CanRLshift[-T, +R=T]`<br>`CanRLshiftSelf[-T]`     |
| `x >> _`       | `do_rrshift`   | `__rrshift__`   | `CanRRshift[-T, +R=T]`<br>`CanRRshiftSelf[-T]`     |
| `x & _`        | `do_rand`      | `__rand__`      | `CanRAnd[-T, +R=T]`<br>`CanRAndSelf[-T]`           |
| `x ^ _`        | `do_rxor`      | `__rxor__`      | `CanRXor[-T, +R=T]`<br>`CanRXorSelf[-T]`           |
| `x \| _`       | `do_ror`       | `__ror__`       | `CanROr[-T, +R=T]`<br>`CanROrSelf[-T]`             |

## How Reflected Operations Work

Python's operator resolution order:

1. Try `x.__add__(y)`
2. If that returns `NotImplemented`, try `y.__radd__(x)`
3. If that also returns `NotImplemented`, raise `TypeError`

```python
# Example flow
x = MyClass(5)
y = OtherClass(3)

result = x + y  # Calls x.__add__(y)
                # If NotImplemented, then calls y.__radd__(x)
```

## Examples

### Reflected Addition

```python
from optype import CanRAddSelf

class Vector(CanRAddSelf[int]):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __radd__(self, other: int) -> "Vector":
        """Support int + Vector by converting int to Vector."""
        if not isinstance(other, int):
            return NotImplemented
        return Vector(self.x + other, self.y + other)
    
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"


vec = Vector(1, 2)
result = 10 + vec  # Calls vec.__radd__(10)
print(result)      # Vector(11, 12)
```

### Reflected Multiplication

```python
from optype import CanRMulSelf

class Scale(CanRMulSelf[float]):
    def __init__(self, factor: float):
        self.factor = factor
    
    def __rmul__(self, other: float) -> "Scale":
        """Support float * Scale by scaling the float."""
        return Scale(other * self.factor)
    
    def __repr__(self) -> str:
        return f"Scale({self.factor})"


scale = Scale(2.0)
result = 3.0 * scale  # Calls scale.__rmul__(3.0)
print(result)         # Scale(6.0)
```

### Reflected String Formatting

```python
from optype import CanRMod

class Template(CanRMod[dict, str]):
    def __init__(self, template: str):
        self.template = template
    
    def __rmod__(self, data: dict) -> str:
        """Support dict % Template."""
        return self.template.format(**data)


template = Template("Hello {name}, you are {age} years old")
result = {"name": "Alice", "age": 30} % template
print(result)  # "Hello Alice, you are 30 years old"
```

### Custom Type Coercion

```python
from optype import CanRAdd, CanRSub, CanRMul

class ComplexNumber(CanRAdd[int, "ComplexNumber"], 
                    CanRSub[int, "ComplexNumber"],
                    CanRMul[int, "ComplexNumber"]):
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag
    
    def __radd__(self, other: int) -> "ComplexNumber":
        """int + ComplexNumber"""
        if isinstance(other, int):
            return ComplexNumber(self.real + other, self.imag)
        return NotImplemented
    
    def __rsub__(self, other: int) -> "ComplexNumber":
        """int - ComplexNumber"""
        if isinstance(other, int):
            return ComplexNumber(other - self.real, -self.imag)
        return NotImplemented
    
    def __rmul__(self, other: int) -> "ComplexNumber":
        """int * ComplexNumber"""
        if isinstance(other, int):
            return ComplexNumber(self.real * other, self.imag * other)
        return NotImplemented
    
    def __repr__(self) -> str:
        sign = "+" if self.imag >= 0 else ""
        return f"({self.real}{sign}{self.imag}j)"


c = ComplexNumber(2, 3)
print(5 + c)  # (7+3j)
print(5 - c)  # (3-3j)
print(5 * c)  # (10+15j)
```

### Reflected Bitwise Operations

```python
from optype import CanRAndSelf, CanROrSelf, CanRXorSelf

class Mask(CanRAndSelf[int], CanROrSelf[int], CanRXorSelf[int]):
    def __init__(self, value: int):
        self.value = value
    
    def __rand__(self, other: int) -> "Mask":
        """int & Mask"""
        return Mask(self.value & other)
    
    def __ror__(self, other: int) -> "Mask":
        """int | Mask"""
        return Mask(self.value | other)
    
    def __rxor__(self, other: int) -> "Mask":
        """int ^ Mask"""
        return Mask(self.value ^ other)
    
    def __repr__(self) -> str:
        return f"Mask({bin(self.value)})"


mask = Mask(0b1010)
print(0b1111 & mask)  # Mask(0b1010)
print(0b1100 | mask)  # Mask(0b1110)
print(0b0011 ^ mask)  # Mask(0b1001)
```

## Important Notes

### Power Operator

`CanRPow` corresponds to the 2-argument `pow(x, y)` only. The 3-argument `pow(x, y, m)` does not have a reflected version, as Python's coercion rules don't support it for ternary operations.

### Return Values

Reflected operations should:

- Return a value of the appropriate type if they can handle the operation
- Return `NotImplemented` if they cannot handle the operation with the given operand type

## Type Parameters

- **Operand (`-T`)**: Contravariant - accepts operands of this type or supertypes
- **Result (`+R`)**: Covariant - returns this type or subtypes (defaults to operand type)

## Related Protocols

- **[Binary Operations](binary.md)**: For standard (left-hand) binary operators
- **[Inplace Operations](inplace.md)**: For augmented assignment operators
