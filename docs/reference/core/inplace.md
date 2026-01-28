# Inplace Operations

Protocols for in-place assignment operators (`+=`, `-=`, `*=`, etc.).

## Overview

In-place operations allow objects to modify themselves in place and return the modified result. These operations are prefixed with `CanI` in `optype`. Each in-place operation has three protocol variants to handle different return value scenarios.

## Protocol Variants

For each in-place operation, `optype` provides three protocol variants:

1. **`CanI*[-T, +R]`**: Standard form accepting operand of type `T` and returning type `R`
2. **`CanI*Self[-T]`**: Returns `Self` after mutation
3. **`CanI*Same[-T?]`**: Accepts `Self | T` and returns `Self`

| Operator  | Function       | Method          | Protocols                                                                   |
| --------- | -------------- | --------------- | --------------------------------------------------------------------------- |
| `_ += x`  | `do_iadd`      | `__iadd__`      | `CanIAdd[-T, +R]`<br>`CanIAddSelf[-T]`<br>`CanIAddSame[-T?]`                |
| `_ -= x`  | `do_isub`      | `__isub__`      | `CanISub[-T, +R]`<br>`CanISubSelf[-T]`<br>`CanISubSame[-T?]`                |
| `_ *= x`  | `do_imul`      | `__imul__`      | `CanIMul[-T, +R]`<br>`CanIMulSelf[-T]`<br>`CanIMulSame[-T?]`                |
| `_ @= x`  | `do_imatmul`   | `__imatmul__`   | `CanIMatmul[-T, +R]`<br>`CanIMatmulSelf[-T]`<br>`CanIMatmulSame[-T?]`       |
| `_ /= x`  | `do_itruediv`  | `__itruediv__`  | `CanITruediv[-T, +R]`<br>`CanITruedivSelf[-T]`<br>`CanITruedivSame[-T?]`    |
| `_ //= x` | `do_ifloordiv` | `__ifloordiv__` | `CanIFloordiv[-T, +R]`<br>`CanIFloordivSelf[-T]`<br>`CanIFloordivSame[-T?]` |
| `_ %= x`  | `do_imod`      | `__imod__`      | `CanIMod[-T, +R]`<br>`CanIModSelf[-T]`<br>`CanIModSame[-T?]`                |
| `_ **= x` | `do_ipow`      | `__ipow__`      | `CanIPow[-T, +R]`<br>`CanIPowSelf[-T]`<br>`CanIPowSame[-T?]`                |
| `_ <<= x` | `do_ilshift`   | `__ilshift__`   | `CanILshift[-T, +R]`<br>`CanILshiftSelf[-T]`<br>`CanILshiftSame[-T?]`       |
| `_ >>= x` | `do_irshift`   | `__irshift__`   | `CanIRshift[-T, +R]`<br>`CanIRshiftSelf[-T]`<br>`CanIRshiftSame[-T?]`       |
| `_ &= x`  | `do_iand`      | `__iand__`      | `CanIAnd[-T, +R]`<br>`CanIAndSelf[-T]`<br>`CanIAndSame[-T?]`                |
| `_ ^= x`  | `do_ixor`      | `__ixor__`      | `CanIXor[-T, +R]`<br>`CanIXorSelf[-T]`<br>`CanIXorSame[-T?]`                |
| `_ \|= x` | `do_ior`       | `__ior__`       | `CanIOr[-T, +R]`<br>`CanIOrSelf[-T]`<br>`CanIOrSame[-T?]`                   |

## Protocol Details

### Standard Form: `CanI*[-T, +R]`

Accepts an operand of type `T` and returns type `R`.

```python
from optype import CanIAdd

def add_inplace(lst: CanIAdd[int, list], value: int) -> list:
    lst += value
    return lst
```

### Self Form: `CanI*Self[-T]`

Returns `typing.Self` after modification. Useful for methods that mutate an object and return itself.

```python
from optype import CanIAddSelf

class Counter(CanIAddSelf[int]):
    def __init__(self, value: int = 0):
        self.value = value
    
    def __iadd__(self, other: int) -> "Counter":
        self.value += other
        return self
```

### Same Form: `CanI*Same[-T?]`

Accepts `Self | T` and returns `Self`. This is useful for operations that accept both instances of the same type and other types.

```python
from optype import CanIAddSame

class Vector(CanIAddSame[int]):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __iadd__(self, other: "Vector | int") -> "Vector":
        if isinstance(other, Vector):
            self.x += other.x
            self.y += other.y
        else:
            self.x += other
            self.y += other
        return self
```

## Examples

### Mutable List-like Container

```python
from optype import CanIAddSelf, CanISubSelf

class MutableList(CanIAddSelf[list], CanISubSelf[object]):
    def __init__(self, items: list):
        self.items = items
    
    def __iadd__(self, other: list) -> "MutableList":
        self.items.extend(other)
        return self
    
    def __isub__(self, item: object) -> "MutableList":
        if item in self.items:
            self.items.remove(item)
        return self


ml = MutableList([1, 2, 3])
ml += [4, 5]
print(ml.items)  # [1, 2, 3, 4, 5]
ml -= 2
print(ml.items)  # [1, 3, 4, 5]
```

### Numeric Type with Multiple Operations

```python
from optype import CanIAddSelf, CanISubSelf, CanIMulSelf, CanITruedivSelf

class Decimal(CanIAddSelf[int], CanISubSelf[int], CanIMulSelf[int], CanITruedivSelf[int]):
    def __init__(self, value: float):
        self.value = value
    
    def __iadd__(self, other: int) -> "Decimal":
        self.value += other
        return self
    
    def __isub__(self, other: int) -> "Decimal":
        self.value -= other
        return self
    
    def __imul__(self, other: int) -> "Decimal":
        self.value *= other
        return self
    
    def __itruediv__(self, other: int) -> "Decimal":
        self.value /= other
        return self
    
    def __repr__(self) -> str:
        return f"Decimal({self.value})"


d = Decimal(10)
d += 5
d *= 2
d /= 4
print(d)  # Decimal(7.5)
```

### Bitwise Operations

```python
from optype import CanIAndSelf, CanIOrSelf, CanIXorSelf

class BitFlags(CanIAndSelf[int], CanIOrSelf[int], CanIXorSelf[int]):
    def __init__(self, value: int):
        self.value = value
    
    def __iand__(self, other: int) -> "BitFlags":
        self.value &= other
        return self
    
    def __ior__(self, other: int) -> "BitFlags":
        self.value |= other
        return self
    
    def __ixor__(self, other: int) -> "BitFlags":
        self.value ^= other
        return self
    
    def __repr__(self) -> str:
        return f"BitFlags({bin(self.value)})"


flags = BitFlags(0b1010)
flags |= 0b0101  # OR
print(flags)     # BitFlags(0b1111)
flags &= 0b1100  # AND
print(flags)     # BitFlags(0b1100)
```

### Generic In-place Operation Handler

```python
from typing import TypeVar
from optype import CanIAdd

T = TypeVar("T")

def accumulate(obj: CanIAdd[T, object], values: list[T]) -> object:
    """Accumulate values using in-place addition."""
    for value in values:
        obj += value
    return obj


result = accumulate([1, 2], [3, 4, 5])
print(result)  # [1, 2, 3, 4, 5]
```

## Type Parameters

- **Operand (`-T`)**: Contravariant - accepts operands of this type or supertypes
- **Result (`+R`)**: Covariant - returns this type or subtypes
- **`Self`**: Returns the same type as the object being modified

## Notes

- In-place operations should typically modify the object and return it (or `Self`)
- If an in-place operation is not defined, Python falls back to the corresponding binary operation
- The `CanI*Self` aliases exist because `Self` cannot be used directly in generic aliases in current Python
- The `CanI*Same` protocols (available since 0.12.1) default `T` to `Never`, meaning they accept only `Self` if `T` is not specified

## Related Protocols

- **[Binary Operations](binary.md)**: For non-mutating binary operators
- **[Reflected Operations](reflected.md)**: For reflected/right-hand operators
