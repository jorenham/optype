# Binary Operations

Protocols for Python's binary operators (`+`, `-`, `*`, `/`, `@`, `%`, `**`, `<<`, `>>`, `&`, `^`, `|`).

## Overview

These operations are called "arithmetic operations" in the Python docs, but they aren't limited to numeric types. The operations aren't required to be commutative, might be non-deterministic, and could have side-effects.

Each binary operation has three protocol variants:

- `Can*[T, R]` - Standard form
- `Can*Self[T]` - Returns `typing.Self`
- `Can*Same[T?, R?]` - Accepts `Self | T`, returns `Self | R`

| Operator               | Protocols                                                                         |
| ---------------------- | --------------------------------------------------------------------------------- |
| `_ + x`                | `CanAdd[-T, +R = T]`<br>`CanAddSelf[-T]`<br>`CanAddSame[-T?, +R?]`                |
| `_ - x`                | `CanSub[-T, +R = T]`<br>`CanSubSelf[-T]`<br>`CanSubSame[-T?, +R?]`                |
| `_ * x`                | `CanMul[-T, +R = T]`<br>`CanMulSelf[-T]`<br>`CanMulSame[-T?, +R?]`                |
| `_ @ x`                | `CanMatmul[-T, +R = T]`<br>`CanMatmulSelf[-T]`<br>`CanMatmulSame[-T?, +R?]`       |
| `_ / x`                | `CanTruediv[-T, +R = T]`<br>`CanTruedivSelf[-T]`<br>`CanTruedivSame[-T?, +R?]`    |
| `_ // x`               | `CanFloordiv[-T, +R = T]`<br>`CanFloordivSelf[-T]`<br>`CanFloordivSame[-T?, +R?]` |
| `_ % x`                | `CanMod[-T, +R = T]`<br>`CanModSelf[-T]`<br>`CanModSame[-T?, +R?]`                |
| `divmod(_, x)`         | `CanDivmod[-T, +R]`                                                               |
| `_ ** x` / `pow(_, x)` | `CanPow2[-T, +R = T]`<br>`CanPowSelf[-T]`<br>`CanPowSame[-T?, +R?]`               |
| `pow(_, x, m)`         | `CanPow3[-T, -M, +R = int]`                                                       |
| `_ << x`               | `CanLshift[-T, +R = T]`<br>`CanLshiftSelf[-T]`<br>`CanLshiftSame[-T?, +R?]`       |
| `_ >> x`               | `CanRshift[-T, +R = T]`<br>`CanRshiftSelf[-T]`<br>`CanRshiftSame[-T?, +R?]`       |
| `_ & x`                | `CanAnd[-T, +R = T]`<br>`CanAndSelf[-T]`<br>`CanAndSame[-T?, +R?]`                |
| `_ ^ x`                | `CanXor[-T, +R = T]`<br>`CanXorSelf[-T]`<br>`CanXorSame[-T?, +R?]`                |
| `_ \| x`               | `CanOr[-T, +R = T]`<br>`CanOrSelf[-T]`<br>`CanOrSame[-T?, +R?]`                   |

## Protocol Variants Explained

### Standard Form: `Can*[-T, +R = T]`

The standard protocol accepts an operand of type `T` and returns type `R` (defaulting to `T`).

### Self Form: `Can*Self[-T]`

Returns `Self` for fluent interfaces. Method signature: `(self, rhs: T, /) -> Self`.

### Same Form: `Can*Same[-T?, +R?]`

Accepts `Self | T` and returns `Self | R`. Both `T` and `R` default to `Never`.

!!! tip "pow() Special Cases"
Because `pow()` can take an optional third argument, `optype` provides:

    - `CanPow2[-T, +R = T]` for `pow(x, y)` 
    - `CanPow3[-T, -M, +R = int]` for `pow(x, y, m)`
    - `CanPow[-T, -M, +R, +RM]` as intersection type for both

See the [full README](https://github.com/jorenham/optype/blob/master/README.md#binary-operations) for complete documentation.
