# optype.inspect

Runtime inspection utilities for types and objects.

## Overview

A collection of functions for runtime inspection of types, modules, and other objects. These are improved alternatives to `typing` module functions that handle modern Python features correctly.

## Functions

### get_args

A better alternative to [`typing.get_args()`](https://docs.python.org/3/library/typing.html#typing.get_args):

```python
def get_args(_) -> tuple[type | object, ...]: ...
```

**Improvements**:

- Unpacks `typing.Annotated` and Python 3.12 `type _` aliases
- Recursively flattens unions and nested `typing.Literal` types
- Raises `TypeError` if not a type expression

**Example**:

```python
from typing import Literal, get_args as typing_get_args
import optype as opt

Falsy = Literal[None] | Literal[False, 0] | Literal["", b""]

# typing.get_args returns nested Literals (incorrect)
typing_get_args(Falsy)
# (Literal[None], Literal[False, 0], Literal['', b''])

# optype.inspect.get_args flattens correctly
opt.inspect.get_args(Falsy)
# (None, False, 0, '', b'')
```

### get_protocol_members

A better alternative to [`typing.get_protocol_members()`](https://docs.python.org/3.13/library/typing.html#typing.get_protocol_members):

```python
def get_protocol_members(_) -> frozenset[str]: ...
```

**Improvements**:

- Works on Python < 3.13
- Supports PEP 695 `type _` alias types
- Unpacks and flattens `Literal` types
- Treats `typing.Annotated[T]` as `T`

**Example**:

```python
from typing import Protocol
import optype as opt

class Drawable(Protocol):
    def draw(self) -> None: ...
    def resize(self, scale: float) -> None: ...

members = opt.inspect.get_protocol_members(Drawable)
# frozenset({'draw', 'resize'})
```

### get_protocols

Returns protocols from a module:

```python
def get_protocols(module, *, private: bool = False) -> frozenset[type]: ...
```

**Example**:

```python
import optype as opt

# Get public protocols from optype
protocols = opt.inspect.get_protocols(opt)
# frozenset({CanAdd, CanSub, CanMul, ...})

# Include private protocols
all_protocols = opt.inspect.get_protocols(opt, private=True)
```

### is_iterable

Check if object can be iterated without attempting to do so:

```python
def is_iterable(_) -> bool: ...
```

**Example**:

```python
import optype as opt

opt.inspect.is_iterable([1, 2, 3])     # True
opt.inspect.is_iterable("abc")         # True
opt.inspect.is_iterable(42)            # False
opt.inspect.is_iterable(range(10))     # True
```

### is_final

Check if type or method is decorated with `@typing.final`:

```python
def is_final(_) -> bool: ...
```

**Example**:

```python
from typing import final
import optype as opt

@final
class FinalClass:
    pass

class NotFinal:
    @final
    def final_method(self): ...

opt.inspect.is_final(FinalClass)                # True
opt.inspect.is_final(NotFinal)                  # False
opt.inspect.is_final(NotFinal.final_method)     # True
```

### is_protocol

Check if type is a `typing.Protocol`:

```python
def is_protocol(_) -> bool: ...
```

Backport of Python 3.13's `typing.is_protocol`.

### is_runtime_protocol

Check if type is a runtime-checkable protocol:

```python
def is_runtime_protocol(_) -> bool: ...
```

**Example**:

```python
from typing import Protocol, runtime_checkable
import optype as opt

@runtime_checkable
class RuntimeProto(Protocol):
    def method(self) -> int: ...

class JustProto(Protocol):
    def method(self) -> int: ...

opt.inspect.is_runtime_protocol(RuntimeProto)  # True
opt.inspect.is_runtime_protocol(JustProto)     # False
```

### is_union_type

Check if type is a `typing.Union`:

```python
def is_union_type(_) -> bool: ...
```

**Example**:

```python
import optype as opt

opt.inspect.is_union_type(str | int)           # True
opt.inspect.is_union_type(int)                 # False
opt.inspect.is_union_type(list[str] | dict)    # True
```

### is_generic_alias

Check if type is a subscripted generic:

```python
def is_generic_alias(_) -> bool: ...
```

**Example**:

```python
import optype as opt

opt.inspect.is_generic_alias(list[str])        # True
opt.inspect.is_generic_alias(list)             # False
opt.inspect.is_generic_alias(dict[str, int])   # True
opt.inspect.is_generic_alias(str | int)        # False (union, not subscript)
```

## Common Use Cases

### Type Introspection

```python
import optype as opt
from typing import Literal

def analyze_type(tp):
    """Analyze a type expression."""
    if opt.inspect.is_union_type(tp):
        args = opt.inspect.get_args(tp)
        print(f"Union of {len(args)} types: {args}")
    elif opt.inspect.is_generic_alias(tp):
        args = opt.inspect.get_args(tp)
        print(f"Generic with args: {args}")
    elif opt.inspect.is_protocol(tp):
        members = opt.inspect.get_protocol_members(tp)
        print(f"Protocol with members: {members}")

analyze_type(str | int | None)
analyze_type(list[str])
```

### Protocol Discovery

```python
import optype as opt

# Find all protocols in module
protos = opt.inspect.get_protocols(opt)

# Check which are runtime-checkable
runtime_protos = {
    p for p in protos
    if opt.inspect.is_runtime_protocol(p)
}
```

## Notes

- All functions work with Python 3.12 `type _` aliases
- All functions work with `typing.Annotated`
- Functions raise `TypeError` for non-type expressions (where applicable)

## Related Modules

- **[Typing](typing.md)**: Type utilities and aliases
- **[Dataclasses](dataclasses.md)**: Dataclass introspection
