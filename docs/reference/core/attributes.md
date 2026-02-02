# Attributes

Protocols for accessing and manipulating object attributes.

## Overview

The `optype` library provides two categories of attribute-related protocols:

1. **`Can*` Protocols** - For attribute operations like `getattr()`, `setattr()`, and `delattr()`
2. **`Has*` Protocols** - For checking the presence of special attributes like `__name__`, `__dict__`, `__doc__`, etc.

## Attribute Operations

These protocols describe the operations for getting, setting, and deleting attributes.

| Operation          | Protocol                    |
| ------------------ | --------------------------- |
| `getattr(_, k)`    | `CanGetattr[+V = object]`   |
| `setattr(_, k, v)` | `CanSetattr[-V = object]`   |
| `delattr(_, k)`    | `CanDelattr`                |
| `dir(_)`           | `CanDir[+R: Iterable[str]]` |

## Attribute Presence Protocols

The `Has*` protocols check for the presence of special attributes:

| Attribute         | Protocol         |
| ----------------- | ---------------- |
| `__name__`        | `HasName`        |
| `__doc__`         | `HasDoc`         |
| `__dict__`        | `HasDict`        |
| `__module__`      | `HasModule`      |
| `__qualname__`    | `HasQualname`    |
| `__annotations__` | `HasAnnotations` |
| `__class__`       | `HasClass`       |

## Examples

```python
import optype as op
```

### Getting Attributes

```python
def get_name(obj: op.CanGetattr[str]) -> str:
    """Get the name of an object."""
    return getattr(obj, '__name__')
```

### Setting Attributes

```python
def set_metadata(obj: op.CanSetattr[dict], metadata: dict) -> None:
    """Set metadata on an object."""
    setattr(obj, '__metadata__', metadata)
```

### Checking Attribute Presence

```python
from typing import Protocol

class HasNameAndDoc(op.HasName, op.HasDoc, Protocol): ...

def describe(obj: HasNameAndDoc) -> str:
    """Get name and docstring."""
    return f"{obj.__name__}: {obj.__doc__}"
```

### Directory Listing

```python
def list_public_attrs(obj: op.CanDir[list]) -> list[str]:
    """List non-private attributes."""
    return [name for name in dir(obj) if not name.startswith('_')]
```
