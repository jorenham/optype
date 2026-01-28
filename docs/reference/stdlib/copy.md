# optype.copy

Protocols for Python's `copy` module.

## Overview

For the [`copy`](https://docs.python.org/3/library/copy.html) standard library, `optype.copy` provides runtime-checkable protocols for copying and deep copying operations.

## Protocols

### Standard Protocols

| Function                        | Method                        | Protocol          |
| ------------------------------- | ----------------------------- | ----------------- |
| `copy.copy(_)`                  | `__copy__() -> R`             | `CanCopy[+R]`     |
| `copy.deepcopy(_, memo={})`     | `__deepcopy__(memo, /) -> R`  | `CanDeepcopy[+R]` |
| `copy.replace(_, /, **changes)` | `__replace__(**changes) -> R` | `CanReplace[+R]`  |

**Note**: `copy.replace` requires Python ≥3.13, but `optype.copy.CanReplace` works on all supported Python versions.

### Self-Returning Variants

Since `typing.Self` cannot be used as a type argument, `optype.copy` provides `Can*Self` variants:

```python
# Conceptually equivalent to (but not expressible as):
type CanCopySelf = CanCopy[Self]
type CanDeepcopySelf = CanDeepcopy[Self]
type CanReplaceSelf = CanReplace[Self]
```

## Usage Examples

### Basic Copying

```python
import copy
from optype.copy import CanCopySelf, CanDeepcopySelf

class Point(CanCopySelf, CanDeepcopySelf):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __copy__(self) -> "Point":
        """Shallow copy."""
        return Point(self.x, self.y)
    
    def __deepcopy__(self, memo: dict) -> "Point":
        """Deep copy."""
        return Point(self.x, self.y)

# Usage
p1 = Point(1.0, 2.0)
p2 = copy.copy(p1)
p3 = copy.deepcopy(p1)
```

### Complex Deep Copy

```python
import copy
from optype.copy import CanDeepcopySelf

class Node(CanDeepcopySelf):
    def __init__(self, value: int, children: list["Node"] | None = None):
        self.value = value
        self.children = children or []
    
    def __deepcopy__(self, memo: dict) -> "Node":
        """Deep copy with cycle detection."""
        # Check if already copied (handles cycles)
        if id(self) in memo:
            return memo[id(self)]
        
        # Create new node
        new_node = Node(self.value)
        memo[id(self)] = new_node
        
        # Deep copy children
        new_node.children = [copy.deepcopy(child, memo) for child in self.children]
        return new_node

# Create tree with cycle
root = Node(1)
child = Node(2)
root.children.append(child)
child.children.append(root)  # Cycle!

# Deep copy handles cycles correctly
root_copy = copy.deepcopy(root)
```

### Using replace (Python 3.13+)

```python
import copy
from optype.copy import CanReplaceSelf
from dataclasses import dataclass

@dataclass(frozen=True)
class Vector(CanReplaceSelf):
    x: float
    y: float
    z: float
    
    def __replace__(self, **changes) -> "Vector":
        """Create modified copy."""
        return Vector(
            x=changes.get('x', self.x),
            y=changes.get('y', self.y),
            z=changes.get('z', self.z),
        )

# Usage
v1 = Vector(1.0, 2.0, 3.0)
v2 = copy.replace(v1, x=10.0)  # Vector(10.0, 2.0, 3.0)
```

### Custom Copy with Different Return Types

```python
import copy
from optype.copy import CanCopy, CanDeepcopy

class Resource(CanCopy["Resource"], CanDeepcopy["SharedResource"]):
    def __init__(self, data: list):
        self.data = data
    
    def __copy__(self) -> "Resource":
        """Shallow copy shares data."""
        return Resource(self.data)  # Same list!
    
    def __deepcopy__(self, memo: dict) -> "SharedResource":
        """Deep copy creates SharedResource."""
        new_data = copy.deepcopy(self.data, memo)
        return SharedResource(new_data)

class SharedResource:
    def __init__(self, data: list):
        self.data = data

# Different copy behaviors
r1 = Resource([1, 2, 3])
r2 = copy.copy(r1)      # type: Resource
r3 = copy.deepcopy(r1)  # type: SharedResource
```

## Type Parameters

- **`R` (Result)**: Covariant - the return type of the copy operation

## Related Protocols

- **[Pickle](pickle.md)**: Serialization protocols
- **[Dataclasses](dataclasses.md)**: Dataclass field introspection
