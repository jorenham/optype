# optype.dataclasses

Protocols for Python's `dataclasses` module.

## Overview

For the [`dataclasses`](https://docs.python.org/3/library/dataclasses.html) standard
library, `optype.dataclasses` provides the `HasDataclassFields` protocol for runtime
introspection of dataclass fields.

## Protocol

### HasDataclassFields

```python
class HasDataclassFields[V: Mapping[str, Field] = Mapping[str, Field]]:
    __dataclass_fields__: V
```

**Purpose**: Runtime-checkable protocol to identify dataclasses and access their field metadata.

## Usage Examples

### Checking if Type is Dataclass

```python
from dataclasses import dataclass, Field
from optype.dataclasses import HasDataclassFields

@dataclass
class Person:
    name: str
    age: int
    email: str = ""

class NotDataclass:
    def __init__(self, name: str):
        self.name = name

# Runtime checking
p = Person("Alice", 30)
if isinstance(p, HasDataclassFields):
    print("p is a dataclass instance")
    print(f"Fields: {list(p.__dataclass_fields__.keys())}")
```

### Introspecting Dataclass Fields

```python
from dataclasses import dataclass, field, Field
from optype.dataclasses import HasDataclassFields

@dataclass
class Config(HasDataclassFields):
    host: str
    port: int = 8080
    debug: bool = field(default=False, metadata={"env": "DEBUG"})

def print_fields(obj: HasDataclassFields) -> None:
    """Print all dataclass fields."""
    for name, fld in obj.__dataclass_fields__.items():
        value = getattr(obj, name)
        print(f"{name}: {value} (type: {fld.type})")

config = Config("localhost")
print_fields(config)
```

### Working with Field Metadata

```python
from dataclasses import dataclass, field
from optype.dataclasses import HasDataclassFields

@dataclass
class Model(HasDataclassFields):
    id: int = field(metadata={"primary_key": True})
    name: str = field(metadata={"max_length": 100})
    score: float = field(default=0.0, metadata={"min": 0.0, "max": 1.0})

def validate_constraints(obj: HasDataclassFields) -> bool:
    """Validate field constraints from metadata."""
    for name, fld in obj.__dataclass_fields__.items():
        value = getattr(obj, name)
        metadata = fld.metadata
        
        if "min" in metadata and value < metadata["min"]:
            return False
        if "max" in metadata and value > metadata["max"]:
            return False
    return True

m = Model(id=1, name="test", score=0.5)
print(validate_constraints(m))  # True
```

## Type Parameter

- **`V` (Value)**: `Mapping[str, Field]` - mapping of field names to Field objects

## Related Protocols

- **[Copy](copy.md)**: Copying protocols
- **[Pickle](pickle.md)**: Serialization protocols
