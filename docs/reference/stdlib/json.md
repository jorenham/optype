# optype.json

Type aliases for JSON serialization.

## Overview

Type aliases for the [`json`](https://docs.python.org/3/library/json.html) standard library, providing precise typing for JSON values, arrays, and objects.

## Type Aliases

### Value Types

| Type       | Use Case                      |
| ---------- | ----------------------------- |
| `Value`    | Return type of `json.load(s)` |
| `AnyValue` | Input type for `json.dump(s)` |

### Container Types

| Type                      | Generic Parameter    | Use Case                |
| ------------------------- | -------------------- | ----------------------- |
| `Array[V: Value]`         | JSON array (output)  | `list` from `json.load` |
| `AnyArray[~V: AnyValue]`  | JSON array (input)   | `list` for `json.dump`  |
| `Object[V: Value]`        | JSON object (output) | `dict` from `json.load` |
| `AnyObject[~V: AnyValue]` | JSON object (input)  | `dict` for `json.dump`  |

## Type Relationships

```python
# Value is a subtype of AnyValue
Value <: AnyValue

# Therefore:
AnyValue | Value ≡ AnyValue

# Composition:
Value ≡ Value | Array | Object
AnyValue ≡ AnyValue | AnyArray | AnyObject
```

## Usage Examples

### Type-Safe JSON Loading

```python
import json
from optype.json import Value, Array, Object

def load_config(path: str) -> Object:
    \"\"\"Load JSON config file.\"\"\"
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be JSON object")
    return data

# Type-safe access
config = load_config("config.json")
host: Value = config.get("host", "localhost")
```

### JSON API Response

```python
from optype.json import Value, Array, Object
import json

def fetch_users() -> Array[Object]:
    \"\"\"Fetch user list from API.\"\"\"
    response = '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
    data = json.loads(response)
    
    if not isinstance(data, list):
        raise ValueError("Expected array")
    return data

users = fetch_users()
for user in users:
    if isinstance(user, dict):
        print(f"User: {user['name']}")
```

### Generic JSON Handler

```python
from optype.json import AnyValue
import json

def serialize(data: AnyValue) -> str:
    \"\"\"Serialize any JSON-compatible data.\"\"\"
    return json.dumps(data, indent=2)

# Works with any JSON type
serialize({"key": "value"})
serialize([1, 2, 3])
serialize("string")
serialize(42)
serialize(None)
serialize(True)
```

### Nested JSON Structures

```python
from optype.json import Object, Array, Value
import json

def process_nested(data: Object[Object | Array]) -> None:
    \"\"\"Process nested JSON with mixed types.\"\"\"
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{key}: Object with {len(value)} keys")
        elif isinstance(value, list):
            print(f"{key}: Array with {len(value)} items")
        else:
            print(f"{key}: {value}")

json_str = '''
{
    "config": {"debug": true, "port": 8080},
    "items": [1, 2, 3],
    "name": "App"
}
'''
process_nested(json.loads(json_str))
```

### Type-Safe JSON Schema Validation

```python
from optype.json import Value, Object, Array
import json
from typing import TypeGuard

def is_user_array(data: Value) -> TypeGuard[Array[Object]]:
    \"\"\"Validate user array schema.\"\"\"
    if not isinstance(data, list):
        return False
    
    for item in data:
        if not isinstance(item, dict):
            return False
        if "id" not in item or "name" not in item:
            return False
    
    return True

# Use with type narrowing
data = json.loads('[{"id": 1, "name": "Alice"}]')
if is_user_array(data):
    # Type checker knows data is Array[Object]
    for user in data:
        print(user["name"])
```

## Primitive JSON Types

The base `Value` and `AnyValue` types include:

- **null**: `None`
- **boolean**: `bool`
- **number**: `int`, `float`
- **string**: `str`
- **array**: `list` (with recursive `Value`/`AnyValue` elements)
- **object**: `dict` (with `str` keys and recursive values)

## Type Parameters

- **`V` (Value)**: Covariant - element type for arrays/objects
  - `Value` for output (from `json.load`)
  - `AnyValue` for input (to `json.dump`)

## Common Patterns

### API Response Type

```python
from optype.json import Object, Array
from typing import TypeAlias

# Define your API types
User: TypeAlias = Object  # {"id": int, "name": str, ...}
UserList: TypeAlias = Array[User]
ApiResponse: TypeAlias = Object  # {"data": UserList, "status": str}
```

### Safe JSON Access

```python
from optype.json import Object, Value

def get_nested(obj: Object, *keys: str, default: Value = None) -> Value:
    \"\"\"Safely access nested JSON object.\"\"\"
    current: Value = obj
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current

data = {"user": {"profile": {"name": "Alice"}}}
name = get_nested(data, "user", "profile", "name")  # "Alice"
missing = get_nested(data, "user", "missing", "key")  # None
```

## Related Types

- **[IO](io.md)**: File I/O protocols
- **[Pickle](pickle.md)**: Binary serialization
- **[String](string.md)**: String literal types
