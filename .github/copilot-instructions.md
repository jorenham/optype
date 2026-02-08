# GitHub Copilot Instructions for optype

This document provides GitHub Copilot with context about the optype project to help
generate better suggestions and code.

## Project Overview

`optype` is a Python library that provides building blocks for precise & flexible type
hints. It focuses on typing utilities, and has optional elaborate NumPy support.
It is compatible with all modern static type-checkers, and also works with runtime
type-checkers like beartype.

## Project Structure

- **Core package**: `optype/` - Main library code with type definitions and utilities
- **Tests**: `tests/` - Comprehensive test suite using pytest
- **Examples**: `examples/` - Usage examples and demonstrations

## Key Technologies & Dependencies

- **Python versions**: 3.11+
- **Build system**: uv (build-backend: uv_build)
- **Testing**: pytest with doctests enabled
- **Type checking**: mypy, basedpyright (a backwards compatible pyright fork with
  additional features)
- **Linting & formatting**: ruff, dprint
- **Git hooks manager**: lefthook
- **Task runner**: tox
- **Optional dependencies**: numpy>=1.26 (minimum version follows
  [SPEC 0](https://scientific-python.org/specs/spec-0000/))

## Code Style & Conventions

### Python Style

- Follow ruff configuration in `pyproject.toml`
- Use type hints extensively (this is a typing library)
- Enable strict type checking (mypy strict mode)
- Line ending: LF
- Preview features enabled in ruff

### Import Conventions

```python
import ctypes as ct
import datetime as dt

import optype as op
import optype.typing as opt

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.numpy.compat as npc
```

### File Organization

- Protocols for core Python functionality in `optype/_core/` (`Just*`, `Can*`, `Has*`,
  `Does*`, and `do_*`)
- NumPy-specific types in `optype/numpy/`
- Standard library types in appropriate modules (e.g., `optype/copy.py` and
  `optype/pickle.py`)
- Tests mirror the package structure in `tests/`

## Development Workflow

### Setup

```bash
uv sync  # Install all dependencies
uvx lefthook install  # Set up git hooks
```

### Testing

```bash
uv run pytest  # Run tests only
uvx tox p  # Run all tests on multiple python versions in parallel
```

### Code Quality

- All code must pass ruff linting and formatting
- Type checking with both mypy and basedpyright
- Doctests are enabled and must pass
- Use `dprint` for formatting TOML, YAML, JSON, and Markdown files
- Formatting happens automatically on commit via git hooks (managed by lefthook)

### Running Tools

Always use `uv run` to execute Python scripts and packages:

- Type checking: `uv run basedpyright` and `uv mypy .`
- Testing: `uv run pytest`
- Linting: `uv run ruff check --fix`
- Formatting: `uv run ruff format`
- TOML/YAML/JSON/Markdown formatting: `uv run dprint fmt`
- Repository review: `uv run repo-review .`

### Specific Rules

- No `typing.Any` expressions (disallow_any_expr = false in mypy, but avoid in practice)
- Don't use `from __future__ import annotations`, as it can interfere with runtime
  type-checkers such as beartype.

## Common Patterns

### Type Definitions

- Use precise type hints and avoid `Any` when possible
- Leverage `typing_extensions` for newer features on older Python versions
- Create protocol classes for structural typing
- Use `TypeVar` and generics appropriately

### Testing

- Write both unit tests and doctests
- Test across all supported Python versions
- Include examples in docstrings that serve as tests
- Use beartype for runtime type checking in tests

### Documentation

- Primary documentation in `README.md`
- Use clear, descriptive docstrings
- Include usage examples in docstrings
- Format markdown with dprint

## File Patterns to Follow

### New Type Modules

```python
"""Module docstring describing the types provided."""

import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 13):
    from typing import ...
else:
    from typing_extensions import ...

if TYPE_CHECKING:
    # Import only needed for static type checking

# Type definitions here
```

### Test Files

```python
"""Test module following pytest conventions."""

import pytest
import optype as op

def test_functionality():
    """Test description."""
    # Test implementation
```

## Special Considerations

- This is a typing-focused library, so type correctness is paramount
- Support multiple Python versions (3.11-3.14)
- Optional numpy integration requires careful handling
- Performance is important for type checking tools
- Maintain backward compatibility within major versions

## Error Handling

- Use specific exception types
- Provide clear error messages (max 42 characters for ruff)
- Handle edge cases in type definitions gracefully

## When Contributing

1. Ensure all type checkers pass (mypy, basedpyright)
2. Run the full test suite with `uvx tox p`
3. Follow the existing code style and patterns
4. Add appropriate tests for new functionality
5. Update the `README.md` documentation as needed
6. Consider cross-Python version compatibility

## Dependency Policy

**No new dependencies should be installed**, not even development dependencies.
If you think you need a new dependency at some point, you should reconsider your
approach instead. The project already has all necessary tools configured in the
dependency groups.
