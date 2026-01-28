# Random

Type annotations for NumPy's random number generation.

## Overview

The `optype.numpy.random` submodule provides [SPEC 7](https://scientific-python.org/specs/spec-0007/)-compatible type aliases for NumPy's random number generation interfaces. It handles both the modern `numpy.random.Generator` API and legacy `numpy.random.RandomState`.

## Type Aliases

### RNG (Random Number Generator)

A union type accepting both modern and legacy random generators:

```python
type RNG = np.random.Generator | np.random.RandomState
```

**Purpose**: Accept either the new or legacy NumPy random API

**Usage**:

```python
import numpy as np
import optype.numpy.random as onr

def generate_random(rng: onr.RNG, size: int) -> np.ndarray:
    """Works with both Generator and RandomState."""
    if isinstance(rng, np.random.Generator):
        return rng.standard_normal(size)
    else:  # RandomState
        return rng.standard_normal(size)

# Works with both
gen = np.random.default_rng(42)
legacy_rng = np.random.RandomState(42)

generate_random(gen, 10)
generate_random(legacy_rng, 10)
```

### ToSeed

Accepts seed values for RNG initialization:

```python
type ToSeed = (
    int
    | np.integer[Any]
    | Sequence[int | np.integer[Any]]
    | np.ndarray[Any, np.dtype[np.uint32 | np.uint64]]
    | np.random.SeedSequence
)
```

**Purpose**: Type-safe seed specification

**Accepted types**:

- **Integer scalars**: `42`, `np.int64(42)`
- **Sequences**: `[1, 2, 3]`, `(42,)`
- **Arrays**: `np.array([1, 2, 3], dtype=np.uint32)`
- **SeedSequence**: `np.random.SeedSequence(42)`

**Usage**:

```python
import numpy as np
import optype.numpy.random as onr

def create_rng(seed: onr.ToSeed) -> np.random.Generator:
    """Create generator with flexible seed type."""
    return np.random.default_rng(seed)

# All these work
rng1 = create_rng(42)
rng2 = create_rng([1, 2, 3])
rng3 = create_rng(np.random.SeedSequence(42))
rng4 = create_rng(np.array([1, 2], dtype=np.uint32))
```

### ToRNG (To Random Number Generator)

The most flexible type that accepts anything that can become an RNG:

```python
type ToRNG = (
    RNG
    | ToSeed
    | np.random.BitGenerator
)
```

**Purpose**: Accept seeds, generators, and bit generators uniformly

**Usage**:

```python
import numpy as np
import optype.numpy.random as onr

def flexible_random(
    rng: onr.ToRNG,
    size: int,
) -> np.ndarray:
    """Accept any RNG-compatible input."""
    gen = np.random.default_rng(rng)
    return gen.standard_normal(size)

# All these work
flexible_random(42, 10)                              # ✓ Integer seed
flexible_random(np.random.default_rng(42), 10)      # ✓ Generator
flexible_random(np.random.PCG64(42), 10)            # ✓ BitGenerator
flexible_random(np.random.SeedSequence(42), 10)     # ✓ SeedSequence
```

## Modern vs Legacy API

### Modern API (Recommended)

```python
import numpy as np
import optype.numpy.random as onr

def modern_random(rng: onr.RNG | int = None) -> float:
    """Use modern Generator API."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    return float(rng.standard_normal())

# Usage
x = modern_random(42)
gen = np.random.default_rng(42)
y = modern_random(gen)
```

### Legacy API (Compatibility)

```python
import numpy as np
import optype.numpy.random as onr

def legacy_random(seed: int) -> float:
    """Use legacy RandomState API."""
    rng = np.random.RandomState(seed)
    return float(rng.standard_normal())

# Usage
x = legacy_random(42)
```

## BitGenerator Types

Specific bit generators for different use cases:

```python
import numpy as np

# PCG64: Modern default generator (fast, high quality)
bg_pcg64 = np.random.PCG64(seed=42)

# MT19937: Mersenne Twister (widespread, slower)
bg_mt19937 = np.random.MT19937(seed=42)

# Philox: Parallel-friendly
bg_philox = np.random.Philox(seed=42)

# SFC64: Simple Fast Counting
bg_sfc64 = np.random.SFC64(seed=42)

# Use with Generator
gen_pcg = np.random.Generator(bg_pcg64)
```

## Type-Safe Function Signatures

```python
import numpy as np
import optype.numpy.random as onr
from optype.numpy import Array1D, AnyFloat64Array

def bootstrap_sample(
    data: AnyFloat64Array,
    rng: onr.ToRNG = None,
    n_samples: int = 1000,
) -> Array1D[np.float64]:
    """Bootstrap resampling with flexible RNG input."""
    gen = np.random.default_rng(rng)
    n = len(data)
    indices = gen.integers(0, n, size=n_samples)
    return np.asarray(data)[indices]

# Usage - flexible seed input
samples1 = bootstrap_sample([1, 2, 3, 4, 5], rng=42)
samples2 = bootstrap_sample([1, 2, 3, 4, 5], rng=[1, 2, 3])

# Or with explicit generator
gen = np.random.default_rng(42)
samples3 = bootstrap_sample([1, 2, 3, 4, 5], rng=gen)
```

## SPEC 7 Compatibility

This module follows [SPEC 7: Accessibility of type hints for NumPy's type annotations](https://scientific-python.org/specs/spec-0007/), ensuring consistent random number generator typing across the scientific Python ecosystem.

## Common Patterns

### Seeding for Reproducibility

```python
import numpy as np
import optype.numpy.random as onr

def reproducible_analysis(
    seed: onr.ToSeed = 42,
) -> dict:
    """Ensure reproducible results with type-safe seed."""
    rng = np.random.default_rng(seed)
    return {
        'normal': rng.standard_normal(10),
        'uniform': rng.uniform(0, 1, 10),
        'integers': rng.integers(0, 100, 10),
    }
```

### Parallel RNG Streams

```python
import numpy as np
import optype.numpy.random as onr

def parallel_random(
    n_workers: int,
    seed: onr.ToSeed = 42,
) -> list[onr.RNG]:
    """Create independent RNG streams for parallel workers."""
    rng = np.random.default_rng(seed)
    # Use SeedSequence to spawn independent generators
    ss = np.random.SeedSequence(seed)
    return [
        np.random.Generator(rng.spawn(1)[0])
        for _ in range(n_workers)
    ]
```

## References

- [SPEC 7: Random Number Generation](https://scientific-python.org/specs/spec-0007/)
- [NumPy Random Documentation](https://numpy.org/doc/stable/reference/random/)
- [NEP 19: Random Number Generator Policy](https://numpy.org/neps/nep-0019-rng-policy.html)

## Related Modules

- **[Aliases](aliases.md)**: Array type aliases
- **[Array-likes](array-likes.md)**: Type-safe array conversion
- **[Low-level](low-level.md)**: Low-level NumPy protocols
