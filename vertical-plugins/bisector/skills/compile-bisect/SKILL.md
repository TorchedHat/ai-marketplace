---
name: compile-bisect
description: Use PyTorch's compiler bisector to automatically find which backend/subsystem/operation causes compilation failures. Binary searches through backends (eager → aot_eager → inductor) and subsystems (passes, lowerings, etc.) to pinpoint exact failing operations. Returns BisectionResult that routes you to the right stage-specific skill (compile-trace-dynamo for eager, compile-trace-aot for aot_*, compile-trace-inductor for inductor).
---

# Compiler Bisector - Isolate Compilation Failures

Quick guide for using PyTorch's compiler bisector to automatically identify which backend/subsystem causes compilation failures.

## Recommended Workflow: Bisect First, Then Trace

**Best practice**: Start with bisector to find what fails, then use targeted tracing to understand why.

```
1. Bisect   → Find which backend/subsystem/operation fails (fast, automated)
2. Trace    → Understand why that specific operation fails (targeted logging)
3. Fix      → Address the identified issue
4. Verify   → Run bisector again to confirm fix
```

**Why bisect-first**:
- ✅ Fast automated binary search vs manual log analysis
- ✅ Pinpoints exact failing operation before deep diving
- ✅ Avoids tracing everything when you only need one subsystem
- ✅ Gives you the right TORCH_LOGS flags to use

## What It Does

Automatically bisects through compilation backends and subsystems to find the exact point where a failure occurs:
1. Tests backends in order: `eager` → `aot_eager` → `aot_eager_decomp_partition` → `inductor`
2. When a backend fails, tests its subsystems (passes, lowerings, etc.)
3. Binary searches to find the exact operation/pass that triggers the issue

## Quick Start: Two Usage Modes

**Recommended for agents**: `run` command (fully automatic)
**Idiomatic in code**: `do_bisect()` (programmatic control)

### Mode 1: Automatic Bisection (CLI `run`)

**Easiest for creating reproducers** - Let the bisector run your test automatically:

```bash
python -m torch._inductor.compiler_bisector run python repro.py
```

Your test script should:
- Return exit code 0 on success (PASS)
- Return exit code 1 on failure (FAIL)
- Use `os.environ.get("TORCH_COMPILE_BACKEND", "inductor")` to select backend

**Example test script**:
```python
import os
import sys
import torch

def main():
    torch._dynamo.reset()
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    
    @torch.compile(backend=backend)
    def fn(x):
        return x.sin().argmin()  # Your failing operation
    
    x = torch.randn(10)
    result = fn(x)
    expected = x.sin().argmin()  # Eager reference
    
    if torch.equal(result, expected):
        return 0  # PASS
    else:
        print(f"FAIL: got {result}, expected {expected}")
        return 1  # FAIL

if __name__ == "__main__":
    sys.exit(main())
```

## Backends and Subsystems

The bisector tests in this order:

1. **`eager`** - Dynamo without AOTAutograd (no subsystems)
2. **`aot_eager`** - Dynamo with AOTAutograd (no subsystems)
3. **`aot_eager_decomp_partition`** - With decompositions and partitioner
   - Subsystems: `cse`, `decomposition`
4. **`aot_eager_decomp_partition_crossref`** - With CrossRefFakeMode
5. **`inductor`** - Full Inductor compiler
   - Subsystems: `pre_grad_passes`, `joint_graph_passes`, `post_grad_passes`
   - Config toggles: `fallback_random`, `emulate_precision_casts`, `layout_optimization`, `comprehensive_padding`
   - Backend: `cudagraphs`, `lowerings`

## Understanding Output

**Backend identification**:
```
Moving to the next system: inductor
The issue is in the inductor system. Moving to the first subsystem: pre_grad_passes
```

**Subsystem isolation**:
```
Disabling lowerings fixed the issue.
Starting bisect by getting upper bound.
Upper bound of 127 found for inductor.
```

**Exact operation found**:
```
Binary search completed for inductor - lowerings. The bisect number is 42.
Debug info: aten.argmin.default
```

**Final result**:
```
Bisection complete: BisectionResult(
    backend='inductor',
    subsystem='lowerings',
    bisect_number=42,
    debug_info='aten.argmin.default'
)
```

### Mode 2: Programmatic Bisection (`do_bisect`)

**Idiomatic Python usage** - Call `do_bisect()` directly in your code:

```python
from torch._inductor.compiler_bisector import CompilerBisector

def test_function() -> bool:
    """Return True on success, False on failure"""
    try:
        result = torch.compile(fn)(x)
        expected = fn(x)
        return torch.allclose(result, expected)
    except Exception:
        return False

# Run bisection - this is the idiomatic way per docstring
result = CompilerBisector.do_bisect(test_function, cli_interface=False)

if result:
    print(f"Found issue in {result.backend}")
    if result.subsystem:
        print(f"  Subsystem: {result.subsystem}")
        print(f"  Bisect number: {result.bisect_number}")
        print(f"  Debug info: {result.debug_info}")
```

**Why use this**:
- Programmatic control in Python scripts
- Direct access to `BisectionResult` object
- Can integrate into test suites
- More flexible than CLI

## Recommended Workflow

**Bisect → Trace → Fix → Verify**

```bash
# Step 1: Bisect to find the problem
python -m torch._inductor.compiler_bisector run python repro.py
# Output: backend='inductor', subsystem='lowerings', bisect_number=42
#         debug_info='aten.argmin.default'

# Step 2: Now trace that specific area
TORCH_LOGS="fusion,schedule" python repro.py
# Look for aten.argmin.default in lowerings

# Step 3: Fix the issue
# Edit torch/_inductor/lowering.py - fix argmin lowering

# Step 4: Verify
python -m torch._inductor.compiler_bisector run python repro.py
# Output: Bisection complete: no issue found
```

## Integration with Compile Trace

**Recommended debugging flow** (bisect-first):

1. **Bisect** to find exact operation:
   ```bash
   python -m torch._inductor.compiler_bisector run python repro.py
   # Output: backend='inductor', subsystem='lowerings', bisect_number=42
   #         debug_info='aten.argmin.default'
   ```

2. **Trace** that specific area (now you know what to trace):
   ```bash
   # Bisector told us it's inductor lowerings, so trace that:
   TORCH_LOGS="fusion,schedule,output_code" python repro.py
   # Focus on aten.argmin.default lowering in the logs
   ```

3. **Fix** the identified operation:
   ```python
   # Edit torch/_inductor/lowering.py
   # Fix aten.argmin lowering at line ~6700
   ```

4. **Verify** fix works:
   ```bash
   python -m torch._inductor.compiler_bisector run python repro.py
   # Output: Bisection complete: no issue found
   ```

**Why this order**:
- Bisector gives you `backend='inductor'` → tells you which TORCH_LOGS to enable
- Bisector gives you `subsystem='lowerings'` → tells you which stage to focus on
- Bisector gives you `debug_info='aten.argmin.default'` → tells you which op to search for
- Trace becomes targeted instead of exploratory

## Tips

**Write robust test functions**:
- Use `torch._dynamo.reset()` at the start
- Test exact equality when possible (`torch.equal`)
- Use tolerance for numerics (`torch.allclose`)
- Return clear exit codes (0=pass, 1=fail)

**Narrow down first**:
- Start with minimal reproducer
- Remove unnecessary operations
- Isolate the specific failing pattern

**Understand debug info**:
- For `lowerings`: Shows the specific aten op that failed
- For `passes`: Shows which optimization pass caused issue
- For config toggles: Shows which config option matters

**Performance**:
- Bisection can take time (multiple test runs)
- Each run recompiles, so use small inputs
- Test function should be fast (<10s ideal)

## Limitations

- Only works with deterministic failures
- Requires clear pass/fail criteria
- Can't bisect non-reproducible issues
- Manual intervention needed for some edge cases

## Next Steps

After bisection identifies the issue:

1. **For lowering issues**: Check `torch/_inductor/lowering.py`
2. **For pass issues**: Check pass implementations in relevant stage
3. **For config issues**: Check why that config matters
4. **Load stage-specific skills**: `compile-trace-inductor`, `compile-trace-aot`, `compile-trace-dynamo`

---

**Reference**: See `compile-overview` skill for complete pipeline context and routing guidance.
