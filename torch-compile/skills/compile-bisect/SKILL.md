---
name: compile-bisect
description: Use PyTorch's compiler bisector to automatically find which backend/subsystem/operation causes compilation failures. Binary searches through backends (eager → aot_eager → inductor) and subsystems (passes, lowerings, etc.) to pinpoint exact failing operations. Outputs backend/subsystem/debug_info that routes you to the right stage-specific skill (compile-trace-dynamo for eager, compile-trace-aot for aot_*, compile-trace-inductor for inductor).
---

# Compiler Bisector

Automatically isolate compilation failures to exact backend/subsystem/operation using binary search.

## What It Outputs

```
backend='inductor'              # Which compilation stage failed
subsystem='lowerings'           # Which subsystem within that stage
debug_info='aten.argmin.default'  # Exact operation that failed
```

**When to use**: Any compilation failure (crash, wrong output, assertion error).

## Usage

### CLI Mode (Recommended)

```bash
python -m torch._inductor.compiler_bisector run python repro.py
```

**Test script requirements:**
- Exit code 0 = PASS, 1 = FAIL
- Use `os.environ.get("TORCH_COMPILE_BACKEND", "inductor")` to select backend
- Call `torch._dynamo.reset()` at start for clean state
- Use `torch._inductor.utils.fresh_cache()` context manager to ensure clean compilation cache

**Minimal example:**
```python
import os, sys, torch
from torch._inductor.utils import fresh_cache

torch._dynamo.reset()
backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

with fresh_cache():
    @torch.compile(backend=backend)
    def fn(x):
        return x.sin().argmin()  # Your failing operation

    result = fn(torch.randn(10))
    expected = torch.randn(10).sin().argmin()
    sys.exit(0 if torch.equal(result, expected) else 1)
```


## Routing

Backend determines which skill to load:

| Backend | → Load Skill |
|---------|--------------|
| `eager` | `compile-trace-dynamo` + `pytorch-dynamo` |
| `aot_*` | `compile-trace-aot` |
| `inductor` | `compile-trace-inductor` + `pytorch-inductor` |

## Subsystems

If bisect reports a subsystem, it tells you what to focus on:

| Subsystem | What It Means |
|-----------|---------------|
| `lowerings` | Missing operator lowering (check `debug_info` for op name) |
| `pre_grad_passes` | Pre-grad optimization (Conv-BN fusion, split-cat, etc.) |
| `post_grad_passes` | Post-grad optimization (GEMM fusion, etc.) |
| `cudagraphs` | CUDA graphs backend wrapper |
| `decomposition` | Operator decomposition rules |
| `cse` | Common subexpression elimination |

## Example Workflow

```bash
$ python -m torch._inductor.compiler_bisector run python repro.py
# Output: backend='inductor', subsystem='lowerings', debug_info='aten.argmin.default'

# Next steps:
# 1. Load compile-trace-inductor + pytorch-inductor (see routing table)
# 2. Check torch/_inductor/lowering.py for aten.argmin lowering (subsystem + debug_info)
# 3. Add or fix the lowering registration
# 4. Run bisector again to verify fix
```

## Recommended Workflow

```
Bisect → Trace → Fix → Verify

1. Bisect:   Find exact failure point (this tool)
2. Trace:    Use TORCH_LOGS guided by bisect result
3. Fix:      Edit code based on debug_info
4. Verify:   Run bisector again (should report no issue)
```

**Why bisect-first**: Pinpoints the exact operation before you enable logging, making trace output much smaller and more focused.

## Tips

- **Keep test fast**: Bisector runs test multiple times; aim for <10s per run
- **Minimal reproducer**: Remove unrelated code to speed up bisection
- **Deterministic failures**: Bisector requires consistent pass/fail behavior
- **Check debug_info**: For lowerings, shows exact aten op; for passes, shows pass name
- **Use specific backends**: Can bisect specific stages by setting `TORCH_COMPILE_BACKEND` manually

## Output Interpretation

**Backend identified:**
```
Moving to the next system: inductor
The issue is in the inductor system.
```

**Subsystem isolated:**
```
Disabling lowerings fixed the issue.
Starting bisect by getting upper bound.
```

**Exact operation found:**
```
Binary search completed for inductor - lowerings. The bisect number is 42.
Debug info: aten.argmin.default
```

## Limitations

- Only works with deterministic failures
- Requires clear pass/fail criteria
- Cannot bisect non-reproducible issues
- Test script must handle backend selection via env var

---

**Next Steps**: Use routing table above to load the appropriate stage skill based on bisect output. Run `TORCH_LOGS` with flags specific to that stage.
