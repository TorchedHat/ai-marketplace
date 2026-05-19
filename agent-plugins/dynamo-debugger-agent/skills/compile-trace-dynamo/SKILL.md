---
name: compile-trace-dynamo
description: Debug PyTorch Dynamo stage - bytecode capture, FX graph construction, graph breaks, and pre-grad passes. Covers TORCH_LOGS for dynamo/graph_breaks/pre_grad_graphs, interpreting FX graph files, understanding graph break reasons, and pre-grad fusion patterns (Conv-BN, split-cat). Load after compile-bisect indicates backend='eager'.
---

# Tracing Dynamo Stage - FX Graph Capture

How to trace and debug Dynamo bytecode capture, graph breaks, and pre-grad FX passes.

## Table of Contents

1. [Stage Overview](#stage-overview)
2. [What to Trace](#what-to-trace)
3. [Logging Setup](#logging-setup)
4. [Output Files](#output-files)
5. [Pre-Grad FX Passes](#pre-grad-fx-passes)
6. [Debugging Workflows](#debugging-workflows)
7. [Common Issues](#common-issues)

---

## Stage Overview

**Dynamo Stage** = Python bytecode capture → FX graph (aten ops) → Pre-grad passes

**What it does**:
- Intercepts Python execution via PEP 523 frame evaluation
- Symbolically executes bytecode to build FX graphs
- Identifies graph breaks (untraceable code)
- Runs pre-grad optimization passes

**For detailed Dynamo mechanics**: See [pytorch-dynamo skill](../pytorch-dynamo/SKILL.md)

**Pipeline Position**:
```
Python → Dynamo (FX + aten ops) → [Pre-Grad Passes] → AOT → Inductor
```

---

## What to Trace

### Trace When...

**Graph breaks**:
- Multiple small compiled graphs instead of one large graph
- Performance degradation from breaks
- Understanding why code isn't fully traced

**Unsupported operations**:
- Compilation failures
- Silent fallbacks to eager mode
- Unexpected behavior after compilation

**Pre-grad optimizations**:
- Conv-BN fusion not happening
- Split-cat patterns not optimizing
- Understanding FX-level transformations

---

## Logging Setup

### Basic Logging

**Minimal** (graph breaks only):
```bash
TORCH_LOGS="graph_breaks" python script.py
```

**Standard** (Dynamo + breaks):
```bash
TORCH_LOGS="dynamo,graph_breaks" python script.py
```

**Comprehensive** (including FX graphs):
```bash
TORCH_LOGS="dynamo,graph_breaks,graph_code,pre_grad_graphs" python script.py
```

### Available Loggers

| Logger | What It Shows | When to Use |
|--------|---------------|-------------|
| `dynamo` | All Dynamo debug output | General debugging |
| `graph_breaks` | Why and where breaks occur | Minimizing graph breaks |
| `graph_code` | Readable FX graph structure | Understanding captured graph |
| `guards` | Generated guards | Dynamic shape issues |
| `recompiles` | Recompilation reasons | Cache misses, performance |
| `pre_grad_graphs` | FX graphs before/after pre-grad | Pre-grad pass effects |
| `bytecode` | Bytecode transformations | Deep Dynamo debugging |

### Programmatic Logging

```python
import os
os.environ['TORCH_LOGS'] = 'dynamo,graph_breaks'

import torch._inductor.config as config
config.debug = True  # Enable debug file output
```

---

## Output Files

### Location

**Default**: `/tmp/torchinductor_$USER/`  
**Custom**: Set `TORCH_COMPILE_DEBUG_DIR=/path/to/dir`

### Generated Files

**1. FX Graph Files**

```
fx_graph_runnable.py         # Standalone reproduction script
fx_graph_readable.py         # Human-readable graph (before passes)
fx_graph_transformed.py      # Graph after pre-grad passes
```

**How to use**:
```bash
# Run reproduction script
python /tmp/torchinductor_$USER/fx_graph_runnable.py

# Compare before/after
diff fx_graph_readable.py fx_graph_transformed.py
```

**2. Graph Break Logs**

Console output shows:
```
Graph break: print(y) 
  Reason: call_function print in skip list
  User code: /path/to/file.py:5 in fn
  Graph Count: 2 (compilation split into multiple graphs)
```

### Interpreting FX Graphs

**Structure**:
```python
graph():
    %x : torch.Tensor = placeholder[target=x]  # Input
    %relu : torch.Tensor = call_function[      # Operation
        target=torch.ops.aten.relu.default
    ](args = (%x,))
    %add : torch.Tensor = call_function[
        target=torch.ops.aten.add.Tensor
    ](args = (%relu, 1))
    return add                                  # Output
```

**Node types**:
- `placeholder` - Function inputs
- `call_function` - ATen ops (most operations)
- `call_module` - nn.Module calls
- `get_attr` - Parameter/buffer access
- `output` - Return values

**What to look for**:
- Number of nodes (complexity)
- ATen ops used (lowering targets)
- Graph structure (dependencies, parallelism)

---

## Pre-Grad FX Passes

### When They Run

**After**: Dynamo capture  
**Before**: AOT Autograd (training) or Inductor (inference)

**Purpose**: Optimize FX graph at aten-op level

### Logging Pre-Grad Passes

```bash
TORCH_LOGS="pre_grad_graphs" python script.py
```

**Shows**:
- FX graph before passes
- Each pass applied
- FX graph after passes
- What changed

### Common Passes

| Pass | What It Does | How to Verify |
|------|--------------|---------------|
| Conv-BN Fusion | Folds BatchNorm into Conv weights | Check if `batch_norm` node removed |
| Split-Cat Elimination | Removes redundant split/cat | Check if `split`/`cat` pair eliminated |
| Normalization | NumPy compatibility rewrites | Compare before/after graph |
| Group Batch Fusion | Batches operations together | Look for combined ops |

### Verifying Pass Effects

**Before Pre-Grad** (`fx_graph_readable.py`):
```python
%conv : Tensor = call_function[target=aten.conv2d](args = (%x, %weight))
%bn : Tensor = call_function[target=aten.batch_norm](args = (%conv, ...))
```

**After Pre-Grad** (`fx_graph_transformed.py`):
```python
%fused_conv : Tensor = call_function[target=aten.conv2d](
    args = (%x, %fused_weight, %fused_bias)  # BN folded in
)
# batch_norm node eliminated
```

---

## Debugging Workflows

### Workflow 1: Minimize Graph Breaks

**Goal**: Reduce number of compiled graphs

**Steps**:
1. Enable logging:
   ```bash
   TORCH_LOGS="graph_breaks" python script.py
   ```

2. Identify break locations:
   ```
   Graph break: <operation>
     Reason: <why it broke>
     User code: <file:line>
   ```

3. Fix each break:
   - **Dynamic control**: Replace with `torch.cond()`
   - **I/O ops**: Move outside compiled region
   - **Custom ops**: Use `torch._dynamo.allow_in_graph()`

4. Verify with fullgraph mode:
   ```python
   @torch.compile(fullgraph=True)  # Errors if any breaks
   def fn(x):
       ...
   ```

### Workflow 2: Debug Unsupported Operation

**Symptom**: `UnsupportedOperationError` or silent graph break

**Steps**:
1. Enable verbose logging:
   ```bash
   TORCH_LOGS="dynamo,graph_breaks" python script.py
   ```

2. Locate unsupported op in graph break message

3. Check if op should be supported:
   - Look in `torch/_dynamo/skipfiles.py` for skip lists
   - Check `torch/_dynamo/variables/` for handler

4. Solutions:
   - **Rewrite** using supported ops
   - **Allow** via `torch._dynamo.allow_in_graph(fn)`
   - **Skip** via explicit `torch._dynamo.graph_break()`

### Workflow 3: Verify Pre-Grad Optimization

**Goal**: Confirm expected optimization happened

**Steps**:
1. Enable pre-grad logging:
   ```bash
   TORCH_LOGS="pre_grad_graphs" python script.py
   ```

2. Check before graph (`fx_graph_readable.py`):
   ```bash
   grep "batch_norm\|conv2d" fx_graph_readable.py
   ```

3. Check after graph (`fx_graph_transformed.py`):
   ```bash
   grep "batch_norm\|conv2d" fx_graph_transformed.py
   ```

4. Verify pattern eliminated:
   - Conv-BN: `batch_norm` should be gone
   - Split-Cat: `split`/`cat` pair should be gone

### Workflow 4: Debug Dynamic Shapes

**Symptom**: Guards failing, recompilation, or wrong shapes

**Steps**:
1. Enable guard logging:
   ```bash
   TORCH_LOGS="guards,recompiles" python script.py
   ```

2. Check generated guards:
   ```
   Guard: tensor 'x' shape[0] == 10
   Guard: tensor 'x' shape[1] == 20
   ```

3. Identify problematic guards:
   - Too specific → causes recompilation
   - Too loose → wrong specialization

4. Fix with shape hints:
   ```python
   x = torch.randn(10, 20)
   torch._dynamo.mark_dynamic(x, 0)  # Dimension 0 is dynamic
   ```

---

## Common Issues

### Issue: Too Many Graph Breaks

**Symptom**: Many small graphs, slow performance

**Debug**:
```bash
TORCH_LOGS="graph_breaks" python script.py | grep "Graph break" | wc -l
```

**Solutions**:
- Refactor to minimize dynamic control flow
- Use `torch.cond()` for conditional execution
- Move non-traceable code outside `@torch.compile`
- Mark functions as traceable: `torch._dynamo.allow_in_graph(fn)`

### Issue: Slow Compilation

**Symptom**: Long wait on first execution

**Debug**:
```bash
TORCH_COMPILE_DYNAMO_PROFILER=1 python script.py
```

**Solutions**:
- Use compilation cache (enabled by default)
- Reduce graph complexity
- Use `mode="reduce-overhead"` for faster compile:
  ```python
  torch.compile(fn, mode="reduce-overhead")
  ```

### Issue: Conv-BN Not Fusing

**Symptom**: Expected fusion didn't happen

**Debug**:
```bash
TORCH_LOGS="pre_grad_graphs" python script.py
# Check if both conv and batch_norm still in transformed graph
```

**Common causes**:
- Model in training mode (use `model.eval()`)
- Conv and BN not consecutive in graph
- Unsupported conv or BN variant

**Fix**:
```python
model.eval()  # Pre-grad conv-bn fusion only in eval mode
```

### Issue: Wrong Output After Compilation

**Symptom**: Compiled function produces incorrect result

**Debug Steps**:
1. Compare with eager backend:
   ```python
   @torch.compile(backend="eager")
   def fn(x):
       ...
   ```

2. Check FX graph structure:
   ```bash
   TORCH_LOGS="graph_code" python script.py
   # Verify captured graph matches expectations
   ```

3. Verify guards:
   ```bash
   TORCH_LOGS="guards" python script.py
   ```

4. Check for mutations:
   - In-place ops may not be properly tracked
   - Verify with `@torch.compile(fullgraph=True)`

---

## Quick Reference

### Essential Commands

```bash
# Basic tracing
TORCH_LOGS="dynamo,graph_breaks" python script.py

# With FX graphs
TORCH_LOGS="dynamo,graph_breaks,graph_code" python script.py

# Include pre-grad passes
TORCH_LOGS="dynamo,graph_breaks,pre_grad_graphs" python script.py

# Full debug
TORCH_LOGS="dynamo,graph_breaks,graph_code,guards,recompiles,pre_grad_graphs" python script.py
```

### Output Files

```bash
# View FX graph
cat /tmp/torchinductor_$USER/fx_graph_readable.py

# Run reproduction script
python /tmp/torchinductor_$USER/fx_graph_runnable.py

# Compare before/after passes
diff /tmp/torchinductor_$USER/fx_graph_{readable,transformed}.py
```

### Programmatic Debugging

```python
# Count compilations
from torch._dynamo.testing import CompileCounter
cnt = CompileCounter()
compiled_fn = torch.compile(fn, backend=cnt)

# Explicit graph break
torch._dynamo.graph_break()

# Force fullgraph (error on break)
@torch.compile(fullgraph=True)
def fn(x): ...

# Compile-time breakpoint
import torch._dynamo.comptime as comptime
comptime.breakpoint()  # Drops into pdb during compile
```

---

## Next Stage

**After Dynamo Stage**: Load `compile-trace-aot` skill - Tracing AOT Autograd transformations

**Or**: Load `compile-trace-inductor` skill - Skip to Inductor stage

---

**Reference**: See `compile-overview` skill for complete pipeline context.
