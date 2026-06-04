---
name: compile-trace-aot
description: Debug PyTorch AOT Autograd stage - functionalization, decompositions, IR transformations, joint forward+backward graph (when requires_grad=True), partitioning/recomputation, and post-grad passes. Use for tracing AOT stage and understanding decomposition application.
---

# Tracing AOT Autograd Stage - Training Transformations

How to trace and debug AOT Autograd: functionalization, joint graph creation, partitioning, and post-grad passes.

## Table of Contents

1. [Stage Overview](#stage-overview)
2. [When AOT Runs](#when-aot-runs)
3. [Logging Setup](#logging-setup)
4. [Output Files and Interpretation](#output-files-and-interpretation)
5. [Tracing Functionalization](#tracing-functionalization)
6. [Tracing Joint Graph](#tracing-joint-graph)
7. [Tracing Partitioning](#tracing-partitioning)
8. [Post-Grad FX Passes](#post-grad-fx-passes)
9. [Debugging Workflows](#debugging-workflows)
10. [Common Issues](#common-issues)

---

## Stage Overview

**AOT Autograd** = Ahead-of-Time autograd lowering (training-specific transformations)

**What it does**:
1. **Functionalization**: Removes mutations and aliases
2. **Joint Graph**: Combines forward + backward in one FX graph
3. **Partitioning**: Splits joint graph into separate forward/backward
4. **Post-Grad Passes**: Optimizes both graphs after partitioning

**Pipeline Position**:
```
Dynamo → [Pre-Grad] → AOT Autograd → [Post-Grad] → Inductor
                      ↓
               Functionalization
               Joint Graph
               Partitioning
```

**Key Location**: `torch/_functorch/aot_autograd.py`

---

## When AOT Runs

### Training vs Inference

**Training Path** (needs_autograd=True):
- Any output requires grad, OR
- Any input requires grad with mutations
- Creates both forward and backward graphs

**Inference Path** (needs_autograd=False):
- No gradients needed
- Skips joint graph, partitioning
- Only forward compilation

### How to Know If AOT Ran

**Check logs**:
```bash
TORCH_LOGS="aot" python script.py
```

**Output shows**:
```
[AOT] Compiling forward graph: model__0_forward_0
[AOT] Compiling backward graph: model__0_backward_0
```

**If AOT didn't run** (inference):
```
# No AOT messages, goes straight to Inductor
```

---

## Logging Setup

### Basic Logging

**Minimal** (AOT compilation info):
```bash
TORCH_LOGS="aot" python script.py
```

**Standard** (with graphs):
```bash
TORCH_LOGS="aot,aot_graphs" python script.py
```

**Comprehensive** (including joint graph):
```bash
TORCH_LOGS="aot,aot_graphs,aot_joint_graph,post_grad_graphs" python script.py
```

### Available Loggers

| Logger | What It Shows | When to Use |
|--------|---------------|-------------|
| `aot` | Basic AOT compilation tracking | Verify AOT ran |
| `aot_graphs` | Forward/backward graphs after partitioning | Understanding graph structure |
| `aot_joint_graph` | Combined forward+backward before split | Debugging partitioning |
| `post_grad_graphs` | FX graphs before/after post-grad passes | Pattern matching effects |

### Programmatic Setup

```python
import os
os.environ['TORCH_LOGS'] = 'aot,aot_graphs,aot_joint_graph'

import torch._inductor.config as config
config.debug = True
```

---

## Output Files and Interpretation

### File Naming Convention

**Format**: `{model_name}_{aot_id}__{graph_type}_{nth_graph}`

**Examples**:
```
model__0__forward_0.py          # First forward graph
model__0__backward_0.py         # First backward graph
model__0__joint_0.py            # Joint graph (if logged)
model__0__forward_transformed_0.py   # After post-grad passes
```

### Graph Structure

**Joint Graph** (before partitioning):
```python
graph():
    # Forward inputs (primals)
    %arg0 : Tensor = placeholder[target=arg0]
    %arg1 : Tensor = placeholder[target=arg1]

    # Forward computation
    %mul : Tensor = call_function[target=aten.mul](args = (%arg0, 2))
    %add : Tensor = call_function[target=aten.add](args = (%mul, %arg1))

    # Backward inputs (tangents)
    %tangent : Tensor = placeholder[target=tangent]

    # Backward computation
    %mul_grad : Tensor = call_function[target=aten.mul](args = (%tangent, 2))

    # Outputs: forward results + gradients
    return (add, mul_grad)
```

**Forward Graph** (after partitioning):
```python
graph():
    %x : Tensor = placeholder[target=x]
    %weight : Tensor = placeholder[target=weight]
    %mul : Tensor = call_function[target=aten.mul](args = (%x, %weight))
    %add : Tensor = call_function[target=aten.add](args = (%mul, 1))
    return (add, mul)  # Output + saved activations for backward
```

**Backward Graph** (after partitioning):
```python
graph():
    %saved_mul : Tensor = placeholder[target=saved_mul]  # From forward
    %grad_output : Tensor = placeholder[target=grad_output]
    %grad_mul : Tensor = call_function[target=aten.mul](args = (%grad_output, 1))
    %grad_weight : Tensor = call_function[target=aten.mul](args = (%grad_mul, %saved_mul))
    return (grad_weight,)
```

### What to Look For

**In Joint Graph**:
- Node metadata: `meta["partitioner_tag"]` = "is_forward" or "is_backward"
- Forward vs backward separation
- Which activations are saved

**In Partitioned Graphs**:
- Forward outputs include saved activations
- Backward inputs match saved activations
- Gradient flow correctness

---

## Tracing Functionalization

### What Functionalization Does

**Creates Core ATen IR** - removes mutations and aliases to produce functional graph.

**Before** (Full ATen IR):
```python
def f(x):
    x.mul_(2)      # In-place mutation
    return x.add(1)
```

**After** (Core ATen IR):
```python
def f(x):
    x_new = x * 2  # Functional
    return x_new + 1
# x.mul_() mutation tracked in metadata, applied at runtime
```

### How to Trace

**Logging** (captured in AOT graphs):
```bash
TORCH_LOGS="aot,aot_graphs" python script.py
```

**What to check**:
1. Graph has no in-place ops (no `mul_`, `add_`, etc.)
2. Mutations tracked in output metadata
3. Wrapper code copies mutations back

### Verifying Mutation Handling

**Check graph nodes**:
```bash
grep "mul_\|add_\|sub_" /tmp/torchinductor_$USER/model__*__forward_0.py
# Should find none (all converted to out-of-place)
```

**Check metadata** (in Python):
```python
# Graph outputs include mutation info
# Look for: return (output, mutated_input)
```

---

## Tracing Joint Graph

### What Is Joint Graph

**Joint Graph** = Forward + Backward traced together in single FX graph

**Purpose**:
- Trace backward pass via autograd.grad()
- Identify what to save for backward
- Enable cross-stage optimizations

### How to Trace

```bash
TORCH_LOGS="aot_joint_graph" python script.py
```

**Output**: `model__*__joint_*.py` file

### Interpreting Joint Graph

**Node Tags** (check metadata):
```python
# Forward nodes:
%mul : Tensor = call_function[...]  # meta["partitioner_tag"] = "is_forward"

# Backward nodes:
%grad_mul : Tensor = call_function[...]  # meta["partitioner_tag"] = "is_backward"
```

**Graph Flow**:
```
Inputs (primals) → Forward computation → Outputs
                         ↓ (saved activations)
Tangents (grad outputs) → Backward computation → Gradients
```

**What to look for**:
- Forward/backward separation
- Activation saving decisions
- Gradient flow correctness

---

## Tracing Partitioning

### What Partitioning Does

**Input**: Joint graph (forward + backward)
**Output**: Separate forward and backward graphs

**Strategies**:
- **Default**: Simple forward/backward split
- **Min-Cut**: Optimizes memory via recomputation

### How to Trace

```bash
TORCH_LOGS="aot,aot_graphs,aot_joint_graph" python script.py
```

**Compare**:
1. Joint graph: `model__*__joint_*.py`
2. Forward graph: `model__*__forward_*.py`
3. Backward graph: `model__*__backward_*.py`

### Verifying Partition

**Check forward outputs**:
```python
# Forward should output:
# 1. User-visible outputs
# 2. Saved activations for backward
return (output, saved_activation_1, saved_activation_2, ...)
```

**Check backward inputs**:
```python
# Backward should receive:
# 1. Saved activations from forward
# 2. Gradient w.r.t. outputs (tangents)
def backward(saved_act_1, saved_act_2, grad_output):
    ...
```

**Verify correspondence**:
```bash
# Forward outputs should match backward inputs
grep "return" model__*__forward_*.py
grep "placeholder" model__*__backward_*.py
```

### Recomputation Analysis

**What is recomputed**:
- Operations recalculated in backward instead of saved
- Trade memory for compute time

**How to identify**:
```bash
# Compare joint vs backward graph
# If operation appears in both, it's recomputed
diff <(grep "call_function" joint.py | grep "is_forward") \
     <(grep "call_function" backward.py)
```

---

## Post-Grad FX Passes

### When They Run

**After**: Partitioning
**Before**: Inductor lowering

**On**: Both forward and backward graphs separately

### How to Trace

```bash
TORCH_LOGS="post_grad_graphs" python script.py
```

**Output shows**:
- Graph before passes
- Each pass applied
- Graph after passes

### Common Passes

| Pass | What It Does | How to Verify |
|------|--------------|---------------|
| Group Batch Fusion | Batches operations together | Look for fused ops |
| B2B GEMM | Fuses back-to-back matrix multiplies | Check for combined mm ops |
| Remove Noop | Eliminates no-op operations | Count nodes before/after |
| Pattern Matching | Various graph rewrites | Compare transformed graph |

### Verifying Pass Effects

**Before Post-Grad**:
```python
%mm1 : Tensor = call_function[target=aten.mm](args = (%x, %w1))
%mm2 : Tensor = call_function[target=aten.mm](args = (%mm1, %w2))
```

**After Post-Grad** (B2B GEMM fusion):
```python
%fused_mm : Tensor = call_function[target=fused_mm_template](
    args = (%x, %w1, %w2)
)
```

---

## Debugging Workflows

### Workflow 1: Verify AOT Ran

**Goal**: Confirm AOT Autograd executed

**Steps**:
1. Enable logging:
   ```bash
   TORCH_LOGS="aot" python script.py
   ```

2. Check for AOT messages:
   ```
   [AOT] Compiling forward graph: ...
   [AOT] Compiling backward graph: ...
   ```

3. If missing:
   - Check if model needs gradients
   - Verify training mode: `model.train()`
   - Check inputs: `x.requires_grad = True`

### Workflow 2: Debug Incorrect Gradients

**Symptom**: Wrong gradients after compilation

**Steps**:
1. Compare with eager:
   ```python
   # Eager mode
   loss = model(x)
   loss.backward()
   grad_eager = x.grad.clone()

   # Compiled
   model_compiled = torch.compile(model)
   loss = model_compiled(x)
   loss.backward()
   grad_compiled = x.grad.clone()

   torch.testing.assert_close(grad_eager, grad_compiled)
   ```

2. Check joint graph:
   ```bash
   TORCH_LOGS="aot_joint_graph" python script.py
   # Verify backward computation looks correct
   ```

3. Check partitioning:
   ```bash
   TORCH_LOGS="aot_graphs" python script.py
   # Verify forward saves correct activations
   # Verify backward receives correct inputs
   ```

4. Isolate issue:
   - Simplify model to minimal reproduction
   - Check specific operation gradients

### Workflow 3: Debug Memory Issues

**Symptom**: OOM during backward pass

**Steps**:
1. Check what's being saved:
   ```bash
   TORCH_LOGS="aot_graphs" python script.py
   # Look at forward graph outputs
   # Count number of saved activations
   ```

2. Enable recomputation:
   ```python
   from torch._functorch.aot_autograd import aot_function
   from functools import partial
   from functorch.compile import min_cut_rematerialization_partition

   # Use min-cut partitioner for memory optimization
   # (Usually automatic, but can force via config)
   ```

3. Analyze activation memory:
   ```bash
   # Count tensors in forward output
   grep "return" model__*__forward_*.py
   # Each returned tensor (except user output) is saved
   ```

### Workflow 4: Verify Post-Grad Optimization

**Goal**: Confirm expected fusion happened

**Steps**:
1. Enable logging:
   ```bash
   TORCH_LOGS="post_grad_graphs" python script.py
   ```

2. Compare before/after:
   ```bash
   # Count operations
   grep "call_function" model__*__forward_0.py | wc -l
   grep "call_function" model__*__forward_transformed_0.py | wc -l
   ```

3. Verify specific pattern:
   - B2B GEMM: Look for `mm` → `mm` fusion
   - Attention: Check for fused attention pattern

---

## Common Issues

### Issue: AOT Not Running (Inference Mode)

**Symptom**: No AOT log messages, straight to Inductor

**Cause**: Model in inference mode (no gradients needed)

**Debug**:
```python
# Check if gradients needed
print(any(p.requires_grad for p in model.parameters()))
print(x.requires_grad)
```

**Fix**:
```python
model.train()  # Enable training mode
x.requires_grad = True  # Or make input require grad
```

### Issue: Saved Activations Too Large

**Symptom**: High memory usage, OOM

**Debug**:
```bash
TORCH_LOGS="aot_graphs" python script.py
# Check forward output size
grep "return" model__*__forward_*.py
```

**Solutions**:
- Use gradient checkpointing: `torch.utils.checkpoint.checkpoint()`
- Enable recomputation (usually automatic)
- Reduce batch size

### Issue: Backward Graph Missing Operations

**Symptom**: Incomplete backward computation

**Debug**:
```bash
TORCH_LOGS="aot_joint_graph" python script.py
# Check if backward nodes present in joint graph
grep "is_backward" model__*__joint_*.py
```

**Common causes**:
- Operation doesn't require grad
- Detached tensors breaking gradient flow
- In-place operations disrupting autograd

**Fix**:
```python
# Ensure gradient flow not broken
# Check for .detach() calls
# Verify requires_grad=True
```

### Issue: Post-Grad Fusion Not Happening

**Symptom**: Expected fusion didn't occur

**Debug**:
```bash
TORCH_LOGS="post_grad_graphs" python script.py
# Compare before/after, verify pattern exists
```

**Common causes**:
- Pattern not exactly matching expected form
- Operations not adjacent in graph
- Unsupported operation variants

**Fix**:
- Verify pattern manually in before graph
- Check for intermediate operations breaking pattern
- Ensure using supported op variants

---

## Quick Reference

### Essential Commands

```bash
# Basic AOT tracing
TORCH_LOGS="aot,aot_graphs" python script.py

# With joint graph
TORCH_LOGS="aot,aot_graphs,aot_joint_graph" python script.py

# Include post-grad passes
TORCH_LOGS="aot,aot_graphs,post_grad_graphs" python script.py

# Full AOT debug
TORCH_LOGS="aot,aot_graphs,aot_joint_graph,post_grad_graphs" python script.py
```

### Output Files

```bash
# View forward graph
cat /tmp/torchinductor_$USER/model__*__forward_0.py

# View backward graph
cat /tmp/torchinductor_$USER/model__*__backward_0.py

# View joint graph (if logged)
cat /tmp/torchinductor_$USER/model__*__joint_0.py

# Compare before/after post-grad
diff model__*__forward_{0,transformed_0}.py
```

### Key Checks

```bash
# Verify functionalization (no in-place ops)
grep "mul_\|add_\|sub_" model__*__forward_0.py
# Should return nothing

# Check partitioning (forward outputs match backward inputs)
grep "return" model__*__forward_0.py
grep "placeholder" model__*__backward_0.py

# Count saved activations
grep "return" model__*__forward_0.py | grep -o "%" | wc -l
```

---

## Next Stage

**After AOT Stage**: Load `compile-trace-inductor` skill - Tracing Inductor lowering through codegen

---

**Reference**: See `compile-overview` skill for complete pipeline context.
