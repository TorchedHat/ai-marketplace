---
name: compile-overview
description: Reference documentation for torch.compile pipeline architecture, IR levels (Full ATen, Core ATen, Prims), stages, TORCH_LOGS flags, output files, and debugging tools. Use for understanding pipeline structure, operator IRs, and available debugging options.
---

# torch.compile Pipeline Reference

Quick reference for pipeline architecture, logging, and debugging tools.

## Pipeline Architecture

```
Python Source
    ↓
┌─────────────── DYNAMO ─────────────┐
│ Bytecode capture → FX graph (aten) │
│ Pre-grad pattern matching           │
└─────────────────────────────────────┘
    ↓
┌─────────────── AOT ────────────────┐
│ Functionalization (remove mutations)│
│ Decompositions (break down ops)    │
│ Joint graph (fwd+bwd if training)  │
│ Partitioning (separate graphs)     │
│ Post-grad pattern matching         │
└─────────────────────────────────────┘
    ↓
┌─────────────── INDUCTOR ───────────┐
│ Lowering (ATen → IR nodes)         │
│ Scheduling & fusion decisions      │
│ LoopBody generation                │
│ Codegen (Triton/C++ kernels)      │
└─────────────────────────────────────┘
    ↓
GPU/CPU Execution
```

## Stage Summary

| Stage | Input | Output | Key Operations | Skills |
|-------|-------|--------|----------------|--------|
| **Dynamo** | Python bytecode | FX graph (Full ATen IR) | Capture, guards, pre-grad passes | `compile-trace-dynamo`, `pytorch-dynamo` |
| **AOT** | FX graph (Full ATen) | FX graph (Core ATen IR) | Functionalization, decompositions, partitioning, post-grad passes | `compile-trace-aot`, `pytorch-aot` |
| **Inductor** | FX graph (Core ATen) | Kernel code | Lowering, fusion, scheduling, codegen | `compile-trace-inductor`, `pytorch-inductor` |

## Operator IR Hierarchy

torch.compile transforms operations through multiple **operator-level IR layers**. These are distinct from Inductor's loop-level IR, Scheduler IR, and Triton IR (covered in pytorch-inductor skill).

| Operator IR Level | Description | Created By |
|-------------------|-------------|------------|
| **Full ATen IR** | Complete ATen API (includes in-place, out variants) | Dynamo graph capture |
| **Core ATen IR** | Functional subset, no mutations/aliases | AOT functionalization |
| **Prims IR** | Primitive operators with explicit broadcasting | Decompositions |

**Operator transformation pipeline:**
```
Full ATen (Dynamo output)
    ↓ Functionalization (AOT Stage 2)
Core ATen (functional ops)
    ↓ Decompositions (AOT + Inductor)
Prims or simpler Core ATen
    ↓ Lowerings (Inductor)
Inductor IR (loop-level: Buffer, Pointwise, Reduction, etc.)
```

## TORCH_LOGS Flags

### Dynamo Stage

| Flag | What It Shows |
|------|---------------|
| `dynamo` | Capture and graph construction |
| `graph_breaks` | Break locations and reasons |
| `graph_code` | Generated FX graph code |
| `guards` | Guard checks and failures |
| `recompiles` | Recompilation events |
| `pre_grad_graphs` | Before/after pre-grad passes |
| `bytecode` | Bytecode transformations |

### AOT Stage

| Flag | What It Shows |
|------|---------------|
| `aot` | General AOT logging |
| `aot_joint_graph` | Joint forward+backward graph |
| `aot_graphs` | Partitioned forward/backward graphs |
| `post_grad_graphs` | Before/after post-grad passes |

### Inductor Stage

| Flag | What It Shows |
|------|---------------|
| `fusion` | Fusion decisions and scheduling |
| `schedule` | Scheduling decisions |
| `ir_post_fusion` | Post-fusion IR dump |
| `output_code` | Generated Triton/C++ code |
| `cudagraphs` | CUDA graphs capture/replay |

### All Stages

```bash
TORCH_LOGS="dynamo,graph_breaks,aot,fusion,schedule,output_code"
```

## Output Files

**Location**: `/tmp/torchinductor_$USER/` (or `$TORCH_COMPILE_DEBUG_DIR`)

### Dynamo Files

| File | Content |
|------|---------|
| `fx_graph_readable.py` | Captured FX graph |
| `fx_graph_transformed.py` | After pre-grad passes |

### AOT Files

| File | Content |
|------|---------|
| `model__*__joint_*.py` | Joint forward+backward graph |
| `model__*__forward_*.py` | Partitioned forward graph |
| `model__*__backward_*.py` | Partitioned backward graph |

### Inductor Files

| File | Content |
|------|---------|
| `ir_*.txt` | Pre-fusion IR (IR nodes) |
| `ir_post_fusion_*.txt` | Post-fusion IR (LoopBody ops) |
| `output_code.py` | Generated Triton/C++ kernels |

## Debugging Tools

### Compiler Bisector

Automatically isolate failures to backend/subsystem/operation:
```bash
python -m torch._inductor.compiler_bisector run repro.py
```

See `compile-bisect` skill for details.

### Config Options

```python
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config

# Dynamo config
dynamo_config.verbose = True
dynamo_config.suppress_errors = False

# Inductor config
inductor_config.debug = True
inductor_config.trace.enabled = True
```

### Fresh Cache

```python
from torch._inductor.utils import fresh_cache

with fresh_cache():
    # Compilation happens with clean cache
    compiled_fn(x)
```

## Common Workflows

**Debugging graph breaks:**
1. Enable `TORCH_LOGS="graph_breaks"`
2. Load `compile-trace-dynamo`
3. Check break reasons in output

**Debugging fusion:**
1. Enable `TORCH_LOGS="fusion,schedule"`
2. Load `compile-trace-inductor`
3. Check fusion decisions in output

**Adding operator support:**
1. Check Dynamo capture (VariableTracker support)
2. Add Inductor lowering in `torch/_inductor/lowering.py`
3. Verify with bisector

**Performance investigation:**
1. Check AOT partitioning (training memory)
2. Check Inductor fusion (kernel count)
3. Check generated code quality

## Skills Overview

| Skill | Type | Purpose |
|-------|------|---------|
| `compile-debug` | Active workflow | End-to-end debugging (bisect → investigate → fix) |
| `compile-bisect` | Tool reference | Bisector usage and routing |
| `compile-overview` | Documentation | This file - architecture and flags |
| `compile-trace-dynamo` | Stage tracing | Debug Dynamo capture |
| `compile-trace-aot` | Stage tracing | Debug AOT transforms |
| `compile-trace-inductor` | Stage tracing | Debug Inductor compiler |
| `pytorch-dynamo` | Implementation | Dynamo internals |
| `pytorch-inductor` | Implementation | Inductor internals |

---

**For active debugging**: Use `/compile-debug` command
**For stage details**: Load stage-specific trace/implementation skills
