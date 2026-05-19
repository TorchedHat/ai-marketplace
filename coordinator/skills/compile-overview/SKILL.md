---
name: compile-overview
description: Meta-skill providing pipeline overview and bisect-first workflow for PyTorch torch.compile debugging. Explains the complete compilation pipeline (Dynamo → AOT → Inductor) and recommends loading compile-bisect first to determine which stage failed, then routing to the appropriate specialized skill (compile-trace-dynamo, compile-trace-aot, compile-trace-inductor) based on bisect results.
---

# PyTorch Compile Pipeline - Overview & Routing Guide

**Purpose**: Entry point for torch.compile debugging. Provides pipeline context and explains the bisect-first workflow.

## Recommended Workflow: Bisect-First

**For any compilation issue, start here:**

```
1. Load compile-overview (this skill) → Get pipeline context
2. Load compile-bisect skill → Find exact failing stage/operation
3. Bisect result tells you which skill to load:
   - backend='eager' → Load compile-trace-dynamo (Dynamo issue)
   - backend='aot_*' → Load compile-trace-aot (AOT issue)
   - backend='inductor' → Load compile-trace-inductor (Inductor issue)
4. Follow stage-specific tracing guidance
5. If fixing code: Load pytorch-dynamo or pytorch-inductor
6. Verify fix by running bisect again
```

**Why bisect-first:**
- ✅ Automated detection vs manual log analysis
- ✅ Pinpoints exact stage and operation before deep diving
- ✅ Tells you which skill to load next
- ✅ Saves time by targeting the right IR level immediately

---

## Complete Compilation Pipeline

**Single Source of Truth** - This is the only place the full pipeline is specified:

```
Python Source
    ↓
┌─────────────── DYNAMO STAGE ───────────────┐
│  1. Dynamo Capture → FX Graph (aten ops)   │
│  2. Pre-Grad Passes → Pattern matching     │
└────────────────────────────────────────────┘
    ↓
┌─────────────── AOT STAGE ──────────────────┐
│  3. Functionalization → Remove mutations   │
│  4. Decompositions → Break down ops        │
│  5. Joint Graph → Fwd + Bwd (if training)  │
│  6. Partitioning → Separate graphs         │
│  7. Post-Grad Passes → Optimizations       │
└────────────────────────────────────────────┘
    ↓
┌─────────────── INDUCTOR STAGE ─────────────┐
│  7. Lowering → IR Nodes (Pointwise, etc.)  │
│  8. Scheduler → Fusion decisions           │
│  9. LoopBody → ops.* operations            │
│  10. Codegen → Triton/C++ kernels          │
│  11. Compilation → PTX → CUBIN             │
└────────────────────────────────────────────┘
    ↓
GPU Execution
```

**Three Major Stages**:

1. **Dynamo Stage**: Python bytecode capture → FX graph (aten ops)
   - Graph breaks, unsupported operations
   - Pre-grad pattern matching (Conv-BN fusion, etc.)
   - **Bisect backends**: `eager`
   - **Load**: `compile-trace-dynamo` skill

2. **AOT Stage**: Functionalization, decompositions, and graph transformations
   - Runs in both training and inference
   - Functionalization, decompositions, joint graph (training), partitioning
   - Post-grad pattern matching (GEMM fusion, etc.)
   - **Bisect backends**: `aot_eager`, `aot_eager_decomp_partition`
   - **Load**: `compile-trace-aot` skill

3. **Inductor Stage**: Backend compiler (lowering → scheduling → codegen)
   - ATen ops → IR nodes → kernels
   - Fusion decisions, memory planning
   - Triton/C++ code generation
   - **Bisect backends**: `inductor`
   - **Load**: `compile-trace-inductor` skill

---

## Bisect-to-Skill Routing

**Bisect output determines which skill to load:**

### Bisect Result: `backend='eager'`
**Stage**: Dynamo (capture, no AOT)  
**Load**: `compile-trace-dynamo` skill  
**Likely issues**: Graph breaks, unsupported operations, VariableTracker issues  
**Fix with**: `pytorch-dynamo` skill (if implementing/fixing)

### Bisect Result: `backend='aot_eager'` or `backend='aot_eager_decomp_partition'`
**Stage**: AOT Autograd  
**Load**: `compile-trace-aot` skill  
**Likely issues**: Functionalization, decompositions, partitioning, post-grad passes  
**Fix with**: Core PyTorch (AOT internals are not in pytorch-inductor skill)

**Subsystem details:**
- `subsystem='cse'` → Common subexpression elimination
- `subsystem='decomposition'` → Operator decomposition

### Bisect Result: `backend='inductor'`
**Stage**: Inductor compiler  
**Load**: `compile-trace-inductor` skill  
**Likely issues**: Lowerings, fusion, scheduling, codegen  
**Fix with**: `pytorch-inductor` skill (if implementing/fixing)

**Subsystem details:**
- `subsystem='pre_grad_passes'` → Pre-grad optimizations (Conv-BN fusion, etc.)
- `subsystem='post_grad_passes'` → Post-grad optimizations (GEMM fusion, etc.)
- `subsystem='lowerings'` → Missing operator lowering (check `debug_info` for operation)
- `subsystem='cudagraphs'` → CUDA graphs backend wrapper

**Example:**
```bash
# Run bisect
python -m torch._inductor.compiler_bisector run python repro.py

# Output example 1:
# backend='inductor', subsystem='lowerings', debug_info='aten.argmin.default'
→ Load compile-trace-inductor + pytorch-inductor
→ Focus on lowering registration for aten.argmin

# Output example 2:
# backend='eager'
→ Load compile-trace-dynamo + pytorch-dynamo
→ Focus on graph breaks and capture issues

# Output example 3:
# backend='aot_eager_decomp_partition', subsystem='decomposition'
→ Load compile-trace-aot
→ Focus on operator decomposition
```

---

## Skill Dependency Map

```
compile-overview (you are here)
    │
    ├─→ compile-bisect
    │   Purpose: Find exact failing backend/subsystem/operation
    │   When: Always start here for failures
    │   Output: Tells you which skill to load next
    │
    ├─→ compile-trace-dynamo
    │   Purpose: Debug Dynamo capture, graph breaks, pre-grad passes
    │   When: bisect says backend='eager'
    │   Pairs with: pytorch-dynamo (for implementation)
    │
    ├─→ compile-trace-aot
    │   Purpose: Debug AOT autograd, functionalization, post-grad passes
    │   When: bisect says backend='aot_*'
    │   Note: Only relevant for training (requires_grad=True)
    │
    └─→ compile-trace-inductor
        Purpose: Debug lowering, fusion, kernel generation
        When: bisect says backend='inductor'
        Pairs with: pytorch-inductor (for implementation)
```

**Tracing vs Implementation Skills:**
- **Tracing skills** (`compile-trace-*`): How to debug, what logs to enable, how to interpret IR
- **Implementation skills** (`pytorch-*`): How internals work, how to fix/add code

---

## When Bisect Isn't Applicable

**Non-failure scenarios** where you can skip bisect:

1. **Understanding behavior** (not debugging a failure)
   - "How does torch.compile work?" → Read this skill
   - "Show me the generated kernel for fn(x)" → Load compile-trace-inductor directly

2. **Performance investigation** (no failure, just slow)
   - Fusion not happening → Load compile-trace-inductor
   - Memory usage → Load compile-trace-aot (for training) or compile-trace-inductor

3. **Implementing new features** (proactive development)
   - Adding new operator → Load pytorch-inductor + compile-trace-inductor
   - Adding VariableTracker → Load pytorch-dynamo + compile-trace-dynamo

**In these cases, use your judgment to load the relevant stage skill directly.**

---

## Common Multi-Stage Workflows

### Adding New Operator Support

**Stages involved**: Dynamo (capture) + Inductor (lowering)

**Skills to load:**
1. `pytorch-dynamo` - Ensure VariableTracker support (if custom object)
2. `pytorch-inductor` - Add lowering registration
3. `compile-trace-inductor` - Verify codegen works
4. `compile-bisect` - Verify no failures

### Debugging Fusion Issues

**Primary stage**: Inductor (scheduler)

**Skills to load:**
1. `compile-trace-inductor` - Understand fusion decisions
2. `pytorch-inductor` - Modify fusion patterns if needed

### Investigating Graph Breaks

**Primary stage**: Dynamo

**Skills to load:**
1. `compile-trace-dynamo` - Identify break locations
2. `pytorch-dynamo` - Understand why breaks occur, add support

### Optimizing Training Performance

**Stages involved**: AOT (partitioning) + Inductor (fusion)

**Skills to load:**
1. `compile-trace-aot` - Check recomputation strategy
2. `compile-trace-inductor` - Verify kernel fusion
3. `pytorch-inductor` - Tune fusion/scheduling

---

## Output Files Reference

**Location**: `/tmp/torchinductor_$USER/` (or `TORCH_COMPILE_DEBUG_DIR` if set)

**Dynamo Stage:**
- `fx_graph_readable.py` - Captured FX graph
- `fx_graph_transformed.py` - After pre-grad passes

**AOT Stage:**
- `model__*__joint_*.py` - Joint forward+backward graph
- `model__*__forward_*.py` - Partitioned forward graph
- `model__*__backward_*.py` - Partitioned backward graph

**Inductor Stage:**
- `ir_*.txt` - Pre-fusion IR (IR nodes)
- `ir_post_fusion_*.txt` - Post-fusion IR (LoopBody ops.*)
- `output_code.py` - Generated Triton/C++ kernels

**These file paths are referenced by stage-specific skills.**

---

## Essential TORCH_LOGS Reference

**Dynamo:**
- `TORCH_LOGS="dynamo"` - Capture and graph construction
- `TORCH_LOGS="graph_breaks"` - Break locations and reasons
- `TORCH_LOGS="pre_grad_graphs"` - Before/after pre-grad passes

**AOT Autograd:**
- `TORCH_LOGS="aot"` - General AOT logging
- `TORCH_LOGS="aot_joint_graph"` - Joint graph construction
- `TORCH_LOGS="aot_graphs"` - Partitioned graphs
- `TORCH_LOGS="post_grad_graphs"` - Before/after post-grad passes

**Inductor:**
- `TORCH_LOGS="fusion"` - Fusion decisions
- `TORCH_LOGS="schedule"` - Scheduling decisions
- `TORCH_LOGS="ir_post_fusion"` - Post-fusion IR dump
- `TORCH_LOGS="output_code"` - Generated kernel code

**All stages:**
```bash
TORCH_LOGS="dynamo,graph_breaks,aot,fusion,schedule,output_code"
```

**Detailed logging flags are in stage-specific skills.**

---

## Agent Routing Guidelines

**When you (as an agent) encounter a torch.compile task:**

### For Failures:
1. Load `compile-overview` (this skill) - Get pipeline context
2. Load `compile-bisect` - Find exact failure point
3. Check bisect output:
   - `backend='eager'` → Load `compile-trace-dynamo`
   - `backend='aot_*'` → Load `compile-trace-aot`
   - `backend='inductor'` → Load `compile-trace-inductor`
4. Follow stage-specific tracing guidance
5. If implementing fix: Load `pytorch-dynamo` or `pytorch-inductor`

### For Non-Failures:
1. Load `compile-overview` (this skill) - Get pipeline context
2. Identify relevant stage from task description
3. Load appropriate stage skill directly
4. If implementing: Load corresponding implementation skill

---

## Next Steps

**Start your debugging journey:**

1. **For compilation failures:**
   - Load `compile-bisect` skill now
   - Run bisection to find exact failure point
   - Bisect will tell you which skill to load next

2. **For non-failure investigations:**
   - Determine which stage is relevant
   - Load the appropriate skill:
     - Dynamo issues → `compile-trace-dynamo`
     - AOT/training issues → `compile-trace-aot`
     - Inductor/fusion/performance → `compile-trace-inductor`

3. **For implementation work:**
   - Load tracing skill for stage (`compile-trace-*`)
   - Load implementation skill (`pytorch-dynamo` or `pytorch-inductor`)

---

**This is your entry point. The bisect result will route you to the right specialized skill.**
