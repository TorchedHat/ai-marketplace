---
name: compile-trace-inductor
description: Debug PyTorch Inductor compiler backend - IR lowering, scheduler/fusion, loopbody ops, and Triton/C++ codegen. Covers TORCH_LOGS for fusion/schedule/ir_post_fusion/output_code, config.trace.enabled for IR dumps, interpreting Inductor IR nodes (Pointwise/Reduction/etc), fusion decisions, kernel generation, and performance optimization. Load after compile-bisect indicates backend='inductor'.
---

# Tracing Inductor Stage - Lowering Through Codegen

Complete guide to tracing PyTorch Inductor: IR lowering, scheduler, fusion, loopbody, and Triton codegen.

## Table of Contents

1. [Stage Overview](#stage-overview)
2. [Inductor Pipeline Summary](#inductor-pipeline-summary)
3. [Stage 1: Inductor Lowering](#stage-1-inductor-lowering)
4. [Stage 2: Fusion & Scheduling](#stage-2-fusion--scheduling)
5. [Stage 3: LoopBody Creation](#stage-3-loopbody-creation)
6. [Stage 4: Triton Codegen](#stage-4-triton-codegen)
7. [Stage 5: Triton Compiler](#stage-5-triton-compiler)
8. [Stage 6: Execution](#stage-6-execution)
9. [IR Level Comparison](#ir-level-comparison)
10. [Debugging Workflows](#debugging-workflows)

---

## Stage Overview

**Inductor** = PyTorch's deep learning compiler backend for `torch.compile`

**What it does**:
- Lowers ATen ops to loop-level IR
- Fuses operations into efficient kernels
- Generates Triton (GPU) or C++ (CPU) code
- Compiles to machine code

**For previous stages**:
- **Dynamo capture**: See `compile-trace-dynamo` skill
- **AOT Autograd**: See `compile-trace-aot` skill

**Key Location**: `torch/_inductor/`

---

## Inductor Pipeline Summary

**Inductor receives**: FX graph with ATen ops (from Dynamo or AOT Autograd)  
**Inductor produces**: Compiled GPU/CPU kernels

```
ATen FX Graph → Lowering → Scheduler → LoopBody → Codegen → Triton → Execution
               (IR Nodes) (Wrappers)  (ops.*)    (Triton)  (PTX/CUBIN)
```

**Inductor Stages**:
1. **Lowering**: ATen ops → Inductor IR nodes (Pointwise, Reduction, etc.)
2. **Scheduling**: IR nodes → SchedulerNode wrappers → Fusion decisions
3. **LoopBody**: Fused nodes → ops.load/store/index_expr operations
4. **Codegen**: ops.* → Triton/C++ source code
5. **Compilation**: Triton → PTX → CUBIN (executable)

**Key Insight**: Two-level IR design:
- **IR nodes** (Pointwise, Reduction) define WHAT to compute
- **SchedulerNode wrappers** define HOW and WHEN to compute

---

## Stage 1: Inductor Lowering

**Location**: `torch/_inductor/lowering.py`, `torch/_inductor/ir.py`

**What Happens**:
- Lowers aten ops to Inductor IR nodes
- Creates Buffer, Pointwise, Reduction nodes
- Each node contains an `inner_fn` that defines computation

**Example Input** (from Stage 2):
```python
%arange = torch.ops.aten.arange.start_step(0, 11, ...)
%mul = torch.ops.aten.mul.Tensor(%arange, 1000000000)
```

**Example Output** (Inductor IR):
```python
# aten.arange lowering
buf0 = Pointwise(
    device=device('cuda:0'),
    dtype=torch.int64,
    inner_fn=lambda index: ops.index_expr(
        index[0],  # p0 (iteration variable)
        torch.int64
    ),
    ranges=[11]  # Iteration space: p0 from 0 to 10
)

# aten.mul lowering
buf1 = Pointwise(
    device=device('cuda:0'),
    dtype=torch.int64,
    inner_fn=lambda index: ops.mul(
        ops.load('buf0', index[0]),
        1000000000
    ),
    ranges=[11]
)
```

**How to View**:
```python
import torch._inductor.config as config
config.trace.enabled = True
config.debug = True
# Run torch.compile
# Output: /tmp/torchinductor_$USER/ir_*.txt
```

**Key Concepts**:
- **Buffer**: Represents a tensor buffer
- **Pointwise**: Element-wise operation
- **Reduction**: Reduction operation (sum, max, etc.)
- **inner_fn**: Python function defining the computation
- **ranges**: Iteration space (e.g., [11] means iterate 0-10)

**When to Debug at This Level**:
- Missing lowerings (operation not implemented)
- Understanding pre-fusion IR structure
- Decomposition issues

---

## Stage 2: Fusion & Scheduling

**Location**: `torch/_inductor/scheduler.py`

**What Happens**:
- Wraps IR nodes in scheduler data structures
- Analyzes dependencies between operations
- Fuses compatible operations into single kernels
- Determines execution order and kernel launch parameters

### Two-Level Design

The scheduler creates **wrapper objects** around IR nodes to track dependencies, fusion opportunities, and execution state:

```
┌─────────────────────────────────────────────┐
│    Scheduler Wrappers (metadata)            │
│  SchedulerNode, SchedulerBuffer, Fused...   │
└─────────────────────────────────────────────┘
           ↓ wraps ↓
┌─────────────────────────────────────────────┐
│    IR Nodes (computation semantics)         │
│  Pointwise, Reduction, ComputedBuffer       │
└─────────────────────────────────────────────┘
```

**Why this separation?**
- IR nodes define WHAT to compute (semantics)
- Scheduler wrappers define HOW and WHEN to compute (optimization)
- IR nodes remain immutable; scheduler creates new fusion wrappers

### SchedulerBuffer - Tracks Buffer Lifetime

**Definition** (simplified):
```python
@dataclasses.dataclass
class SchedulerBuffer:
    scheduler: Scheduler
    node: ir.Buffer                     # The actual IR buffer
    defining_op: BaseSchedulerNode      # SchedulerNode that creates this
    users: list[NodeUser]               # SchedulerNodes that read this
    mpi_buffer: MemoryPlanningInfo      # Memory planning metadata
```

**Purpose**: 
- Tracks which operation creates a buffer
- Tracks which operations use the buffer
- Enables buffer lifetime analysis and memory reuse

**Example**:
```python
# For buf0 created by relu operation
SchedulerBuffer(
    node=buf0,                    # ir.ComputedBuffer
    defining_op=snode_relu,       # SchedulerNode wrapping relu
    users=[
        NodeUser(node=snode_add)  # SchedulerNode wrapping add operation
    ]
)
```

### BaseSchedulerNode - Base Class

**Definition** (key fields):
```python
class BaseSchedulerNode:
    node: ir.Operation | None             # Wrapped IR node
    outputs: list[SchedulerBuffer]        # Buffers this produces
    read_writes: ReadWrites               # Dependency information
    unmet_dependencies: OrderedSet[Dep]   # Not yet satisfied
    ancestors: OrderedSet[str]            # All upstream nodes
    min_order: int                        # Position in schedule
    max_order: int
```

**Subclasses**:
- `SchedulerNode` - Wraps single IR operation
- `FusedSchedulerNode` - Represents multiple fused operations
- `ExternKernelSchedulerNode` - External operations (matmul, conv)

### SchedulerNode - Wraps Single IR Operation

**Definition** (key fields):
```python
class SchedulerNode(BaseSchedulerNode):
    node: ir.ComputedBuffer | ir.TemplateBuffer
    _sizes: tuple[Sequence[sympy.Expr], ...]  # Iteration ranges
    _body: LoopBody                           # Loop body computation
```

**Example** - Wrapping `x.relu()`:
```python
# Stage 4 created this IR node:
buf0 = ir.ComputedBuffer(
    name="buf0",
    data=ir.Pointwise(
        inner_fn=lambda i: ops.maximum(ops.load("x", i), 0.0),
        ranges=[100]
    )
)

# Stage 5 wraps it in SchedulerNode:
snode_relu = SchedulerNode(
    scheduler=scheduler,
    node=buf0,                           # The IR node
    _sizes=([100],),                     # Iteration space
    outputs=[
        SchedulerBuffer(
            node=buf0,
            defining_op=snode_relu,
            users=[snode_add]            # Next operation
        )
    ],
    read_writes=ReadWrites(
        reads={Dep("x")},                # Depends on input x
        writes={Dep("buf0")}             # Produces buf0
    ),
    unmet_dependencies={Dep("x")},       # Must wait for x
    group=(cuda:0, ((100,),))            # Grouping key for fusion
)
```

### FusedSchedulerNode - Represents Fused Operations

**Definition**:
```python
class FusedSchedulerNode(BaseSchedulerNode):
    """
    Combines multiple operations into a single kernel.
    Maintains union of constituent nodes' dependencies.
    """
    snodes: list[BaseSchedulerNode]  # The nodes being fused
```

**Example** - Fusing `x.relu().add(1)`:
```python
# Before fusion: Two separate SchedulerNodes
snode_relu = SchedulerNode(node=buf0_relu, ...)
snode_add = SchedulerNode(node=buf1_add, ...)

# After fusion: Single FusedSchedulerNode
fused = FusedSchedulerNode(
    scheduler=scheduler,
    snodes=[snode_relu, snode_add],      # Both operations combined
    outputs=[
        SchedulerBuffer(
            node=buf1,                    # Only final output needed
            defining_op=fused,
            users=[]                      # Output to user
        )
    ],
    read_writes=ReadWrites(
        reads={Dep("x")},                 # Only depends on original input
        writes={Dep("buf0"), Dep("buf1")} # Produces both intermediates
    ),
    unmet_dependencies={Dep("x")},        # Wait for input x
)

# Key insight: buf0 is no longer exposed as an output!
# It's computed internally within the fused kernel
```

### Fusion Decision Process

**Step 1: Group Compatible Operations**

Operations can fuse if they have:
- Same device (cuda:0)
- Same grouping key (iteration space structure)
- Producer-consumer or consumer-consumer relationship
- No dependency cycles

**Step 2: Score Fusion Opportunities**

```python
# Higher score = better fusion
score = (
    memory_saved          # Intermediate buffers eliminated
    - locality_penalty    # Operations farther apart in graph
)
```

**Step 3: Create FusedSchedulerNode**

Merge compatible operations into `FusedSchedulerNode` while maintaining correct dependencies.

### Complete Example: `x.relu().add(1).sum()`

**Stage 4 Output** (Inductor IR):
```python
buf0 = ir.ComputedBuffer(
    name="buf0",
    data=ir.Pointwise(
        inner_fn=lambda idx: ops.maximum(ops.load("x", idx), 0.0),
        ranges=[10, 100]
    )
)

buf1 = ir.ComputedBuffer(
    name="buf1", 
    data=ir.Pointwise(
        inner_fn=lambda idx: ops.add(ops.load("buf0", idx), 1.0),
        ranges=[10, 100]
    )
)

buf2 = ir.ComputedBuffer(
    name="buf2",
    data=ir.Reduction(
        inner_fn=lambda outer, reduction: ops.load("buf1", outer*100 + reduction),
        ranges=[10],
        reduction_ranges=[100],
        reduction_type="sum"
    )
)
```

**Stage 5 Step 1** - Create SchedulerNodes:
```python
# Wrap each IR node
snode0 = SchedulerNode(node=buf0, group=(cuda:0, ((10, 100),)))
snode1 = SchedulerNode(node=buf1, group=(cuda:0, ((10, 100),)))
snode2 = SchedulerNode(node=buf2, group=(cuda:0, ((10,), (100,))))

# Track dependencies via SchedulerBuffers
buf0_buffer = SchedulerBuffer(
    node=buf0,
    defining_op=snode0,
    users=[snode1]  # buf1 uses buf0
)

buf1_buffer = SchedulerBuffer(
    node=buf1,
    defining_op=snode1,
    users=[snode2]  # buf2 uses buf1
)
```

**Stage 5 Step 2** - Fusion Analysis:
```
Can fuse snode0 + snode1?
✓ Same device: cuda:0
✓ Same group: ((10, 100),)
✓ Producer-consumer: buf0 → buf1
✓ No cycles
→ YES, create FusedSchedulerNode

Can fuse with snode2?
✗ Different group: ((10,), (100,)) - has reduction dimension
→ NO, keep separate
```

**Stage 5 Step 3** - Create Fused Node:
```python
# Fuse snode0 and snode1
fused_pw = FusedSchedulerNode(
    snodes=[snode0, snode1],
    outputs=[SchedulerBuffer(node=buf1, users=[snode2])]
)

# Final schedule: [fused_pw, snode2]
# Result: 2 kernels instead of 3
```

**How to View**:
```bash
# View IR before fusion decisions
TORCH_LOGS="ir_pre_fusion" python script.py
# Output: /tmp/torchinductor_$USER/ir_pre_fusion_*.txt

# View fusion decision logs
TORCH_LOGS="fusion,schedule" python script.py
# Output: Console logs showing fusion decisions

# Combine both for complete analysis
TORCH_LOGS="ir_pre_fusion,fusion,schedule" python script.py
```

**Example Console Output**:
```
FusionDecision: buf0 (Pointwise) <- producer
FusionDecision: buf1 (Pointwise) <- consumer
  ✓ Ranges match: [10, 100]
  ✓ Vertical fusion (producer-consumer)
  → Fused into 1 kernel

FusionDecision: fused_pw (Fused Pointwise) <- producer  
FusionDecision: buf2 (Reduction) <- consumer
  ✗ Cannot fuse: Different iteration structure
  → Separate kernel
```

### Fusion Types

**Vertical Fusion (Producer-Consumer)**:
```python
# buf0 → buf1 (buf1 consumes buf0)
x.relu()    # buf0
 .add(1)    # buf1 (uses buf0)
# Fuse: both operations in single kernel
```

**Horizontal Fusion (Consumer-Consumer)**:
```python
# buf0 → buf1, buf0 → buf2 (both consume buf0)
a = x.relu()        # buf0
b = a.add(1)        # buf1 (uses buf0)
c = a.mul(2)        # buf2 (uses buf0)
# Fuse buf1 + buf2: read buf0 once, compute both
```

**Reduction Fusion**:
```python
# Reduction followed by pointwise on result
x.sum(dim=-1)    # buf0: Reduction
 .add(bias)      # buf1: Pointwise on reduced result
# Can fuse if reduction and pointwise compatible
```

### Fusion Constraints

**Required for fusion**:
- Same device
- Compatible iteration spaces
- No dependency cycles
- Satisfies memory constraints

**Blockers**:
- Cross-device operations
- Mismatched iteration ranges (unless broadcast)
- Circular dependencies
- Extern kernel boundaries (matmul, conv)

### When to Debug at This Level

- Operations not fusing as expected
- Understanding why fusion was/wasn't applied
- Performance regression from missed fusion
- Too many kernels launched
- Memory usage from intermediate buffers

---

## Stage 3: LoopBody Creation

**Location**: `torch/_inductor/loop_body.py`

**What Happens**:
- Traces `inner_fn` into FX graph (Inductor IR level, NOT aten)
- Creates `ops.load`, `ops.store`, `ops.index_expr` nodes
- This is the graph that gets code-generated

**Example Input** (from Stage 5):
```python
# Fused inner_fn for arange + mul
def inner_fn(index):
    # arange: idx = index[0]
    idx_val = ops.index_expr(index[0], torch.int64)
    # mul: idx * 1e9
    result = ops.mul(idx_val, 1000000000)
    return result
```

**Example Output** (LoopBody FX Graph):
```python
class triton_poi_fused_arange_mul_0_loop_body:
    var_ranges = {p0: 11}  # Iteration space
    
    # Index expressions (computed once, reused)
    index0 = p0                 # For store position
    index1 = 1000000000*p0      # For computed value
    
    def body(self, ops):
        # Get index for value computation
        get_index = self.get_index('index1')
        
        # Convert index to value with specified dtype
        index_expr = ops.index_expr(get_index, torch.int64)
        
        # Get index for store position
        get_index_2 = self.get_index('index0')
        
        # Store result
        store = ops.store('buf0', get_index_2, index_expr, None)
        #                         ^^^^^^^^^^^^  ^^^^^^^^^^
        #                         INDEX (addressing)  VALUE (data)
```

**How to View**:
```bash
TORCH_LOGS="ir_post_fusion" python script.py
# Output: /tmp/torchinductor_$USER/ir_post_fusion_*.txt
```

**Key Operations**:

**ops.load(buffer, index)**:
- Loads from `buffer` at `index`
- `index`: Used for memory addressing
- Result: The loaded value

**ops.store(buffer, index, value, mode)**:
- Stores `value` to `buffer` at `index`
- `index`: Memory address (argument position 2)
- `value`: Data to store (argument position 3)
- `mode`: Optional (None, 'atomic_add', etc.)

**ops.index_expr(expr, dtype)**:
- Converts index expression to value with dtype
- Critical for when iteration variables are used in computations
- Example: `ops.index_expr(p0, torch.int64)` → iteration var as int64 value

**Arithmetic ops**: `ops.add`, `ops.mul`, `ops.maximum`, etc.
- Standard arithmetic operations
- All arguments are values (not indices)

**When to Debug at This Level**:
- Dtype handling issues
- Understanding operation semantics
- Data flow analysis
- Index vs value usage

---

## Stage 4: Triton Codegen

**Location**: `torch/_inductor/codegen/triton.py`

**What Happens**:
- Interprets LoopBody FX graph via `InterpreterShim`
- Generates Triton kernel code (string)
- Each `ops.*` call becomes Triton code
- Applies optimizations (tiling, vectorization)

**Example Input** (from Stage 6):
```python
# LoopBody FX graph operations
get_index = self.get_index('index1')           # p0 * 1000000000
index_expr = ops.index_expr(get_index, int64)  # Convert to value
store = ops.store('buf0', index_pos, index_expr, None)
```

**Example Output** (Generated Triton Code):
```python
import triton
import triton.language as tl

@triton.jit
def triton_poi_fused_arange_mul_0(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Get thread/block indices
    xnumel = 11
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    # Computation (from ops.* operations)
    x0 = xindex  # Iteration variable (int32)
    tmp0 = 1000000000*x0  # Value computation
    
    # Store result
    tl.store(out_ptr0 + x0, tmp0, xmask)
```

**How to View**:
```bash
TORCH_LOGS="output_code" python script.py
# Output: /tmp/torchinductor_$USER/output_code.py
```

**Key Triton Concepts**:

**Index Variables**:
- `xindex`, `x0`, `x1`, etc. - Iteration variables
- `r0`, `r1`, etc. - Reduction variables
- Named to match iteration dimension

**Tiling Parameters**:
- `XBLOCK` - Tile size (auto-tuned)
- `tl.program_id(0)` - Block/workgroup ID
- `tl.arange(0, XBLOCK)` - Thread indices within block

**Masking**:
- `xmask = xindex < xnumel` - Bounds checking
- Ensures threads don't access out-of-bounds

**Memory Operations**:
- `tl.load(ptr + offset, mask)` - Load from memory
- `tl.store(ptr + offset, value, mask)` - Store to memory

**When to Debug at This Level**:
- Performance issues
- Understanding generated kernel structure
- Memory access patterns
- Tiling/vectorization problems

---

## Stage 5: Triton Compiler

**What Happens**:
- Compiles Triton code to PTX (NVIDIA assembly)
- PTX compiled to CUBIN (binary) via `ptxas`
- Binary cached for fast warmup on reruns

**Example Input** (from Stage 7):
```python
@triton.jit
def triton_poi_fused_arange_mul_0(...):
    x0 = xindex
    tmp0 = 1000000000*x0
    tl.store(out_ptr0 + x0, tmp0, xmask)
```

**Example Output**:
```
PTX (NVIDIA assembly) → CUBIN (binary)
Cached at: /tmp/triton_cache/...
```

**When to Debug at This Level**:
- Triton compilation failures
- PTX/CUBIN generation issues
- Usually handled by Triton itself

---

## Stage 6: Execution

**What Happens**:
- Wrapper code allocates tensors
- Launches compiled kernel on GPU
- Returns result to Python

**Example**:
```python
# Wrapper code (generated)
buf0 = torch.empty([11], dtype=torch.int64, device='cuda')
triton_poi_fused_arange_mul_0[grid](buf0, 11, XBLOCK=256)
return buf0
```

**Result**:
```python
tensor([0, 1000000000, 2000000000, ..., 10000000000], device='cuda:0')
```

---

## IR Level Comparison

| Level | Language | Example | When to Use |
|-------|----------|---------|-------------|
| **Python** | Python | `torch.arange(0, 11)` | User code |
| **FX/Aten** | FX nodes | `aten.arange.start_step(...)` | Graph breaks |
| **Inductor IR** | IR nodes | `Pointwise(inner_fn=...)` | Pre-fusion, lowering |
| **Scheduler** | Wrappers | `SchedulerNode(node=buf0)` | Fusion analysis |
| **Fused Scheduler** | Wrappers | `FusedSchedulerNode(snodes=[...])` | Fusion decisions |
| **LoopBody** | ops.* | `ops.index_expr(...)` | Dtype/semantics |
| **Triton** | Triton | `tl.store(...)` | Performance |
| **PTX** | Assembly | `st.global.u64 [%rd1], %rd2` | Low-level debug |

---

## Common Patterns Across Stages

### Pattern 1: Simple Pointwise Operation

```python
# Python
x.relu()

# FX/Aten
%relu = aten.relu.default(%x)

# Inductor IR
buf = Pointwise(inner_fn=lambda idx: ops.maximum(ops.load('x', idx), 0.0))

# LoopBody
tmp0 = ops.load('x', xindex)
tmp1 = ops.maximum(tmp0, 0.0)
ops.store('buf', xindex, tmp1)

# Triton
tmp0 = tl.load(x_ptr + xindex, xmask)
tmp1 = tl.maximum(tmp0, 0.0)
tl.store(out_ptr + xindex, tmp1, xmask)
```

### Pattern 2: Reduction Operation

```python
# Python
x.sum(dim=-1)

# FX/Aten
%sum = aten.sum.dim_IntList(%x, [-1])

# Inductor IR
buf = Reduction(
    inner_fn=lambda idx, reduction_idx: ops.add(
        ops.load('x', idx + reduction_idx),
        reduction_accumulator
    )
)

# LoopBody
# Reduction variable: r0
tmp0 = ops.load('x', xindex + r0)
tmp1 = ops.add(accumulator, tmp0)

# Triton
for r0 in range(0, rmax):
    tmp0 = tl.load(x_ptr + xindex*stride + r0)
    tmp1 = tmp1 + tmp0
tl.store(out_ptr + xindex, tmp1)
```

---

## Key Takeaways

1. **Each stage serves a specific purpose**: Dynamo captures, AOT transforms, Inductor lowers, Scheduler fuses, LoopBody defines, Codegen generates

2. **Two-level design (IR + Scheduler)**: 
   - IR nodes (Pointwise, Reduction) define WHAT to compute
   - Scheduler wrappers (SchedulerNode, FusedSchedulerNode) define HOW and WHEN
   - This separation enables fusion without modifying IR nodes

3. **Multiple IR levels**: Understanding which IR level is relevant for your issue is critical

4. **Progressive lowering**: Each stage takes higher-level representation and lowers it

5. **FX graphs appear twice**: 
   - Dynamo FX graph (aten ops)
   - LoopBody FX graph (ops.* operations)
   These are different!

6. **Index vs Value semantics**: In LoopBody, distinguish between:
   - Indices (for memory addressing)
   - Values (for computation)

7. **Fusion is key**: Most performance comes from fusing operations into fewer kernels
   - SchedulerBuffer tracks buffer lifetime and users
   - FusedSchedulerNode combines multiple operations
   - Intermediate buffers eliminated in fused kernels

8. **Scheduler operates between lowering and codegen**: It's the bridge that decides which IR nodes get combined before LoopBody generation

---

## Debugging Workflows

**Issue → Inductor Stage**:
- Missing lowerings → Inductor Lowering (Stage 1)
- Fusion not happening → Scheduler (Stage 2)
- Why specific fusion decision → Scheduler fusion logs (Stage 2)
- Wrong dtype/semantics → LoopBody (Stage 3)
- Slow kernel performance → Triton Codegen (Stage 4)
- Compilation errors → Triton Compiler (Stage 5)

**For issues before Inductor**:
- Graph breaks → See `compile-trace-dynamo` skill
- AOT Autograd issues → See `compile-trace-aot` skill

**General approach**:
1. Identify symptom
2. Choose relevant IR level from pipeline
3. Enable appropriate logging (`TORCH_LOGS`, `config.trace.enabled`)
4. Analyze IR output files or console logs
5. Trace transformation through pipeline stages
6. Understand what changed between stages
