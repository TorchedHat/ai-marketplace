# Functorch & AOT Autograd Architecture

Essential architecture overview for developing in `torch/_functorch`. Focus on what components do, not implementation details (use steering MCP for those).

## Functorch Overview

**Philosophy**: JAX-inspired functional programming for PyTorch. Enable composable function transformations without modifying PyTorch core.

**Key Idea**: Wrap PyTorch operations with transformation semantics:
- `vmap`: Add batching dimension
- `grad`: Add gradient computation
- Compose: `vmap(grad(fn))` = batched gradients

**Relationship to AOT**: AOT Autograd is functorch's compilation component. It uses functorch transforms (grad, make_fx) to compile training graphs.

## AOT Autograd Pipeline

**Pipeline Position**:
```
Python code
  ↓
[Dynamo] Bytecode → FX graph (aten ops)
  ↓
[AOT Autograd] Functionalization → Joint → Partition → Post-grad
  ↓
[Inductor] Lowering → Fusion → Codegen
  ↓
Compiled kernels
```

**Detailed AOT Stages**:

```
Input: FX Graph from Dynamo
  ↓
┌────────────── STAGE 1: Graph Capture ──────────────┐
│ 1. Functionalization                               │
│    - Wrap tensors in FunctionalTensor              │
│    - Remove mutations, track in metadata           │
│    - Record view operations in ViewMeta            │
│                                                     │
│ 2. Joint Graph Creation (if requires_grad=True)    │
│    - Run forward with FakeTensors via make_fx()    │
│    - Create tangents for outputs needing grads     │
│    - Run torch.autograd.grad() to get backward     │
│    - Tag nodes as is_forward vs is_backward        │
│                                                     │
│ 3. Input/Output Analysis                           │
│    - Classify inputs: params, buffers, user inputs │
│    - Create AOTInput descriptors                   │
│    - Analyze mutations and aliases                 │
│    - Build GraphSignature                          │
└─────────────────────────────────────────────────────┘
  ↓ Joint graph (or just forward if inference)
┌────────────── Partitioning ─────────────────────────┐
│ - Split joint → separate forward + backward        │
│ - Determine saved activations                      │
│   Simple: save all intermediate values             │
│   Min-cut: optimize memory by recomputation        │
│ - Create AOTOutput descriptors                     │
└─────────────────────────────────────────────────────┘
  ↓ Forward graph, Backward graph
┌────────────── STAGE 2: Compilation ─────────────────┐
│ 1. Post-Grad Passes (on both graphs)               │
│    - Pattern matching: GEMM fusion, Conv-BN, etc.  │
│    - Memory planning                               │
│                                                     │
│ 2. Compile Graphs                                  │
│    - Send to backend (Inductor)                    │
│    - Get compiled forward/backward functions       │
│                                                     │
│ 3. Create Runtime Wrappers                         │
│    - Unwrap FunctionalTensors on entry            │
│    - Apply saved activations in backward           │
│    - Rewrap outputs if needed                      │
└─────────────────────────────────────────────────────┘
  ↓
Output: Compiled forward/backward callables
```

**Key Files**:
- `aot_stage1_graph_capture()` in `graph_compile.py`: Implements stage 1
- `aot_stage2_compile()` in `graph_compile.py`: Implements stage 2
- `aot_function()` in `aot_autograd.py`: Main entry point

## Functionalization System

**Problem**: Backends assume functional graphs (no mutations, no aliases). PyTorch code has mutations (`x.add_(1)`) and views (`x.t()`).

**Solution**: Functionalization transparently converts mutations to functional ops while preserving semantics.

### FunctionalTensor

**Concept**: Wrapper around regular tensors that intercepts operations.

**Structure**:
- Wraps a "functional" tensor (immutable data)
- Tracks mutations separately as metadata
- Records view operations in ViewMeta chain

**Example**:
```python
# User code
x.mul_(2)
x_t = x.t()
x_t.add_(1)

# What functionalization does
x_func = FunctionalTensor(x)          # Wrap
x_new = x_func * 2                    # mul_ → *, mark x mutated
x_t_func = x_new.t()                  # Record view in ViewMeta
x_t_new = x_t_func + 1                # add_ → +
# Metadata: x was mutated twice (mul, add through view)
```

**Key insight**: Original semantics preserved, but graph is pure functional ops.

### ViewMeta System

**Purpose**: Track view relationships to reconstruct aliases after functionalization.

**What it records**:
- View operations: `transpose`, `permute`, `reshape`, `slice`, `expand`
- Base tensor reference
- Parameters for reconstruction

**Replay**: Given a base tensor and ViewMeta chain, reconstruct the view.

**Example**:
```python
y = x.t()           # ViewMeta: Transpose(dims=(0,1))
z = y[:, :10]       # ViewMeta: Transpose → Slice(dim=1, end=10)

# Later reconstruction from base tensor
reconstructed_z = base.t()[:, :10]  # Replays ViewMeta chain
```

**Files**: `functional_utils.py` (FunctionalTensor, ViewMeta), `schemas.py` (ViewMeta types).

### Mutation Tracking

**Types of mutations**:
1. **Data mutations**: `add_`, `mul_`, `copy_` - change tensor values
2. **Metadata mutations**: `resize_`, `set_` - change shape/storage
3. **Mutations through views**: `x.t().add_(1)` - mutates base through alias

**How tracked**:
- `GraphSignature.input_mutations`: Which inputs were mutated
- `GraphSignature.output_mutations`: Which outputs alias mutated inputs
- Metadata on FunctionalTensor: What ops caused mutations

**Steering**: Query "functionalization mutation tracking" for detailed implementation.

## Partitioning Strategies

**Goal**: Split joint forward+backward graph into separate graphs for execution.

**Why partition**: Separate compilation enables different optimizations for forward vs backward. Clear boundary for saving activations.

### Simple Partitioner

**Strategy**: Topological ordering. Tag nodes as `is_forward` or `is_backward`. Split at boundary.

**Activation Saving**: Conservative - save all intermediate values needed by backward.

**When to use**: Default. Works for most cases. Predictable behavior.

### Min-Cut Partitioner

**Strategy**: Formulate as graph cut problem. Minimize memory (edges crossing cut = saved activations).

**Recomputation**: Some forward ops recomputed in backward instead of saving activations. Trade compute for memory.

**When to use**: Memory-constrained training (large models, large batch sizes). Activation checkpointing enabled.

**Config**: `torch._functorch.config.use_min_cut_rematerialization = True`.

**Algorithm**: Uses min-cut solver on dependency graph. Cost model estimates memory vs recompute trade-off.

**Steering**: Query "min-cut partitioning" for algorithm details.

### Activation Saving

**Concept**: Forward must save intermediate tensors needed by backward's autograd computation.

**Decision**: Which activations to save vs recompute.

**Simple**: Save everything (safe, high memory).
**Min-cut**: Save only necessary (lower memory, more compute).

**Integration**: `_activation_checkpointing/` directory - policies for activation checkpointing.

**Files**: `partitioners.py` (partitioning algorithms), `_activation_checkpointing/` (checkpointing integration).

## vmap System

**Purpose**: Automatically vectorize functions over batch dimensions.

### BatchedTensor

**Concept**: Wrapper that adds implicit batch dimension to tensors.

**Structure**:
- Underlying tensor (batched)
- Batch dimension index (which dim is batched)
- Batch size

**Operations**: PyTorch ops have batching rules that handle BatchedTensor inputs.

**Example**:
```python
# User code
vmap(torch.sin)(x)  # x: [B, N]

# What happens
x_batched = BatchedTensor(x, bdim=0)  # Mark dim 0 as batch
result_batched = torch.sin(x_batched) # Batching rule: sin applies element-wise, batch dim unchanged
result = result_batched.unwrap()      # [B, N]
```

### Batching Rules

**Concept**: Each PyTorch op has a rule for how to handle batched inputs.

**Examples**:
- `sin(BatchedTensor)`: Apply sin element-wise, preserve batch dim
- `matmul(BatchedTensor, BatchedTensor)`: Handle batch dims correctly, may need transpose
- `sum(BatchedTensor, dim)`: Adjust dim index if summing over batch dim

**Adding rules**: For custom ops, register batching rule with `torch.library.register_vmap`.

**Steering**: Query "vmap batching rules" for existing rules and how to add new ones.

### Vmap Nesting

**Concept**: Multiple vmap calls create nested batch dimensions.

**Example**:
```python
vmap(vmap(fn))(x)  # x: [B1, B2, N]
# Inner vmap: batch over dim 1 (size B2)
# Outer vmap: batch over dim 0 (size B1)
```

**Implementation**: BatchedTensor tracks nesting level. Batching rules handle multiple batch dims.

**Files**: `vmap.py` (vmap implementation, batching rules), `_C.pyi` (C++ vmap bindings).

## Key Classes and Data Structures

### AOTInput Hierarchy

**Purpose**: Describe where graph inputs come from.

```python
AOTInput (base)
├── PlainAOTInput        # Regular user input
├── ParamAOTInput        # Model parameter
├── BufferAOTInput       # Model buffer
└── DifferentiableAOTInput  # Can be wrapped by GradAOTOutput
```

**Methods**:
- `is_param()`: True for parameters
- `is_buffer()`: True for buffers
- `is_tangent()`: True for gradient inputs
- `expr()`: Returns expression for this input

**Usage**: Runtime wrapper uses these to correctly bind arguments.

**File**: `schemas.py`.

### AOTOutput

**Purpose**: Describe where graph outputs go.

**Types**:
- Regular output (user-visible)
- Gradient output (wrapped as GradAOTOutput)
- Mutated input (output aliases input)

**Methods**:
- `is_grad()`: True if this is a gradient
- `expr()`: Returns expression for this output

**File**: `schemas.py`.

### GraphSignature

**Purpose**: Contract for a compiled graph - what inputs/outputs mean, what mutations occur.

**Fields**:
- `input_specs`: List of AOTInput descriptors
- `output_specs`: List of AOTOutput descriptors
- `input_mutations`: Which inputs are mutated
- `output_aliasing`: Which outputs alias inputs
- `view_mutations`: Mutations through views

**Usage**: Runtime wrapper uses signature to correctly call compiled graph and apply mutations.

**File**: `schemas.py`.

### FunctionalModule and Friends

**Purpose**: Represent nn.Module in functional form (stateless).

**FunctionalModule**: Created by `make_functional(module)`.
- Takes params as first argument
- Calls module with those params

**FunctionalModuleWithBuffers**: Created by `make_functional_with_buffers(module)`.
- Takes params and buffers as first two arguments
- Useful when module has stateful buffers (e.g., BatchNorm)

**Usage**: Enables vmap over parameters for per-sample operations.

**Example**:
```python
model = nn.Linear(10, 1)
func_model, params = make_functional(model)

# Compute gradient w.r.t. params for one sample
grad_fn = grad(lambda p, x, y: loss(func_model(p, x), y))
grads = grad_fn(params, input, target)

# Compute per-sample gradients for batch
per_sample_grads = vmap(grad_fn)(params, batch_inputs, batch_targets)
```

**Files**: `make_functional.py` (conversion), `functional_call.py` (calling), `apis.py` (public exports).

## AOT Configuration

**Key Config Options** (in `config.py`):

- `use_min_cut_rematerialization`: Enable min-cut partitioning (default False)
- `debug_joint`: Print joint graph (default False)
- `debug_graphs`: Print all graphs (default False)

**Runtime Config** (via `AOTConfig` in `schemas.py`):

- `fw_compiler`: Backend compiler for forward graph
- `bw_compiler`: Backend compiler for backward graph (or None)
- `partition_fn`: Partitioning function (simple or min-cut)
- `decompositions`: Op decompositions to apply
- `num_params_buffers`: How many inputs are params/buffers
- `keep_inference_input_mutations`: Whether to preserve mutations in inference

**Usage**: `aot_function()` and `aot_module()` take `fw_compiler`, `bw_compiler`, and `partition_fn` arguments.

## Compilation Flow Summary

**Entry**: User calls `torch.compile(model)` → Dynamo captures → calls AOT backend.

**AOT Flow**:
1. `aot_module_simplified()` or `aot_function()` entry point
2. Create `AOTConfig` with compiler settings
3. Call `aot_stage1_graph_capture()`:
   - Functionalize graph
   - Create joint graph (if training)
   - Analyze inputs/outputs
4. Call partitioner to split graph
5. Call `aot_stage2_compile()`:
   - Apply post-grad passes
   - Compile forward/backward with backend
   - Create runtime wrappers
6. Return compiled function/module

**Key Entry Points**:
- `aot_function(fn, fw_compiler, bw_compiler)`: Compile a function
- `aot_module(mod, fw_compiler, bw_compiler)`: Compile an nn.Module
- `aot_module_simplified(mod, ...)`: Simplified version used by torch.compile

**Files**: `aot_autograd.py` (entry points), `graph_compile.py` (stages), `runtime_wrappers.py` (generated wrappers).

## Gradient Transforms

**grad(fn)**: Returns function that computes gradient of fn w.r.t. first argument.

**vjp(fn, inputs)**: Vector-Jacobian product. Returns `(outputs, vjp_fn)` where `vjp_fn(v)` computes `v^T * J`.

**jvp(fn, primals, tangents)**: Jacobian-vector product. Returns `(outputs, jvp_outputs)` where `jvp_outputs = J * tangents`.

**jacrev(fn)**: Full Jacobian via reverse mode (uses vjp internally).

**jacfwd(fn)**: Full Jacobian via forward mode (uses jvp internally).

**Composition**: These compose naturally:
```python
# Hessian via jacrev(grad)
hessian = jacrev(grad(fn))

# Per-sample gradients via vmap(grad)
per_sample = vmap(grad(loss_fn))
```

**AOT Usage**: `create_joint()` in AOT uses grad internally to create backward graph.

**Files**: `apis.py` (public APIs), `eager_transforms.py` (implementations), `_C.pyi` (C++ transforms).

## Where to Add Code

**Add mutation support**: `functional_utils.py` - extend FunctionalTensor to handle new op.

**Add post-grad pass**: `torch/_inductor/fx_passes/post_grad.py` - add pattern matcher.

**Add batching rule**: Register with `torch.library.register_vmap` in op definition.

**Modify partitioning**: `partitioners.py` - customize `min_cut_rematerialization_partition()`.

**Extend AOT config**: `schemas.py` - add fields to `AOTConfig`.

**Testing**: `test/functorch/test_aotdispatch.py` (AOT), `test/functorch/test_vmap.py` (vmap), `test/functorch/test_ops.py` (grad transforms).

## Steering MCP Queries

For detailed API docs, query steering:

- `query_api_docs(query="aot_function")` - AOT entry points
- `query_api_docs(query="make_functional")` - Functional modules
- `query_api_docs(query="vmap")` - Batching details
- `query_api_docs(query="grad")` - Gradient transforms
- `query_api_docs(query="AOTInput")` - Input descriptor classes
- `query_api_docs(query="FunctionalTensor")` - Functionalization internals
- `query_api_docs(query="min_cut")` - Partitioning algorithms

## IR Transformation

AOT transforms graphs through multiple IR levels:

### Stage 2.1: Functionalization

**Input**: Full ATen IR (may have mutations like `x.add_()`)

**Output**: **Core ATen IR** (functional: `x = x + 1`)

**Mechanism**: FunctionalTensor wrapping, ViewMeta tracking

**What it does**: Removes in-place mutations and aliases to produce a functional graph. This is WHY Core ATen IR is functional - functionalization creates it.

### Stage 2.2: Decompositions

**Input**: Core ATen IR

**Applies**: `core_aten_decompositions()` from `torch/_decomp/__init__.py`

**Output**: Simpler Core ATen ops (some may decompose to prims)

**Examples**:
- `silu` → `x * sigmoid(x)`
- `gelu` → erf-based formula

### Key files

- `torch/_decomp/__init__.py`: `core_aten_decompositions()`
- `torch/export/decomp_utils.py`: CustomDecompTable
- `torch/_functorch/_aot_autograd/graph_compile.py`: Application point

### Core ATen ops

- Marked with `tags: core` in `native_functions.yaml`
- These ARE Core ATen IR - the functional subset
- CompositeImplicitAutograd ops decompose TO these core ops

### Refs, Prims, and Decompositions

- **Prims** (`torch._prims`): Primitive operations (e.g., `prims.add`, `prims.mul`)
- **Refs** (`torch._refs`): Python reference implementations that use prims; register themselves as decompositions via `@register_decomposition`
- **Decomps** (`torch._decomp/decompositions.py`): Manual decompositions
- **Key insight**: Refs ARE decompositions, not a separate category
- **No duplicates**: `torch/_decomp/__init__.py` raises `RuntimeError` if same op registered twice

---

**Next**: See `COMMON-PATTERNS.md` for implementation patterns and code examples.
