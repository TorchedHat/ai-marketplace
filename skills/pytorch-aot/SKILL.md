---
name: pytorch-aot
description: Expert guidance for PyTorch functorch and AOT Autograd development. Covers torch/_functorch architecture, AOT Autograd pipeline (functionalization, joint graphs, partitioning, post-grad passes), vmap/batching, functional transforms (grad, vjp, jvp, jacrev, jacfwd), make_functional, functional_call, activation checkpointing, and implementation patterns. Use for implementing AOT features, understanding functorch internals, and adding functionalization/partitioning support. For debugging, use compile-trace-aot.
---

# PyTorch Functorch & AOT Autograd Development

Expert guidance for implementing features in PyTorch's functorch system (`torch/_functorch`), with focus on AOT (Ahead-of-Time) Autograd.

## Quick Start

This skill provides **implementation and architecture guidance** for developing in `torch/_functorch`. Use it when you need to understand how AOT works internally or implement new features. For debugging compilation failures, use `compile-trace-aot` instead. For detailed API signatures, query steering MCP.

## Skill Navigation

Use the right tool for your task:

| Your Goal | Use This | What It Provides |
|-----------|----------|------------------|
| **Implement feature** in torch/_functorch | `pytorch-aot` (this skill) | Architecture, code patterns, where to add code |
| **Debug AOT failure** with TORCH_LOGS | `compile-trace-aot` skill | Debugging workflows, log interpretation, tracing |
| **Understand API signature** for function | Steering MCP query | Detailed API docs, function signatures, examples |

**Navigation pattern from compile-overview**:
```
bisect shows backend='aot_*'
  → compile-trace-aot (debug the issue)
  → pytorch-aot (implement the fix)
```

## What is Functorch?

**Functorch** is PyTorch's functional programming system, providing JAX-inspired function transformations: `vmap` (batching), `grad` (gradients), `jacrev`/`jacfwd` (Jacobians), and functional module utilities (`make_functional`, `functional_call`). It enables composable transformations like computing per-sample gradients with `vmap(grad(loss_fn))`.

**AOT Autograd** is functorch's compilation component, serving as the critical middle layer in `torch.compile`:

```
Dynamo (bytecode → FX graph)
  → AOT Autograd (training transformations)
  → Inductor (kernel generation)
```

AOT transforms FX graphs for training: removes mutations (functionalization), creates joint forward+backward graphs, partitions them, and applies post-grad optimizations (GEMM fusion, attention fusion).

## Core Concepts

### AOT Autograd Pipeline

**Purpose**: Transform FX graphs for efficient training execution.

**Stages**:
1. **Functionalization**: Convert mutations (`x.add_()`) to functional ops (`x + 1`), track aliases via ViewMeta
2. **Joint Graph**: Combine forward + backward into single FX graph (when `requires_grad=True`)
3. **Partitioning**: Split joint graph into separate forward and backward graphs, decide what activations to save vs recompute
4. **Post-Grad Passes**: Optimize both graphs (GEMM fusion, Conv-BN fusion, etc.)

**Key insight**: AOT sees the entire training computation at once, enabling cross-forward/backward optimizations impossible in eager mode.

### Functionalization

**What**: Removes in-place mutations and aliases from FX graphs, converting them to functional equivalents.

**Why**: Backends like Inductor assume functional graphs. Mutations complicate analysis and code generation.

**How**: Wraps tensors in `FunctionalTensor`, intercepts mutation ops, records them as metadata, emits functional ops instead. ViewMeta system tracks view operations for alias reconstruction.

**Example**: `x.mul_(2)` becomes `x_new = x * 2` with metadata recording the mutation.

### vmap (Vectorizing Map)

**What**: Automatically vectorizes functions over batch dimensions.

**Usage**: `vmap(fn, in_dims=0, out_dims=0)` - maps `fn` over dimension 0 of inputs.

**Mechanism**: Wraps inputs in `BatchedTensor`, pushes batching through PyTorch ops via batching rules.

**Composition**: Combine with `grad` for per-sample gradients: `vmap(grad(loss_fn))(params, inputs, targets)`.

### Gradient Transforms

**grad**: Computes gradients of scalar output w.r.t. inputs.
**vjp/jvp**: Vector-Jacobian product (reverse mode) and Jacobian-vector product (forward mode).
**jacrev/jacfwd**: Full Jacobian computation (reverse or forward mode).

**AOT integration**: AOT uses these transforms internally when creating joint graphs.

### Functional Modules

**make_functional(module)**: Converts `nn.Module` to functional form - returns `(functional_fn, params)`.

**functional_call(module, params, inputs)**: Calls module with explicit parameters without modifying module state.

**Use case**: Per-sample gradients with vmap over parameters, meta-learning, functional optimization.

## Key Files

```
torch/_functorch/
├── aot_autograd.py                    # Main AOT entry: aot_function, aot_module
├── _aot_autograd/
│   ├── graph_capture_wrappers.py      # Functionalization, joint graph creation
│   ├── graph_compile.py               # Stage 1 & 2 compilation logic
│   ├── runtime_wrappers.py            # Generated runtime code
│   ├── schemas.py                     # AOTInput, AOTOutput, GraphSignature
│   ├── functional_utils.py            # Mutation tracking, ViewMeta
│   ├── input_output_analysis.py       # I/O descriptor analysis
│   └── collect_metadata_analysis.py   # Metadata collection
├── partitioners.py                    # Min-cut partitioning, recomputation
├── vmap.py                            # vmap implementation, BatchedTensor
├── apis.py                            # Public APIs: grad, vjp, jvp, jacrev, jacfwd
├── make_functional.py                 # Module → functional conversion
├── functional_call.py                 # Functional module calling
├── _activation_checkpointing/         # Gradient checkpointing integration
├── compile_utils.py                   # Compilation utilities
└── config.py                          # Functorch config options
```

## Common Development Tasks

### Understand AOT Pipeline Flow
**Read**: `ARCHITECTURE.md` for detailed pipeline explanation.
**Key file**: `torch/_functorch/_aot_autograd/graph_compile.py:aot_stage1_graph_capture()` and `aot_stage2_compile()`.

### Add Support for New Mutation Pattern
**Pattern**: See `COMMON-PATTERNS.md` → "Adding mutation support".
**Key file**: `torch/_functorch/_aot_autograd/functional_utils.py`.
**Test**: Add test to `test/functorch/test_aotdispatch.py`.

### Implement Post-Grad FX Pass
**Pattern**: See `COMMON-PATTERNS.md` → "Implementing post-grad pass".
**Key file**: `torch/_inductor/fx_passes/post_grad.py`.
**Integration**: Passes run after partitioning in AOT stage 2.

### Use vmap for Batched Gradients
**Pattern**: See `COMMON-PATTERNS.md` → "Using vmap for batched gradients".
**Steering**: Query "vmap batching functorch" for detailed API.
**Example**: `vmap(grad(loss_fn))(params, batch_inputs, batch_targets)`.

### Make Module Functional for Per-Sample Analysis
**Pattern**: See `COMMON-PATTERNS.md` → "Making module functional".
**Steering**: Query "make_functional functional_call" for API details.
**Use case**: Compute per-sample gradients, meta-learning.

### Debug Gradient Mismatch (Eager vs Compiled)
**Use**: `compile-trace-aot` skill (not this skill).
**Workflow**: compare outputs, inspect joint graph, verify partitioning.
**Cross-ref**: compile-trace-aot has complete debugging workflows.

### Optimize Memory with Min-Cut Partitioning
**Pattern**: See `COMMON-PATTERNS.md` → "Custom partitioning policy".
**Config**: `torch._functorch.config.use_min_cut_rematerialization = True`.
**Steering**: Query "min-cut partitioning" for algorithm details.

## Progressive Disclosure

For deeper understanding, see:

- **ARCHITECTURE.md**: Detailed architecture of AOT pipeline, functionalization system, partitioning strategies, vmap internals, key classes (AOTInput/AOTOutput, FunctionalTensor, GraphSignature).

- **COMMON-PATTERNS.md**: Implementation patterns with code examples for adding mutation support, implementing post-grad passes, using vmap, making modules functional, extending functionalization, custom partitioning.

## When to Use Steering MCP

Query steering MCP for:
- **Detailed API signatures**: `query_api_docs(query="aot_function")`
- **Implementation examples**: `query_api_docs(query="make_functional")`
- **Class method details**: `query_api_docs(query="AOTInput")`
- **Batching rules**: `query_api_docs(query="vmap")`

Steering provides comprehensive function signatures, parameters, return types, and usage examples from the codebase.

## When to Use compile-trace-aot

Use `compile-trace-aot` skill for:
- **Debugging compilation failures**: Gradient errors, memory issues, functionalization bugs
- **TORCH_LOGS setup**: Which flags to enable, how to interpret output
- **Output file analysis**: Reading joint graphs, forward/backward graphs
- **Debugging workflows**: Step-by-step troubleshooting guides
- **Common issues**: Known problems and solutions

**Cross-reference**: compile-trace-aot (646 lines) covers all tracing and debugging workflows for the AOT stage.

## Development Principles

1. **Respect Functionalization Semantics**: Mutations must be tracked accurately. ViewMeta chains must preserve aliasing relationships.

2. **Maintain Metadata Consistency**: AOTInput/AOTOutput descriptors must correctly classify parameters, buffers, and tangents. GraphSignature must reflect all mutations and aliases.

3. **Preserve Autograd Correctness**: Joint graph must correctly compute gradients. Partitioning must save necessary activations.

4. **Test Training and Inference**: AOT behaves differently when `requires_grad=True` (joint graph) vs `requires_grad=False` (inference path).

5. **Use Descriptive Types**: Prefer `PlainAOTInput`, `ParamAOTInput`, `BufferAOTInput` over raw tuples. Use `GraphSignature` for I/O contracts.

6. **Point to Steering for Details**: This skill provides orientation. For API details, query steering MCP rather than duplicating here.

## Getting Help

- **Implementation guidance**: Read ARCHITECTURE.md and COMMON-PATTERNS.md in this skill.
- **Debugging help**: Load `compile-trace-aot` skill for debugging workflows.
- **API details**: Query steering MCP for function signatures and examples.
- **Testing**: See `test/functorch/` for test examples.
- **Broader context**: Load `compile-overview` skill for pipeline understanding.

---

**Related Skills**: compile-trace-aot (debugging), pytorch-dynamo (frontend), pytorch-inductor (backend), compile-overview (pipeline routing)
