# Fusion Decisions

How Inductor decides which operations share a kernel. Fusion is the primary
source of performance — it eliminates memory round-trips between operations.

For how fused operations become executable code, see [CODEGEN.md](CODEGEN.md).

## Legality vs Scoring

Fusion is a two-phase process. These are architecturally distinct:

1. **Legality** (`can_fuse`) — binary gate: can these two nodes legally share
   a kernel without changing semantics?
2. **Scoring** — among legal candidates, which fusion saves the most memory
   traffic? Higher score = more shared inputs eliminated.

Debugging "why didn't these fuse?" always starts with `can_fuse`. Scoring
only matters when multiple legal fusions compete.

## `can_fuse` Decision Tree

`SIMDScheduling.can_fuse()` in `codegen/simd.py` is a multi-branch decision
tree. The common misconception is that fusion requires "same numel" — the
actual logic has at least five branches:

1. **ForeachKernel delegation** — multi-tensor ops (`torch._foreach_*`) have
   their own fusion rules and are handled first.

2. **Split-scan incompatibility** — split scans cannot fuse with reductions.

3. **Reduction + Reduction** — requires matching `(numel, rnumel)`. If that
   fails, falls through to:
   - `MixOrderReduction.can_fuse` — can the two reductions share a
     mixed-order loop?
   - Staged-plan check — are these reductions over the same logical elements
     at different granularities, with compatible pointwise domains and source
     relationships?
   - Native-matmul tiling compatibility

4. **Non-reduction pairs** — requires `numel` and `rnumel` match, with
   exceptions for template/prologue fusion and tiling compatibility checks.

5. **Reduction + Pointwise epilogue** — when one node is a reduction and the
   other is pointwise, args are swapped and re-checked. Standard epilogues use
   a two-pass kernel; multi-domain staged fusion can additionally place
   pointwise consumers in a derived domain when their accesses and layouts are
   proven compatible.

## Key Data Structure: `MemoryDep`

Fusion operates on **Schedule IR** (see [ARCHITECTURE.md](ARCHITECTURE.md#ir-levels)).
`MemoryDep` objects are extracted from Node IR's `inner_fn` during scheduler
initialization and attached to Schedule IR nodes. All fusion decisions
examine these objects from `dependencies.py`:

```python
class MemoryDep:
    name: str                            # Buffer name (e.g., "buf0")
    index: sympy.Expr                    # Access pattern (e.g., 8192*d0 + d1)
    var_names: tuple[sympy.Symbol, ...]  # Iteration variables (d0, d1, ...)
    size: tuple[sympy.Expr, ...]         # Extent per variable

    @property
    def ranges(self) -> dict:            # {var: extent} — the iteration domain
        return dict(zip(self.var_names, self.size))
```

The `index` field is a sympy expression encoding the access pattern — e.g.,
`8192*d0 + d1` for row-major access into a `[2048, 8192]` buffer. The
`var_names` and `size` fields define the iteration domain. Together they
fully describe which elements a node reads or writes and in what order.

This is the raw material for any fusion analysis. When `can_fuse` checks
"compatible iteration space," it is comparing these fields across nodes.
When extending fusion to new patterns, the data is already here — the
question is whether the analysis recognizes the compatibility.

Other dependency types:
- `StarDep` — entire buffer, unknown access pattern (prevents fusion analysis)
- `WeakDep` — ordering only, no data dependency

## Fusion Patterns

**Vertical (Producer-Consumer)**:
```python
x.relu().add(1)
# Fused: single kernel, no intermediate buffer materialized
```

**Horizontal (Consumer-Consumer)**:
```python
y1 = x.relu(); y2 = x.sigmoid()
# Fused: load x once, compute both outputs
```

**Reduction + Epilogue**:
```python
mean = x.mean(keepdim=True); norm = x - mean
# Fused: reduction pass computes mean, then pointwise pass subtracts it
# Enabled by DisableReduction/EnableReduction schedule markers (see CODEGEN.md)
```

**Multi-Domain Staged Fusion**:
```python
# A reduction may feed a grouped reduction or an epilogue at a different
# resolution. A staged plan lets those domains share one kernel when proven safe.
```

## Multi-Domain Staged Fusion

Most fusion uses one common iteration domain, so matching groups and ordinary
dependency checks are sufficient. A reduction can also feed work at a different
granularity: a grouped reduction, or a pointwise epilogue that consumes a
derived portion of the parent tile. Those cases require an explicit staged plan
rather than a relaxed version of ordinary fusion.

### Staged Fusion Contract

1. **Prove dataflow, not just shape.** A shared buffer name or equal element
   count does not show that a derived-domain read reaches the producer value
   intended by the original graph. Fusion must prove the source-to-consumer
   index mapping.
2. **Record the stage structure.** The plan identifies parent work, optional
   grouped-reduction work, derived epilogue work, their order, and the values
   that must be available across those stages.
3. **Preserve observable ordering.** Aliasing, mutation, indirect accesses,
   ambiguous writers, non-leaf outputs, and dependencies that would be moved
   across a stage boundary make the plan unsafe.
4. **Validate after loop changes.** Fusion planning precedes transformations
   that may normalize or merge loops. Before specialized codegen, the plan is
   rebuilt or validated against the final loop structure.
5. **Decline safely.** An unprovable staged fusion is not a compiler failure:
   Inductor retains the ordinary schedule and its materialized intermediates.

This distinction matters when debugging: normal fusion legality asks whether
nodes may share a kernel; staged fusion additionally asks whether their
different coordinate systems and execution times can be reconciled exactly.

### What It Covers

- A dependent reduction at a different granularity from its parent reduction.
- A standalone reduction followed by a derived-resolution pointwise epilogue.
- A nested reduction with a compatible derived-resolution epilogue.

These are intentionally narrow patterns. They do not make arbitrary
cross-domain producer-consumer fusion legal.

### Interaction with Scheduling Choices

Generic fusion benchmarking and multi-kernel grouping assume ordinary node
shapes and schedules. A staged group may therefore decline those optional
optimizations when they cannot represent its derived domain. That is a
performance trade-off, not a change to the staged fusion's correctness
contract.

For how a validated plan becomes a kernel, see
[CODEGEN.md — Derived-Domain Staged Codegen](CODEGEN.md#derived-domain-staged-codegen).

## Scheduler Node Types

The scheduler wraps IR nodes in its own type hierarchy for fusion tracking:

```
BaseSchedulerNode
├── SchedulerNode              — single IR operation
├── FusedSchedulerNode         — multiple operations fused into one kernel
│   └── FusedStagedReduction   — plan-carrying multi-domain fusion
│       └── FusedNestedReductions — dependent grouped-reduction specialization
├── ExternKernelSchedulerNode  — external call (matmul, conv)
├── NopKernelSchedulerNode     — eliminated operation
└── ForeachKernelSchedulerNode — multi-tensor operation
```

## Fusion to Codegen Dispatch

Once fusion decisions are made, the scheduler drives code generation. Fusion
owns both the decision (above) AND the dispatch — routing each fused node
group to the appropriate codegen handler.

### The Dispatch Loop

`Scheduler._codegen()` iterates the fused schedule in order and routes each
node to its backend handler based on type:

```
for node in nodes:
    enter_context(node)              # device guards, stream switching
    buffer_names_to_free.update(node.last_usage)

    if node.is_template():           → codegen_template()
    elif node.is_extern():           → codegen_extern_call()
    elif node.is_foreach():          → codegen_combo_kernel()
    elif FusedStagedReduction:       → specialized staged codegen
    elif specialized reduction group: → its dedicated codegen path
    elif ordinary fused/single node: → backend codegen
    else:                            → mark the eliminated operation run
```

The type-based routing pattern is stable. Specific node types may be added or
removed, but the architecture — iterate schedule order, dispatch by type to
the registered backend handler — is fundamental.

### Buffer Lifetime Handoff

Before codegen begins, `compute_last_usage()` walks the schedule in reverse
to determine when each buffer is last read. During the dispatch loop,
`buffer_names_to_free` is updated per node, and `codegen_free()` emits
`FreeIfNotReusedLine` into the wrapper. This is the scheduler's contract with
memory planning — see [MEMORY-PLANNING.md](MEMORY-PLANNING.md).

### Node Schedule Construction

For SIMD nodes (the common path), `generate_node_schedule()` creates a
structured schedule that interleaves `SchedulerNode` objects with control flow
markers. This is where the fusion decisions from Part 1 become a concrete
execution plan.

Nodes are classified into two categories:

- **Main body**: Fits inside the reduction loop
  (`node_numel == numel and node_rnumel == rnumel`, or the node's entire
  iteration space is `numel * rnumel` with `rnumel == 1`).

- **Epilogue**: Pointwise operations that consume reduction output
  (`node_numel == numel and node_rnumel == 1 and rnumel != 1`). These get
  bracketed by `DisableReduction`/`EnableReduction` markers.

The marker system is how reduction + epilogue fusion (decided by `can_fuse`
above) becomes executable: the scheduler decides the fusion is legal, then
`generate_node_schedule()` arranges the nodes so the reduction runs first,
markers flush the loop, and the epilogue runs as pointwise code consuming
the reduction result — all in one kernel.

Staged reductions extend this structure with explicit parent, optional grouped,
and derived epilogue stages. The scheduler decides which stages are legal, and
specialized codegen emits the corresponding structured schedule.

### The Bridge to Kernel Codegen

The SIMD path flows through a chain of methods that progressively narrow
from "fused node group" to "kernel object being populated":

```
codegen_node(node)
  unwrap FusedSchedulerNode → list of SchedulerNodes
  → _codegen_nodes(nodes)
      determine (numel, rnumel) from the reduction node
      generate_node_schedule() → structured schedule with markers
      → codegen_node_schedule(features)
          compute tiling
          create kernel object (e.g. TritonKernel)
          codegen_node_schedule_with_kernel() → traces nodes into kernel
          kernel.codegen_kernel() → generates source string
          define_kernel() → assigns name, writes to wrapper
          mark_run() → allocates output buffers
          call_kernel() → emits kernel launch into wrapper
```

`codegen_node_schedule_with_kernel()` drives the two-pass codegen mechanism
described in [CODEGEN.md — Two-Pass Schedule Codegen](CODEGEN.md#two-pass-schedule-codegen).

### mark_run and Output Allocation

After kernel source is generated, `mark_run()` calls `buf.allocate()` for
each output buffer. This emits `AllocateLine` into the wrapper — the starting
point for memory planning (see
[MEMORY-PLANNING.md](MEMORY-PLANNING.md)). Outputs are allocated after source
generation but before the kernel call is emitted, ensuring buffers exist when
the kernel runs.

## Extending Fusion

Use this playbook when an intended fusion cannot be expressed by an existing
fusion family. Begin with the program transformation and its invariants, then
choose the smallest extension that preserves them. Do not start by relaxing a
generic fusion predicate: a broader predicate is difficult to audit and can
silently make unrelated graphs incorrect.

### 1. Classify the Transformation

First decide which existing execution model the transformation needs:

- **Ordinary fusion**: all operations share one iteration domain and the
  normal producer-consumer or sibling rules describe the relationship.
- **Staged fusion**: operations need more than one compatible iteration domain,
  such as a parent reduction and a derived consumer domain.
- **Template or external-kernel epilogue fusion**: a specialized kernel owns
  part of the computation and exposes a supported epilogue interface.
- **Graph rewrite or pattern replacement**: the desired transformation changes
  the graph before ordinary scheduling is the right abstraction.

Prefer an existing family when its contract fits. Introduce a new
plan-carrying representation only when ordinary fused groups cannot faithfully
describe the required execution domains or ordering.

In the current scheduler, `FusedStagedReduction` is the common plan-carrying
representation. `FusedNestedReductions` specializes it for a dependent grouped
reduction nested in a parent reduction. These are useful anchors when tracing
the implementation; the plan and its semantic invariants, rather than the
types' particular fields, are the lasting contract.

### 2. Prove Legality Before Considering Speed

State the semantic proof the fusion requires. At minimum, account for:

- **Dataflow**: every consumer reads the producer value intended by the
  original graph. Equal element counts or a shared buffer name are not enough
  when iteration domains differ.
- **Index mapping**: express accesses in a common coordinate system and prove
  the source-to-consumer mapping. Unknown or indirect accesses are not a basis
  for a precise equivalence proof.
- **Ordering**: preserve dependencies that make reordering observable,
  including mutation and aliasing constraints.
- **Lifetime**: a value forwarded in registers must be live at the consumer;
  otherwise it must be materialized or safely reloaded.
- **Backend capability**: only form the fusion when its target code generator
  can represent the resulting schedule and layout.

Encode the proof as explicit planner facts or a plan that codegen consumes.
Avoid duplicating loosely similar index arithmetic in legality and codegen;
those implementations can drift apart while still appearing individually
reasonable.

### 3. Define the Schedule–Codegen Contract

Fusion analysis and codegen observe the same lowered computation at different
times and sometimes in different loop representations. Give both sides one
source of truth for any new relationship: a shared pure mapping, or facts
produced by planning and validated by codegen.

For multi-domain fusion, the plan should state the participating stages, their
iteration domains, ordering, and required source-value relationships. Validate
or reconstruct the plan after transformations that can normalize, merge, or
otherwise change loops. If a proof cannot be formed while deciding fusion,
decline the fusion. If a previously accepted plan cannot be reconstructed at
codegen, surface that as an internal compiler error rather than generating a
kernel from a guessed mapping.

### 4. Make Materialization and Resource Effects Explicit

Fusion changes which intermediates are kernel-local. Ensure that scheduling,
wrapper generation, and memory planning agree on whether a buffer is
materialized, forwarded, reloaded, or freed. When a derived stage changes
value resolution or masking, preserve the source program's visible indexing,
tail behavior, and numerical semantics.

### 5. Keep Profitability Separate

Legality answers whether a fusion preserves semantics; profitability answers
whether to take a legal fusion. Do not let an estimated benefit relax a
correctness condition.

When a legal fusion can regress performance, assess it using the same schedule
and codegen choices that will be emitted, and compare it with the unfused
baseline. If the cost cannot be evaluated reliably, do not let the
profitability guard reject the fusion solely on that uncertainty. Any
speculative scheduling mutation used for pricing must be fully reversible.

### 6. Land and Expand Incrementally

Start with one end-to-end topology whose legality proof, codegen path, and
tests are complete. Then broaden one dimension at a time: first quantitative
variation such as shape or partitioning, then qualitatively different layouts,
dataflow, or backends. Keep each capability boundary explicit and covered by a
test so later expansion is intentional.

### 7. Test at Three Altitudes

1. **Proof tests**: test symbolic mappings and dependency predicates directly,
   including unknown-access and ambiguous-producer rejection cases.
2. **Scheduling tests**: assert that eligible graphs form the intended fusion
   and ineligible graphs deliberately retain a safe fallback.
3. **End-to-end tests**: check numerics against an unfused reference and verify
   the relevant generated-kernel properties. Cover dynamic shapes,
   non-contiguous layouts, masked tails, and alternative reduction execution
   forms when they apply.

## Key Files

- `scheduler.py` — fusion legality and scoring, staged-plan construction,
  dispatch, and buffer-lifetime analysis
- `dependencies.py` — `MemoryDep`, `StarDep`, `WeakDep`, dependency analysis
- `codegen/simd.py` — `SIMDScheduling.can_fuse()`, `codegen_node()`,
  `generate_node_schedule()`, `codegen_node_schedule()`,
  `codegen_node_schedule_with_kernel()`

---

**For codegen internals** (kernel lifecycle, two-pass mechanism, CSE):
[CODEGEN.md](CODEGEN.md)
**For memory planning** (buffer reuse, allocation pools):
[MEMORY-PLANNING.md](MEMORY-PLANNING.md)
**For architecture overview**: [ARCHITECTURE.md](ARCHITECTURE.md)
