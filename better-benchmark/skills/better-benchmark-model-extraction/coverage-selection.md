# Select model coverage

## Choose full or representative depth

If the user requires the complete model graph, retain full depth. Otherwise,
inspect configuration and implementation for repeated blocks and exceptions:
attention variants, dense/MoE blocks, head geometry, shared KV, cross-layer
state, per-layer inputs, first/last-layer behavior, and modality paths.

Choose the smallest contiguous slice or complete repeating cycle covering the
required block types and producer/consumer boundaries. Record original layer
indices/types and omitted computation. Do not assume the first few layers cover
every variant. Confirm the coverage hypothesis against source and the extracted
operation/shape set: wiring, shapes, strides, constants, and state can create
new patterns even when blocks look similar.

## Prepare the configuration before construction

Reduce depth before allocating weights. Preserve widths, vocabulary, head and
expert dimensions, selected-layer semantics, and the workload contract. Update
coupled layer-type lists, per-layer overrides, sharing indices, and other
depth-dependent fields consistently. Verify config initialization does not
silently replace selected layer types.

Construct directly in the intended dtype where supported, avoiding a full model
followed by truncation or an FP32 model plus a BF16 copy. Use eval/no-grad for
inference. Fake/meta capture and selective materialization are not assumed
capabilities; use them only if existing repository APIs support the workload.

Keep smoke inputs separate from production inputs. In particular, a shorter
sequence can change local/global attention behavior and is not evidence of
production-shape coverage. Output truncation can reduce allocation when the
contract permits it, but retain the task head when labeling a task forward and
document the resulting head/kernel coverage.

## Interpret accounting

Canonical merging deduplicates regions after extraction; it does not select
source layers. A reduced-depth manifest counts only the captured slice. Do not
relabel or multiply it into full-depth accounting without a separate supported
accounting workflow.
