---
name: better-benchmark-model-extraction
description: Extract inference fusion regions from Hugging Face or other PyTorch models into better-benchmark while preserving kernel shapes and limiting capture memory. Use when adding model coverage, choosing representative layers, or importing TORCH_TRACE/tlparse runnables into the corpus. Generates and validates full graphs, input sidecars, canonical repros, shape points, and model attribution.
---

# Extract model coverage into better-benchmark

Use existing better-benchmark APIs to generate and validate inference corpus
artifacts. This workflow follows source compile, tlparse export, runnable
replay/capture, staging validation, and final integration. Existing exports
start at replay. Training capture needs a separate loss/backward workflow.

## Example requests

- "Plan a representative capture for the local Qwen3 model at batch 1,
  sequence 2048." Inspect source/config and propose coverage and resource needs;
  stop before GPU execution or corpus changes.
- "Add inference coverage using BF16 and SDPA at batch 1, sequence 2048."
  Establish the workload, select coverage, then capture and validate artifacts.
- "Import the runnable from my existing tlparse export." Establish its
  provenance and captured scope, then replay without reconstructing the model.

## Requirements

Locate better-benchmark, the relevant model source/config, and the configured
Python environment. Read applicable checkout instructions, including
`AGENT_INSTRUCTIONS.md`, `CLAUDE.md`, and the Extraction and Model accounting
sections of `README.md` when present. Check git state, installed import paths,
existing captures, and GPU capacity; preserve unrelated work.

Execution requires compatible PyTorch/CUDA and an available GPU with sufficient
memory. Export requires tlparse. HF construction requires transformers and
config access; use huggingface_hub when resolving remote revisions. Inspect the
installed environment before changing it. Do not reinstall packages by default
or build PyTorch without an explicit request.

## Workflow

### 1. Establish the workload

Resolve choices from the request, model source, and corpus conventions. Ask
only for missing information that materially changes the workload; record
routine assumptions. Retain a workload contract outside the repository:

- Model/config identity, immutable revision, original config bytes and digest,
  and actual base/text/task class.
- Original/captured depth, original layer indices/types, unchanged widths,
  vocabulary, head geometry, and expert dimensions.
- Mode, parameter/input dtypes, input dimensions, native backend dispatch,
  cache/past state, output policy, seed/initialization, and constant folding.

Use config-based random weights and legal synthetic inputs when sufficient;
avoid pretrained weight/tokenizer downloads by default. Follow the repository's
BF16 inference policy where supported, preserving integer/bool inputs and
explicit internal higher-precision operations. For an existing export, record
available provenance and identify missing evidence rather than inventing it.

Do not guess, ask user what they want. Continue asking questions until either
every value is specieed by a default in a config or the user has answered every
remaining field.

### 2. Select coverage

For a new source capture, follow [coverage selection](coverage-selection.md)
before allocating weights. For an existing runnable, verify its captured scope
against the workload contract instead of constructing the source model.
For a planning request, report the contract, proposed coverage, resource needs,
and unresolved choices here, then stop.

### 3. Capture, validate, and integrate

Follow the [artifact workflow](artifact-workflow.md) for the command sequence,
runnable replay, capture gates, saved-artifact execution, and final integration.
Consult the [API reference](api-reference.md) for entry points and return values;
verify their local signatures before invocation.

### 4. Report results

Report actual captured scope and limitations, new/reused canonical patterns and
shape points, provenance/report locations, and validation outcomes. Use the
[completion evidence](artifact-workflow.md#completion-evidence) table to check
coverage of the report. For a blocker, retain the failing invocation, artifact,
exception, and missing acceptance checks.

## Scope and failure rules

- Keep handwritten workload setup and short invocations of existing APIs in
  retained scratch storage outside the repository. Corpus additions contain
  generator-produced code/data plus documentation; record each code producer.
  Do not add a permanent model-specific driver or importer.
- Use structured artifacts, existing loaders, and live FX graphs. Do not
  regex/AST-parse generated source or hand-edit graphs, repros, manifests,
  metadata, shapes, or attribution.
- Access failures, insufficient memory, invalid inputs, drops, failed
  roundtrips, and failed artifact execution are blockers. Do not substitute a
  different model, narrower widths, or another backend to conceal a failure.
  Shared capture/importer/validator/compiler development needs its own scope.
- A smoke compile, in-memory success, or metadata-only check cannot replace
  production-shape and saved-artifact execution. Preserve validators and
  explain pre-existing failures with baseline evidence.
- Extraction checks execution and corpus structure, not numerical accuracy or
  performance. Do not invoke `validate_full_graph_numerics` or
  `run_numerics_gate` for extraction alone; missing FP64 support is not a gate.
  Numerical comparisons, oracle authoring, and benchmarking require separate
  requested scope. Benchmarking uses the repository's shared runners and GPU
  isolation.
