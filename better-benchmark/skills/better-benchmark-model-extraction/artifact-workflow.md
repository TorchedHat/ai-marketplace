# Trace export and corpus validation

Use better-benchmark's existing commands. Paths below are relative to its
checkout; replace angle-bracket placeholders before execution. See the
[API reference](api-reference.md) for existing entry points.

Use `<python>` for the user's configured interpreter, `<bundle>` for the small
reproduction directory, and `<cache>` for a run directory under a standard
cache location such as `$XDG_CACHE_HOME` or `~/.cache`. Generated graphs,
repros, manifests, and model documentation belong in better-benchmark.
The bundle contains configs, necessary generation scripts, exact commands,
and required replay inputs; logs, caches, HTML exports, and validation reports
belong under `<cache>`. Check baseline corpus invariants before mutation and
retain their result in the cache for comparison with final checks.

## 1. Compile and export

For a new source capture, resolve the material workload choices before running
it. Use separate trace/cache/export directories for smoke and production
shapes. Create the directories and select an available GPU before running the
requested production workload.

For fixed-shape pattern extraction, use
`torch.compile(..., fullgraph=False, dynamic=False)` unless the workload says
otherwise. Use `fullgraph=True` when the user requires a single complete graph.
Record the settings used for source capture and runnable replay. Do not change
the model's backend just to make fullgraph capture succeed.

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TORCH_TRACE=<cache>/trace \
TORCH_LOGS=graph_breaks,recompiles \
TORCHINDUCTOR_CACHE_DIR=<cache>/inductor \
  <python> <bundle>/workload.py

tlparse <exact-trace-log> -o <cache>/tlparse-export

<python> scripts/ingest_tlparse.py \
  --tlparse-dir <cache>/tlparse-export --use-runnables --list-only
```

Check the source output's structure, shapes, dtypes, and appropriate finiteness;
synchronize before recording success. Cache the exact trace filename and log.
Keep the original and effective config plus the command needed to regenerate
the trace in the reproduction bundle.

`--list-only` discovers runnable files. It does not ingest, capture, merge, or
validate corpus artifacts. Record the selected runnable paths and frame/compile
identities; exclude smoke runs and duplicate representations of the same graph.
For an existing export, start with listing.

Graph breaks do not block region extraction. Select all required FX regions
and record eager gaps and recompilations. Do not equate compiled graph count
with runtime call count: a cached region can execute repeatedly without a new
export. Use supported occurrence accounting or report that multiplicity is
unmeasured. Successful Dynamo capture alone is not evidence of successful
Inductor lowering or artifact execution.

If replay commands use an exported runnable, retain that exact file in the
bundle with any external input files its native loader requires. Record when
tensor payloads are omitted and synthesized. The complete tlparse HTML export
can remain in the cache; it is not a required bundle input when only one of its
runnables is consumed.

## 2. Replay into an isolated corpus

The compile/export/list sequence above identifies the artifact; it does not
create the corpus addition. Replay the selected runnable through existing
better-benchmark APIs in a fresh process using a short scratch invocation.

Use one capture path. The identity examples below use HF inference
(`suite="hf", mode="infer"`); use the agreed suite for other model sources
and the same suite in the workload directory.
Use a staging corpus root ending in `repros` with the current path-based source
inference, and verify the generated suite/mode/model attribution.

1. Load the selected exported module without running its `__main__` block.
   Use its native `load_args` with existing input-loading APIs.
2. Verify shapes, strides, dtypes, scalar values, constants, aliasing, and legal
   index bounds. Record any synthesized payloads; check execution without
   claiming identical values across different input/weight payloads.
3. Enter `temporary_capture_for_merge(staging_root, workload_name,
   suite="hf", mode="infer")` with a fresh staging root and explicit identity.
4. Call `install_capture_hook` with the context's temporary capture directory,
   an explicit label, `validate=True`, `capture_only=False`, the workload's
   constant-folding policy, and
   `graph_dir=staging_root/models/hf/infer/<workload_name>`.
5. Execute the loaded runnable and compile it under `torch.no_grad()` with
   the workload's compile settings. Synchronize and check output structure,
   shapes, dtypes, and appropriate finiteness.
6. Always call `uninstall_capture_hook()` in `finally`. Cache the finalized
   capture index before temporary cleanup. Require zero drops, a nonempty
   expected captured set, the expected saved graphs/sidecars, and
   `roundtrip == "ok"` on every full-graph sidecar.
7. Call `capture.merge()` after those gates pass and before the context exits.
   This generates the staging manifest and canonical artifacts. Do not also
   process a printed representation of the same graph as another occurrence.

Inspect the current `scripts/ingest_tlparse.py` before choosing an entry point.
When its processing helper cannot supply `graph_dir` and the required identity,
use a short capture API invocation that supplies those arguments. Prefer the
shared CLI if it supports the workload; do not create another importer.

Use distinct workload identities for different depths, shapes, backends, or
output policies. Keep all selected graph occurrences from one forward under
that workload. Reject stale workload directories and preserve generated files.

## 3. Validate generated artifacts

Use the shared validators directly. The capture hook's `run_write_gate` checks
input/partition roundtrips; require its `roundtrip: ok` stamp. For saved-graph
execution not exposed by a shared CLI, invoke
`full_graph_harness.load_full_graph` directly in a short run command to load
and execute the artifact. Do not build a model-specific validation script or
duplicate the canonical validator's input generation, execution, and reports.

Resolve manifest pattern hashes through `canonical/*/meta.json`, not directory
names. In `shapes.json`, find points attributed to the qualified
`suite/mode/workload` key. Compare pattern/shape sets and occurrence counts with
the retained capture index, including attribution added to existing patterns.
The captured index deduplicates `(pattern_hash, shape_hash)` and records
occurrences separately; candidate count is not original layer count.

Pass all affected repro paths to the shared eager validator with `--all-shapes`,
including additions merged into existing patterns. A short metadata lookup to
select paths from the manifest is sufficient; validation stays in the shared
tool. Preserve shape arguments, strides, aliasing, and integer bounds. Check
the source/replay output contract in the existing capture invocation.

```bash
<python> scripts/validate_eager.py <affected-repro-paths> \
  --all-shapes --gpus <gpu> --output <cache>/eager-validation.json

<python> scripts/validate_corpus_invariants.py --corpus-root <staging-root>
```

The eager CLI checks execution; it is not a numerical-equivalence check.
A non-default corpus root skips baseline-relative invariant counts. Avoid
overlapping memory-heavy workloads on the same GPU. If subprocess launches
exhaust thread/process limits, use CPU thread limits and the shared runner's
worker options, then rerun the failed check without changing the workload.

## 4. Integrate and revalidate

Use a supported repository promotion/capture/merge path that preserves saved
graphs and canonical merging. Replay the selected runnable again for the final
root when no other supported promotion path exists; do not reconstruct the HF
model. Derive the final affected set again and repeat saved-graph and
affected-shape execution, then run final invariants from the better-benchmark
checkout:

```bash
<python> scripts/validate_corpus_invariants.py
```

Retain logs and exit codes. Review generated-file provenance and update model
inventory/reproduction docs with the actual captured scope. Compare unrelated
full-corpus failures against the retained baseline, and report them separately
from validation of the addition. Do not describe a failing global check as
passing.

Finish with a small reproduction bundle: remove redundant wrappers and move
remaining cache material to normal cache locations. Ensure every documented
generation command has its required configs and input files. Put the compact
coverage/validation summary in the model's repository documentation; detailed
reports can remain cached.

## Completion evidence

| Result | Evidence |
| --- | --- |
| Source workload, when newly captured | Config/revision/digest, selected indices/types, production dimensions, output checks and log |
| Selected export | Trace/export paths and runnable frame/compile identities |
| Complete capture | Expected saved graphs and sidecars, successful roundtrips, nonempty capture set, zero drops |
| Staging validation | Saved-graph execution, affected pattern/shape and occurrence accounting, eager report, invariant log |
| Final validation | Final saved-graph/shape execution and default-root invariant log |
| Provenance and coverage | File producers, new/reused patterns and points, actual reduced/full depth, reproduction docs |

For existing-export imports, distinguish supplied source-execution evidence
from checks performed during import.
