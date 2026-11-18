---
name: better-benchmark-model-extraction
description: Extract inference fusion patterns from PyTorch models into better-benchmark using direct source compilation, capture, merge, and saved-artifact validation.
---

# Extract model patterns

Use the local model source and better-benchmark's existing capture/merge APIs.
For a planning request, inspect and propose the workload, then stop.

1. Read checkout instructions. Check git state, API signatures, imports,
   runtime/source versions, and GPU capacity. Use the configured Python and
   intended local source paths. Run corpus invariants before changing the
   corpus; retain the baseline in a standard cache directory.

2. Construct the agreed workload from its pinned config with random weights
   and legal synthetic inputs. If representative depth is sufficient, select
   layers covering the distinct block types and reduce depth before allocation.
   Preserve all other geometry, coupled config fields, weight tying, and
   explicit internal precision. Record original layer indices, dtypes, seed,
   input/output shapes, cache policy, and backend. Preserve native dispatch
   unless another backend was requested.

3. Install capture around the workload's first compiled invocation and merge
   directly into `repros`. Preserve its construction, output checks, and
   provenance reporting. Example below assumes an eval-mode `model`, positional
   `inputs`, agreed `name`/`suite`/`fullgraph`, a cache `Path`, and a
   workload-specific `check_output` asserting shapes, dtypes, and finiteness:

   ```python
   import json
   from pathlib import Path
   import torch
   from capture_hook import install_capture_hook, uninstall_capture_hook
   from merge_captures import temporary_capture_for_merge

   root = Path("repros")
   with temporary_capture_for_merge(root, name, suite=suite, mode="infer") as capture:
       install_capture_hook(
           str(capture.capture_dir), label=name,
           graph_dir=str(root / "models" / suite / "infer" / name),
           capture_only=False, constant_folding=False,
       )
       try:
           with torch.no_grad():
               output = torch.compile(model, fullgraph=fullgraph, dynamic=False)(*inputs)
               torch.cuda.synchronize()
           check_output(output)
       finally:
           uninstall_capture_hook()
       index = json.loads((capture.capture_dir / "index.json").read_text())
       (cache / "capture-index.json").write_text(json.dumps(index, indent=2))
       assert not index["dropped"], index["dropped"]
       capture.merge()
   ```

   Set `fullgraph` from the workload contract. Keep constant folding disabled
   unless the intended deployment uses runtime constant folding. The temporary
   raw capture is removed on context exit; saved graphs go directly to the
   model directory. Trace export, tlparse replay, and staging are unnecessary
   for this source-capture path.

4. Validate the written artifacts with shared loaders and validators:
   - Zero dropped regions and `roundtrip: ok` on every saved graph sidecar.
   - Execute saved graphs with `full_graph_harness.load_full_graph`; check
     output shape, dtype, and finiteness.
   - Resolve pattern hashes through canonical `meta.json`. Compare exact
     pattern/shape/occurrence accounting across the capture index, merged
     `shapes.json`, and live FX using shared partition/accounting helpers.
   - Run `scripts/validate_eager.py <affected-paths> --all-shapes`, including
     older points of reused patterns, then `scripts/validate_corpus_invariants.py`.
     Compare failures with the baseline. Check for stale cache-only directories
     before treating missing-repro directories as lost source files.

5. Review the corpus diff, including metadata/shapes of reused patterns, and
   confirm older points and attribution are preserved. Report captured scope,
   new/reused patterns, occurrences, resolved attention/expert operators, and
   validation results. Keep configs, provenance, commands, and reports in the
   requested scratch location or standard cache. Do not add a model README.

Use structured data, shared loaders, and live FX; do not hand-edit generated
artifacts or regex/AST-parse their renderings. Do not rebuild PyTorch, run
numerical comparisons, or benchmark unless separately requested.
