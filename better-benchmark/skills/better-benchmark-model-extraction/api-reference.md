# Better-benchmark API reference

Paths are relative to better-benchmark. This is an entry-point map; procedures
and required flags are in the [artifact workflow](artifact-workflow.md).
Verify local signatures before invocation.

Use the configured Python interpreter and installed modules. The package may
install flat modules such as `capture_hook` without a `better_benchmark`
namespace. Locate checkout-only scripts through an installed module's
`__file__`; verify transitive helper imports before starting GPU work. Use a
minimal source-path fallback only when a needed helper is absent from the
installation, rather than reinstalling or changing packaging for extraction.

| File | Entry points and responsibility |
| --- | --- |
| `scripts/run_recapture.py` | `capture_single_model`, `_load_via_upstream_runner`: existing suite lifecycle and dtype policy. |
| `scripts/ingest_tlparse.py` | `find_fx_graph_runnables` and `--list-only`: runnable discovery. |
| `capture_hook.py` | `install_capture_hook`, `uninstall_capture_hook`: post-grad capture, serialization, and finalization. |
| `merge_captures.py` | `temporary_capture_for_merge(corpus_root, workload, suite=..., mode=...)` yields a context with `capture_dir` and `merge()`. The root contains both `models/` and `canonical/`. |
| `full_graph_harness.py` | `load_full_graph(path, default_device="cuda")` returns `(module, inputs, definition)`; `load_full_graph_sidecar(path)` reads sidecar metadata. |
| `roundtrip_validation.py` | `run_write_gate(path, original_gm=gm)`: capture-time input/partition roundtrip validation and sidecar stamping. |
| `repro_harness.py` | `load_shape_configs(str(repro_path))`, `make_inputs_from_config(config)`, `make_inputs_safely(module.make_inputs)`: saved shape/input consumers. |
| `input_codec.py` | Structured input encodings, symbol bindings, and alias metadata used by existing loaders. |
| `scripts/validate_eager.py` | Invoke directly with affected repro paths and `--all-shapes` for subprocess-isolated execution. Do not duplicate it in a model-specific validator. |
| `scripts/validate_corpus_invariants.py` | Staging and final corpus consistency checks. |
| `scripts/recapture_full_graphs.py` | Repartition already-saved graphs when separately requested. |
| `extract_vllm.py`, `extract_reductions.py` | Legacy construction/extraction examples. `extract_vllm.py` constructs HF models despite its name; its `max_layers` only limits `num_hidden_layers`, without arbitrary layer selection or guaranteed compatibility with the current corpus lifecycle. |
