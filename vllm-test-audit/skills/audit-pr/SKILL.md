---
name: audit-pr
description: >-
  Audit vLLM tests for fragile assertions and coincidentally-correct oracles.
  Scopes to a PR, directory, or single test, runs the candidate finder, and
  produces structured evidence. Use when auditing tests, checking a PR for
  brittle numeric assumptions, or analyzing test oracle correctness. Feed
  output to test-audit-review (Phase 2) in a separate invocation.
---

# Audit PR — Phase 1: Evidence Generation

Find and analyze vLLM test assertions that may be coincidentally correct. Produces structured evidence for Phase 2 classification.

**You produce evidence, not conclusions. Do not classify tests as COINCIDENTALLY_CORRECT, STRONG_CONTRACT, or HAS_UPDATE_PATH. That is Phase 2's job.**

## Modes

### PR Mode (`audit-pr 1234`)

1. Get changed files:
   ```bash
   gh pr view <number> --repo vllm-project/vllm --json files --jq '.files[].path'
   ```
2. Filter to test files (`test_*.py` only)
3. If no test files changed, report "No test files in this PR" and stop

### Directory Mode (`audit-pr tests/compile/correctness_e2e/`)

Skip the PR diff step. Run candidate finder on the given directory.

### File Mode (`audit-pr tests/v1/e2e/general/test_cascade_attention.py`)

Run candidate finder on the single file.

### File + Test Mode (`audit-pr tests/v1/e2e/general/test_cascade_attention.py::test_cascade_attention`)

Skip candidate finding entirely. Read the named test function directly and produce evidence for it. Use pytest-style `file::function` syntax.

## Step 1: Candidate Finding

Run the deterministic pre-filter:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/candidate_finder.sh" <files-or-directories>
```

Outputs `FILE:LINE:PATTERN_NAME` triples. If no results, report "No candidate assertions found" and stop.

## Step 2: Analyze Candidates

For each candidate, read the test function and extract evidence. Group by file to avoid re-reading.

### Helper Function Shortcuts

When a candidate uses one of these helpers, you already know the assertion logic — do not re-read the helper source code:

**`compare_two_settings` / `compare_all_settings`** (`tests/utils.py:1165`/`1205`):
Launches two vLLM API servers with different CLI args, generates outputs from both, asserts `ref_result == compare_result` (exact dict equality on the full API response). For `method="encode"`, uses cosine similarity >= 0.999. Never sets `VLLM_BATCH_INVARIANT`. When `force_v1_runner=True`, sets `VLLM_USE_V2_MODEL_RUNNER=0` on both sides.

**`check_outputs_equal`** (`tests/models/utils.py:25`):
Asserts `output_str_0 == output_str_1` and `output_ids_0 == output_ids_1`. Zero-tolerance string + token ID comparison.

**`validate_generated_texts`** (`tests/models/quantization/test_bitsandbytes.py:260`):
Runs same prompts through vLLM (bitsandbytes) and HuggingFace, asserts `hf_str == vllm_str`. Cross-runtime exact text equality. The correct pattern exists in the same file: `test_4bit_bnb_moe_model` uses `check_logprobs_close`.

### What to extract for each candidate

1. **COMPARISON**: What two executions are compared? Name concretely (e.g., "batch=1 vs batch=64", "eager vs compile (async TP)", "vLLM vs HuggingFace")
2. **ORACLE**: What assertion type? (exact text, exact token ID, exact dict, match ratio, tolerance, cosine similarity, smoke test)
3. **BATCH_INVARIANT_ENABLED**: Check the test file and its conftest.py for `VLLM_BATCH_INVARIANT` or `batch_invariant`
4. **CODE_PATH_VERIFIED**: Does the test assert the feature actually ran? (compilation_counter, graph break checks, backend assertions)
5. **FIXTURES**: Active autouse fixtures in the file or directory conftest.py
6. **STRONG_CONTRACT_APPLICABLE**: Note if one applies (see table below), but do not classify

### Strong Contracts

Note these in evidence when they apply — but do not classify:

| Contract | Why it holds |
|----------|-------------|
| Eager vs eager, same request/engine/sampling | Same CUDA kernels in same order |
| Same compile mode/config vs itself | Same optimization = same execution |
| Eager vs cudagraph, same execution family | Cudagraph replays exact kernel sequence |
| CPU offload / sleep-wake / reload / tensorizer / KV-transfer | Data movement must not change math |
| Streaming vs non-streaming reconstruction | API transport contract |
| Duplicate identical requests, same batch/sampling | Same engine state, padding, kernels |
| Same prompt + same seed + same engine | Seeded determinism is explicit vLLM contract |
| Spec decode with forced batch-invariant mode | BI mode provides the contract |
| `tests/v1/determinism/` | Autouse conftest sets `VLLM_BATCH_INVARIANT=1` |

## Output Format

```
CANDIDATE: <test function name>
  FILE: <relative path>
  LINE: <line number>
  COMPARISON: <what two executions>
  ORACLE: <assertion type>
  HELPER: <helper function or "direct assertion">
  BATCH_INVARIANT_ENABLED: yes/no
  CODE_PATH_VERIFIED: yes/no
  FIXTURES: <relevant fixtures>
  STRONG_CONTRACT_APPLICABLE: <contract name or "none">
  CODE_SNIPPET: |
    <assertion and 5-10 lines of context>
```

After all blocks:

```
# Evidence Summary
Candidates found by script: <N>
Candidates analyzed: <N>
Files read: <N>
```

**Do not add classifications, risk ratings, recommendations, or fix suggestions.**
