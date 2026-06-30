---
name: audit-pr
description: >-
  Audit vLLM tests for fragile assertions and coincidentally-correct oracles.
  Scopes to a PR, directory, or single test, finds test files, and produces
  structured evidence. Use when auditing tests, checking a PR for brittle
  numeric assumptions, or analyzing test oracle correctness. Feed output to
  test-audit-review (Phase 2) in a separate invocation.
---

# Audit PR — Phase 1: Evidence Generation

Find and analyze vLLM test assertions that may be coincidentally correct. Produces structured evidence with initial classifications for Phase 2 adversarial verification.

**You produce structured evidence with initial classifications. Phase 2 adversarially verifies your claims — it will challenge reasoning and reclassify where warranted.**

## Numeric Stability Audit Guidance

A test is "coincidentally correct" only when **all four** are true:

1. **Weak oracle** — depends on exact generated text/token/logprob equality, match-ratio equality, or another weak generated-output oracle.
2. **Realistic breakage** — a PyTorch numeric/scheduling/compiler change has a realistic chance of changing the asserted value.
3. **No update path** — no obvious fix during a PyTorch version bump (e.g., refreshing a golden output or adjusting an intentional tolerance).
4. **No strong contract** — no vLLM/PyTorch/product contract requires the two compared executions to be bitwise/text identical.

### Excluded by Default

These are NOT coincidentally correct:

1. **Golden output tests** — maintainer can inspect and update the fixture. Clear update path (criterion 3 fails). Caveat: if the same golden is shared across TP=1/2/4, the implicit cross-TP assertion is uncontracted — note this but still classify as HAS_UPDATE_PATH.
2. **Kernel tolerance tests** (`assert_close(atol=...)`, `torch.allclose`) — tolerance IS the contract (criteria 3+4 fail).
3. **Difference-only tests** (`assert a != b`) — numeric drift unlikely to flip inequality into equality (criterion 2 fails).
4. **Smoke/liveness tests** (`assert len(output) > 0`) — FP changes don't produce empty output (criterion 2 fails).

### Strong Contracts

Treat these as strong enough to classify as STRONG_CONTRACT unless the test adds another weak oracle on top:

1. Eager vs eager with the same request sequence, same engine state, and deterministic sampling.
2. Same compile mode/artifact/config vs itself. Do NOT generalize to different compile strategies or fused distributed passes.
3. Eager vs cudagraph for the same graph/execution family.
4. CPU offload, prefetch offload, sleep/wake restoration, reload, tensorizer, and KV-transfer restoration — data movement/restoration should not change model math.
5. Streaming vs non-streaming response reconstruction — API transport contract.
6. Duplicate identical requests in the same batch with the same sampling settings.
7. Same prompt with the same explicit seed in the same engine/request setup.
8. Spec decode exact matching only when the test explicitly forces batch-invariant mode/kernels.
9. Tests under `tests/v1/determinism/` get `VLLM_BATCH_INVARIANT=1` from the autouse `conftest.py` fixture — account for that before classifying as ordinary batch-invariance assumptions.

### Not Strong By Default

These remain suspicious unless the test explicitly establishes a stronger contract:

1. Eager vs compile.
2. Non-compiled vs compiled mode parity hidden inside `compare_two_settings` or `compare_all_settings`.
3. Different compile strategies, graph partitioning strategies, or fused distributed compile passes versus baseline.
4. Tensor parallel vs pipeline parallel vs expert parallel exact generated output equality.
5. Sequence parallel, async TP, or fused distributed compile-pass parity against an unfused baseline.
6. Batch size invariance, including BS=1 vs BS=N, unless batch-invariant kernels/mode are explicitly enabled by the test.
7. Cascade attention vs non-cascade attention when the comparison also changes batch geometry.
8. Spec decoding vs base decoding exact text/token/rank matching, or exact-match ratios, when target verification changes batch geometry and the test does not force batch-invariant mode.
9. Prompt text vs prompt_embeds equality when the only oracle is final generated text.
10. Single request vs first item in a larger multimodal batch, unless the test forces batch-invariant execution.

## Modes

### PR Mode (`audit-pr 1234`)

1. Get changed files:
   ```bash
   gh pr view <number> --repo vllm-project/vllm --json files --jq '.files[].path'
   ```
2. Filter to test files using the test file filter
3. If no test files changed, report "No test files in this PR" and stop

### Directory Mode (`audit-pr tests/compile/correctness_e2e/`)

Run the test file filter on the given directory.

### File Mode (`audit-pr tests/v1/e2e/general/test_cascade_attention.py`)

Analyze the single file directly.

### File + Test Mode (`audit-pr tests/.../test_cascade_attention.py::test_cascade_attention`)

Read the named test function directly and produce evidence for it. Pytest-style `file::function` syntax.

## Step 1: Find Test Files

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/test_file_filter.sh" <files-or-directories>
```

Outputs one `test_*.py` path per line. If no files found, stop.

## Step 2: Analyze Test Files

Read each test file and identify test functions with generated-output assertions. Apply the strong contracts and not-strong-by-default guidance above to determine if each is suspicious.

**Exact equality assertions — apply all 4 criteria:**
- `compare_two_settings` / `compare_all_settings` — exact dict equality across two server configs
- `check_outputs_equal` — exact text + token ID equality between two output sequences
- `validate_generated_texts` — exact text equality, cross-runtime (vLLM vs HuggingFace)
- Direct `assert .text ==` or `assert output_ids ==` — inline exact comparison
- Batch duplication (`[prompt] * N`) followed by exact equality check

**Threshold-based assertions — check if threshold is principled:**
- `check_answers` with `accept_rate` — loose match ratio (default 70%)
- Match-ratio patterns (`matches >= int(0.6 * total)`) — spec decode acceptance thresholds
- `check_accuracy` with percentage thresholds — element-wise match ratio

**Tolerance-based assertions — generally contracted, flag only if tolerance is unreasonable:**
- `check_logprobs_close` — allows token divergence, checks top-k logprob overlap
- `check_embeddings_close` — cosine similarity with configurable tolerance
- `torch.testing.assert_close` / `torch.allclose` — numeric tolerance is the contract
- `pytest.approx` — explicit relative/absolute tolerance

Group by file to avoid re-reading.

### What to extract

For each suspicious test function, identify the comparison and oracle, then rate each of the 4 inclusion criteria:

1. **COMPARISON**: What two executions are compared? (e.g., "batch=1 vs batch=64", "eager vs compile", "vLLM vs HuggingFace")
2. **ORACLE**: Assertion type (exact text, exact token ID, exact dict, match ratio, tolerance, smoke test)
3. **BATCH_INVARIANT_ENABLED**: Check test file and conftest.py for `VLLM_BATCH_INVARIANT`
4. **CODE_PATH_VERIFIED**: Does the test assert the feature actually ran?
5. **FIXTURES**: Relevant autouse fixtures
6. **Rate each criterion** from the Numeric Stability Audit Guidance:
   - **C1 WEAK_ORACLE**: Does it use exact equality or a weak generated-output oracle?
   - **C2 REALISTIC_BREAKAGE**: Could a PyTorch numeric/scheduling/compiler change realistically change the result?
   - **C3 NO_UPDATE_PATH**: Is there no obvious fix on PyTorch bump (no golden to refresh, no tolerance to adjust)?
   - **C4 NO_STRONG_CONTRACT**: Is there no contract requiring these executions to be identical? (Check the strong contracts list above)

A test is COINCIDENTALLY_CORRECT only when all four are YES.

For a complete worked example showing how to apply the criteria, see [example.md](example.md).

## Output Format

For each suspicious test function:

```
CANDIDATE: <test function name>
  FILE: <path>
  LINE: <line number>
  COMPARISON: <what two executions>
  ORACLE: <assertion type>
  HELPER: <helper function or "direct assertion">
  BATCH_INVARIANT_ENABLED: yes/no
  CODE_PATH_VERIFIED: yes/no
  FIXTURES: <relevant fixtures>
  C1_WEAK_ORACLE: yes/no — <brief reason>
  C2_REALISTIC_BREAKAGE: yes/no — <brief reason>
  C3_NO_UPDATE_PATH: yes/no — <brief reason>
  C4_NO_STRONG_CONTRACT: yes/no — <cite clause number, e.g., "no, Strong Contract #5: streaming transport" or "yes, Not Strong #6: batch size invariance without BI mode">
  CLASSIFICATION: <COINCIDENTALLY_CORRECT / STRONG_CONTRACT / HAS_UPDATE_PATH / NOT_REALISTIC>
  CODE_SNIPPET: |
    <assertion and surrounding context>
```

End with:

```
# Evidence Summary
Test files in scope: <N>
Candidates analyzed: <N>
```

**Include your classification for each candidate. Phase 2 will adversarially verify.**

## Two-Phase Design

Phase 2 (`test-audit-review`) must run in a **separate Claude invocation** to prevent bias propagation. This skill outputs structured evidence with initial classifications. Phase 2 independently verifies each claim.
