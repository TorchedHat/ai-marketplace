---
name: test-audit-review
description: >-
  Phase 2 of the two-phase test oracle auditor. Classify vLLM test candidates as
  COINCIDENTALLY_CORRECT, STRONG_CONTRACT, or HAS_UPDATE_PATH. Takes structured
  evidence from audit-pr (Phase 1) and applies the decision tree independently.
  Must run in a separate Claude invocation from Phase 1 to prevent bias propagation.
---

# Test Audit: Review (Phase 2)

Classify candidate tests through a formal decision tree. Takes structured evidence from `audit-pr` (Phase 1) or raw findings from `test-audit-explore` and produces a reviewed, classified list.

**You are Phase 2 of a two-phase pipeline. You receive structured evidence (facts about test assertions) and independently classify each candidate. You have no knowledge of Phase 1's reasoning — only its structured output. Your job is adversarial: find reasons to REMOVE candidates from the suspicious list (strong contract? update path? unrealistic drift?).**

## When to use

- After `audit-pr` has produced structured evidence (PR-scoped, pre-filtered)
- When triaging a specific test that broke on a PyTorch upgrade
- When deciding whether a test needs fixing or is actually correct
- When reviewing a PR that adds a new cross-config comparison test

## Review Workflow

The review process moves candidates through a decision tree. Keep a broad raw candidate inventory — false positives are acceptable in the raw file.

For each candidate:

1. **Identify what two executions or outputs are being compared.**
2. **Ask whether PyTorch/vLLM/product behavior requires those executions to be bitwise/text identical.** If yes → move to the reviewed/strong bucket with the contract named explicitly.
3. **If no strong contract, ask whether a maintainer has an obvious update path on PyTorch bump** (refresh a golden, adjust a tolerance, tune a config).
4. **Only keep it in the coincidentally correct list** when the answer is no and numeric drift has a realistic chance of changing the test outcome.

```
candidate test
    │
    ├─ strong contract? ──yes──► STRONG_CONTRACT (remove from list)
    │                              name the contract explicitly
    │
    ├─ update path? ──────yes──► HAS_UPDATE_PATH (remove from list)
    │                              name the path (refresh golden, adjust tolerance)
    │
    ├─ realistic drift? ──yes──► COINCIDENTALLY_CORRECT (keep on list)
    │                              needs fixing
    │
    └─ no ────────────────────► NOT_REALISTIC (remove from list)
```

## Inclusion Criteria

A test is coincidentally correct only when **all four** are true:

1. **Weak oracle** — depends on exact generated text/token/logprob equality, match-ratio equality, or another weak generated-output oracle
2. **Realistic breakage** — a PyTorch numeric/scheduling/compiler change has a realistic chance of changing the asserted value
3. **No update path** — no obvious fix during a PyTorch version bump (refreshing a golden, adjusting a tolerance)
4. **No strong contract** — no vLLM/PyTorch/product contract requires the two compared executions to be bitwise/text identical

## Strong Contracts

These comparisons have genuine guarantees. Classify as `STRONG_CONTRACT` with the contract named:

| Contract | Why it holds |
|----------|-------------|
| Eager vs eager, same request sequence, same engine state, deterministic sampling | Same CUDA kernels in same order |
| Same compile mode/artifact/config vs itself | Same optimization = same execution. Does NOT extend to different strategies |
| Eager vs cudagraph for the SAME graph/execution family | Cudagraph replays exact captured kernel sequence |
| CPU offload / prefetch offload / sleep-wake / reload / tensorizer / KV-transfer restoration | Data movement must not change model math |
| Streaming vs non-streaming response reconstruction | API transport contract |
| Duplicate identical requests in the same batch, same sampling | Same engine state, same padding, same kernels |
| Same prompt + same explicit seed + same engine setup | Seeded determinism is an explicit vLLM contract |
| Spec decode with explicitly forced batch-invariant mode/kernels | BI mode provides the contract |
| `tests/v1/determinism/` | Autouse conftest sets `VLLM_BATCH_INVARIANT=1` |

## Not Strong By Default

These remain suspicious. Classify as `COINCIDENTALLY_CORRECT` unless the test explicitly establishes a stronger contract:

| Comparison | Why no contract |
|------------|----------------|
| Eager vs compile | Inductor fuses kernels, reorders operations, changes reduction patterns |
| Non-compiled vs compiled hidden inside `compare_two_settings` | Quietly asserts cross-mode parity via `assert ==` |
| Different compile strategies / graph partitioning / fused passes vs baseline | Async TP, SP, fused all-reduce+norm — different math orderings |
| TP vs PP vs EP exact equality | Each changes which GPU computes what, changing FP accumulation |
| Batch size invariance (BS=1 vs BS=N) without `VLLM_BATCH_INVARIANT` | Different batch sizes change cuBLAS kernel selection |
| Cascade attention with changed batch geometry | Changes batch size AND attention algorithm simultaneously |
| Spec decode vs base exact matching without BI mode | Target verification runs at different seqlen = different kernel |
| Prompt text vs prompt_embeds via generated text | Entirely different numerical paths |
| Single vs batched multimodal without BI mode | Different batch = different padding = different kernels |

## Excluded by Default

These are NOT coincidentally correct — classify as `HAS_UPDATE_PATH` or `NOT_REALISTIC`:

- **Golden output tests** (`== EXPECTED_OUTPUT` constant) — maintainer refreshes the golden. Update path exists (criterion 3 fails). **Caveat**: if the same golden is shared across TP=1/2/4, the implicit cross-TP assertion is uncontracted — note this but still classify as HAS_UPDATE_PATH.
- **Kernel tolerance tests** (`assert_close(atol=1e-2)`) — tolerance IS the contract (criteria 3+4 fail).
- **Difference-only tests** (`assert a != b`) — drift makes values closer, not equal (criterion 2 fails).
- **Smoke/liveness tests** (`assert len(output) > 0`) — FP changes don't produce empty output (criterion 2 fails).

## Output Format

For each test, output one line:

```
test_name → CLASSIFICATION | contract/path/reason: <explanation>
```

### Examples

```
test_cascade_attention → COINCIDENTALLY_CORRECT | reason: batch=1 vs batch=64 without VLLM_BATCH_INVARIANT; cascade vs non-cascade kernel paths
test_cpu_offload → STRONG_CONTRACT | contract: data restoration must not change model math
test_chatglm3_lora_tp4 → HAS_UPDATE_PATH | path: refresh EXPECTED_LORA_OUTPUT golden constant
test_streaming_input → STRONG_CONTRACT | contract: API transport — same engine/prompt/sampling, only input timing differs
test_async_tp_pass_correctness → COINCIDENTALLY_CORRECT | reason: eager vs compile (fused gemm comms); not a strong contract
test_mtp_correctness → COINCIDENTALLY_CORRECT | reason: spec decode without BI mode; 80% threshold has no principled update path
```

## Summary Table

After reviewing all candidates, produce a summary:

| Classification | Count | Action |
|---|---|---|
| COINCIDENTALLY_CORRECT | N | Needs fixing — add BI mode, tolerance, or golden strings |
| STRONG_CONTRACT | N | Remove from list — exact match is correct by design |
| HAS_UPDATE_PATH | N | Remove from list — maintainer can refresh on bump |
| NOT_REALISTIC | N | Remove from list — drift won't change outcome |
