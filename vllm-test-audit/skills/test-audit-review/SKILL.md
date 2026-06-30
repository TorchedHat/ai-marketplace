---
name: test-audit-review
description: >-
  Phase 2 of the two-phase test oracle auditor. Adversarially verify Phase 1
  classifications of vLLM test candidates. Challenge each criterion rating,
  reclassify where warranted. Use after audit-pr has produced structured
  evidence with initial classifications.
---

# Test Audit: Review (Phase 2) — Adversarial Verification

You receive structured evidence with initial classifications from Phase 1 (`audit-pr`). Your job is to **adversarially verify** each claim — challenge the reasoning, check the criterion ratings, and reclassify where warranted.

**You are the skeptic. For each candidate, try to find reasons Phase 1 got it wrong. Look for strong contracts Phase 1 missed, update paths it overlooked, or unrealistic breakage it assumed.**

## When to use

- After `audit-pr` has produced structured evidence with classifications
- When triaging a specific test that broke on a PyTorch upgrade
- When deciding whether a test needs fixing or is actually correct
- When reviewing a PR that adds a new cross-config comparison test

## How to Verify

For each candidate from Phase 1, check the four criterion ratings:

### C1 WEAK_ORACLE — Is the oracle actually weak?

Phase 1 may have flagged an assertion as weak when it's actually appropriate:
- Did Phase 1 miss that the test uses `check_logprobs_close` (tolerance-based, not exact)?
- Is the "exact equality" actually comparing within the same execution path (not cross-config)?
- Is there a tolerance or threshold that Phase 1 overlooked?

### C2 REALISTIC_BREAKAGE — Would PyTorch changes actually break this?

Phase 1 may overestimate breakage risk:
- Are both sides using the same CUDA kernels in the same order?
- Is the comparison within a single engine invocation (no kernel selection variance)?
- Would the test need a truly exotic PyTorch change to break?

### C3 NO_UPDATE_PATH — Is there really no update path?

Phase 1 may miss obvious fixes:
- Is there a golden constant that could be refreshed?
- Could a tolerance be added or adjusted?
- Is the reference output stored in a fixture or conftest that a maintainer could update?

### C4 NO_STRONG_CONTRACT — Did Phase 1 miss a contract?

This is the most common Phase 1 error. Check against the numbered clauses below:

#### Strong Contracts

1. Eager vs eager with the same request sequence, same engine state, and deterministic sampling.
2. Same compile mode/artifact/config vs itself. Does NOT extend to different compile strategies or fused distributed passes.
3. Eager vs cudagraph for the same graph/execution family.
4. CPU offload, prefetch offload, sleep/wake restoration, reload, tensorizer, and KV-transfer restoration — data movement/restoration should not change model math.
5. Streaming vs non-streaming response reconstruction — API transport contract.
6. Duplicate identical requests in the same batch with the same sampling settings.
7. Same prompt with the same explicit seed in the same engine/request setup.
8. Spec decode exact matching only when the test explicitly forces batch-invariant mode/kernels.
9. Tests under `tests/v1/determinism/` get `VLLM_BATCH_INVARIANT=1` from the autouse `conftest.py` fixture — account for that before classifying as ordinary batch-invariance assumptions.

#### Not Strong By Default

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

#### Excluded by Default

1. **Golden output tests** — maintainer can refresh the fixture. Classify as HAS_UPDATE_PATH. Caveat: if the same golden is shared across TP=1/2/4, the implicit cross-TP assertion is uncontracted — note this.
2. **Kernel tolerance tests** (`assert_close(atol=...)`, `torch.allclose`) — tolerance IS the contract. Classify as NOT_REALISTIC.
3. **Difference-only tests** (`assert a != b`) — drift unlikely to flip. Classify as NOT_REALISTIC.
4. **Smoke/liveness tests** (`assert len(output) > 0`) — FP changes don't produce empty output. Classify as NOT_REALISTIC.

## Verification Decision

For each candidate, produce a verdict:

- **AGREE** — Phase 1 classification is correct, all criterion ratings hold
- **RECLASSIFY** — one or more criterion ratings are wrong, provide corrected classification
- **NEEDS_CONTEXT** — cannot verify without reading additional code (name what's needed)

## Output Format

For each candidate:

```
CANDIDATE: <test function name>
  PHASE_1_CLASSIFICATION: <what Phase 1 said>
  VERDICT: AGREE / RECLASSIFY / NEEDS_CONTEXT
  C1_WEAK_ORACLE: <agree/disagree> — <reason if disagree>
  C2_REALISTIC_BREAKAGE: <agree/disagree> — <reason if disagree>
  C3_NO_UPDATE_PATH: <agree/disagree> — <reason if disagree>
  C4_NO_STRONG_CONTRACT: <agree/disagree> — <cite clause if Phase 1 missed one>
  FINAL_CLASSIFICATION: <COINCIDENTALLY_CORRECT / STRONG_CONTRACT / HAS_UPDATE_PATH / NOT_REALISTIC>
```

After all candidates:

## Summary Table

| Classification | Count | Action |
|---|---|---|
| COINCIDENTALLY_CORRECT | N | Needs fixing — add BI mode, tolerance, or golden strings |
| STRONG_CONTRACT | N | Remove from list — exact match is correct by design |
| HAS_UPDATE_PATH | N | Remove from list — maintainer can refresh on bump |
| NOT_REALISTIC | N | Remove from list — drift won't change outcome |

Also report:
- How many Phase 1 classifications you agreed with
- How many you reclassified (and the pattern — e.g., "3 reclassified from CC to STRONG_CONTRACT due to missed streaming contract")
