---
name: test-audit-explore
description: Explore the vLLM test suite to find tests that are "coincidentally correct" — tests that assume batch invariance without enforcing it and could break when PyTorch changes numerics. Use when searching for fragile tests, auditing test correctness assumptions, or investigating vLLM-PyTorch dependency surface. Fan out subagents across directory slices for coverage.
---

# Test Audit: Explore

Find vLLM tests that are "coincidentally correct" — they pass today by luck of current numerics, not because the thing they test guarantees the assertion.

## When to use

- Auditing the vLLM test suite for fragile tests before a PyTorch release
- Searching a specific directory for batch invariance assumptions
- Expanding the known list of coincidentally-correct tests
- Investigating CI failures that may be caused by PyTorch numeric changes

## Numeric Stability Audit Guidance

This section captures the working criterion for reviewing vLLM tests whose oracle depends on generated output equality, token equality, logprob equality, or mode parity.

### When a test is "coincidentally correct"

A test belongs on the suspicious list only when **all four** are true:

1. It depends on exact generated text/token/logprob equality, match-ratio equality, or another weak generated-output oracle.
2. A PyTorch numeric/scheduling/compiler change has a realistic chance of changing the asserted value.
3. There is no obvious update path during a PyTorch version bump, such as refreshing a golden output or adjusting an intentional tolerance.
4. There is no strong vLLM/PyTorch/product contract that requires the two compared executions to be bitwise/text identical.

### What is NOT coincidentally correct

**Golden output tests** are not coincidentally correct by default. If the model output changes after a PyTorch bump and the test has a clear expected string/token fixture, the maintainer can inspect and update the golden.

**Kernel tolerance tests** are not coincidentally correct. Their tolerance is the contract, and PyTorch changes should generally remain within it; if not, the tolerance/reference is the obvious update point.

**Difference-only tests** are usually not coincidentally correct. A slight numeric change is unlikely to make an intended inequality collapse into equality.

**Smoke/liveness tests** such as non-empty output checks are usually not coincidentally correct because slight numeric drift is unlikely to change the pass/fail result.

### Strong Contracts

Treat these as strong enough to remove from the suspicious list unless the test adds another weak oracle on top:

- Eager vs eager with the same request sequence, same engine state, and deterministic sampling.
- Same compile mode/artifact/config vs itself. Do not generalize this to different compile strategies or fused distributed passes.
- Eager vs cudagraph for the same graph/execution family.
- CPU offload, prefetch offload, sleep/wake restoration, reload, tensorizer, and KV-transfer restoration, because data movement/restoration should not change model math.
- Streaming vs non-streaming response reconstruction, because that is an API transport contract.
- Duplicate identical requests in the same batch with the same sampling settings.
- Same prompt with the same explicit seed in the same engine/request setup.
- Spec decode exact matching only when the test explicitly forces the needed batch-invariant mode/kernels.
- Tests under `tests/v1/determinism/` currently get `VLLM_BATCH_INVARIANT=1` from that directory's autouse `conftest.py` fixture; account for that before classifying them as ordinary batch-invariance assumptions.

### Not Strong By Default

These remain suspicious unless the test explicitly establishes a stronger contract:

- Eager vs compile.
- Non-compiled vs compiled mode parity hidden inside `compare_two_settings` or `compare_all_settings`.
- Different compile strategies, graph partitioning strategies, or fused distributed compile passes versus baseline.
- Tensor parallel vs pipeline parallel vs expert parallel exact generated output equality.
- Sequence parallel, async TP, or fused distributed compile-pass parity against an unfused baseline.
- Batch size invariance, including BS=1 vs BS=N, unless batch-invariant kernels/mode are explicitly enabled by the test.
- Cascade attention vs non-cascade attention when the comparison also changes batch geometry.
- Spec decoding vs base decoding exact text/token/rank matching, or exact-match ratios, when target verification changes batch geometry and the test does not force batch-invariant mode.
- Prompt text vs prompt_embeds equality when the only oracle is final generated text.
- Single request vs first item in a larger multimodal batch, unless the test forces batch-invariant execution.

## The Base Example: What to Look For

The canonical bad test is `tests/v1/e2e/general/test_cascade_attention.py`:

```python
single_prompt = [example_system_message + prompt]
responses = llm.generate(single_prompt, sampling_params)  # temp=0.0
ref_output = responses[0].outputs[0].text

# (Probably) Use cascade attention.
prompts = [example_system_message + prompt] * 64
responses = llm.generate(prompts, sampling_params)
for response in responses:
    assert response.outputs[0].text == ref_output  # EXACT string match
```

Three problems make this coincidentally correct:

1. **Assumes batch invariance without enabling it** — compares batch=1 vs batch=64 via exact string equality without `VLLM_BATCH_INVARIANT=1`
2. **Doesn't verify the intended code path ran** — comment says "(Probably) Use cascade attention" with no assertion
3. **No independent oracle** — reference is the same model at batch=1

## Detection Signals

Search for these patterns in test files:

### Code patterns
- `* 64`, `* 32`, `* 16`, `* 100` — duplicating prompts into batches
- `== ref_output`, `== ref_text`, `== expected_output` — exact string comparison of generated text
- `temperature=0.0` combined with cross-batch or cross-config comparison
- `matches >= int(0.6` or similar — heuristic match thresholds instead of 100%
- `compare_two_settings` or `compare_all_settings` — utility that hides exact `assert ==` inside
- `check_outputs_equal` — zero-tolerance output comparison

### Comment patterns
- "probably", "should", "expected to", "likely" about whether a feature is active
- Comments acknowledging non-determinism but proceeding with exact match anyway

### Missing patterns (absence signals)
- No `VLLM_BATCH_INVARIANT` in tests comparing across batch configurations
- No `compilation_counter` in tests comparing compiled vs eager output
- No assertion/log check that the feature under test actually ran

## Categories of Suspicious Tests

| Category | What to look for |
|----------|-----------------|
| **Batch size invariance** | BS=1 vs BS=N output comparison without `VLLM_BATCH_INVARIANT=1` |
| **Cross-config parity** | TP vs PP vs EP exact output equality via `compare_two_settings` |
| **Eager vs compile** | Non-compiled vs compiled output exact match |
| **Fused passes vs baseline** | Async TP, sequence parallel, fused all-reduce+norm vs unfused |
| **Spec decode without BI** | Speculative vs base decoding exact text match or loose thresholds |
| **Cross-representation** | Prompt_embeds vs text-tokenized input compared via generated text |
| **Single vs batched multimodal** | One multimodal request alone vs in a batch |
| **Cross-runtime** | vLLM vs HuggingFace exact output equality |

## Directories to Search

Recommended fan-out for full coverage:

| Agent | Directories | Focus |
|-------|------------|-------|
| 1 | `tests/v1/e2e/` | E2E: cascade, spec decode, streaming, scheduling |
| 2 | `tests/compile/` | Compile correctness: eager vs compiled, async TP, SP |
| 3 | `tests/distributed/` + `tests/lora/` + `tests/models/` | Distributed parity, LoRA, multimodal batch |
| 4 | `tests/entrypoints/` + `tests/basic_correctness/` | API-level: prompt_embeds vs text, basic correctness |

**Exclude** `tests/v1/determinism/` — those have `VLLM_BATCH_INVARIANT=1` via autouse conftest.

## Output Format

For each suspicious test found, report:

```
FILE: <path>
TEST: <function name>
COMPARES: <what two executions> (e.g., "batch=1 vs batch=64")
METHOD: <exact string / token ratio / tolerance / semantic>
BATCH_INVARIANT_ENABLED: yes/no
CODE_PATH_VERIFIED: yes/no
CATEGORY: <from table above>
NOTES: <one-line explanation>
```

## Systemic Amplifiers

Watch for these utilities that propagate the pattern:

- **`compare_two_settings`** (`tests/utils.py`) — launches two servers, compares via `assert ref_result == compare_result`. Used by ~15 tests. Never sets `VLLM_BATCH_INVARIANT`.
- **`check_outputs_equal`** (`tests/utils.py`) — zero-tolerance string+token comparison.
- **`validate_generated_texts`** (`tests/models/quantization/test_bitsandbytes.py`) — does `assert hf_str == vllm_str`.

## Reference

- Batch invariance docs: https://docs.vllm.ai/en/latest/features/batch_invariance/
- Original audit gist: https://gist.github.com/zou3519/ded34ec406db9ba71cf622652664f103
- `VLLM_BATCH_INVARIANT` implementation: `vllm/model_executor/layers/batch_invariant.py`
- Cascade attention heuristic: `vllm/v1/attention/backends/flash_attn.py:use_cascade_attention()`
