# Worked Example: Cascade Attention

This example walks through a complete Phase 1 analysis of `test_cascade_attention` — the canonical coincidentally-correct test.

## The Test

From `tests/v1/e2e/general/test_cascade_attention.py`:

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

## Analysis

**What two executions?** batch=1 (single prompt) vs batch=64 (64 duplicated prompts). The test generates a reference with one prompt, then checks that all 64 copies in a larger batch produce identical text.

**What oracle?** Direct `assert .text ==` — exact string equality on generated text.

**VLLM_BATCH_INVARIANT?** No. Not set in the test, not in any fixture, not in conftest. The test relies on batch-invariant behavior without requesting it.

**Code path verified?** No. The comment says "(Probably) Use cascade attention" — no assertion that cascade attention actually ran.

**Fixtures?** `@create_new_process_for_each_test()` decorator — isolates process state but doesn't set batch invariant mode.

**Rating the 4 criteria:**

- **C1 WEAK_ORACLE**: Yes — exact string `==` on generated text is a weak oracle.
- **C2 REALISTIC_BREAKAGE**: Yes — PyTorch issue 182700 showed cuBLAS kernel selection changes with batch size, changing FP accumulation order.
- **C3 NO_UPDATE_PATH**: Yes — the reference output is self-generated at batch=1, not a refreshable golden constant.
- **C4 NO_STRONG_CONTRACT**: Yes — Not Strong #6: batch size invariance without BI mode. Different batch sizes change cuBLAS kernel selection, tiling, and accumulation.

All four YES → COINCIDENTALLY_CORRECT.

## Output

```
CANDIDATE: test_cascade_attention
  FILE: tests/v1/e2e/general/test_cascade_attention.py
  LINE: 43
  COMPARISON: batch=1 vs batch=64
  ORACLE: exact text equality (assert .text ==)
  HELPER: direct assertion
  BATCH_INVARIANT_ENABLED: no
  CODE_PATH_VERIFIED: no
  FIXTURES: @create_new_process_for_each_test()
  C1_WEAK_ORACLE: yes — exact string == on generated text
  C2_REALISTIC_BREAKAGE: yes — PyTorch #182700, cuBLAS kernel selection changes with batch size
  C3_NO_UPDATE_PATH: yes — reference is self-generated at batch=1, not a refreshable golden
  C4_NO_STRONG_CONTRACT: yes — Not Strong #6: batch size invariance without BI mode
  CLASSIFICATION: COINCIDENTALLY_CORRECT
  CODE_SNIPPET: |
    single_prompt = [example_system_message + prompt]
    responses = llm.generate(single_prompt, sampling_params)
    ref_output = responses[0].outputs[0].text
    # (Probably) Use cascade attention.
    prompts = [example_system_message + prompt] * 64
    responses = llm.generate(prompts, sampling_params)
    for response in responses:
        assert response.outputs[0].text == ref_output
```

## Contrast: A STRONG_CONTRACT Example

For comparison, `test_cpu_offload` from `tests/basic_correctness/test_cpu_offload.py` uses `compare_two_settings` but with a strong contract:

```
CANDIDATE: test_cpu_offload
  FILE: tests/basic_correctness/test_cpu_offload.py
  LINE: 42
  COMPARISON: normal loading vs CPU offload loading
  ORACLE: exact dict equality (compare_two_settings)
  HELPER: compare_two_settings
  BATCH_INVARIANT_ENABLED: no
  CODE_PATH_VERIFIED: no
  FIXTURES: none relevant
  C1_WEAK_ORACLE: yes — exact dict equality via compare_two_settings
  C2_REALISTIC_BREAKAGE: no — data movement/restoration uses identical kernels
  C3_NO_UPDATE_PATH: n/a (criterion 2 already fails)
  C4_NO_STRONG_CONTRACT: no — Strong Contract #4: CPU offload restoration must not change model math
  CLASSIFICATION: STRONG_CONTRACT
  CODE_SNIPPET: |
    compare_two_settings(model, base_args, offload_args)
```

C2 is NO and C4 cites Strong Contract #4, so this is not coincidentally correct despite using exact equality.
