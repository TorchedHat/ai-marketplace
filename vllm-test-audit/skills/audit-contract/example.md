# Worked Examples

Two examples showing the full Phase 1 → Phase 2 pipeline: one COINCIDENTALLY_CORRECT, one STRONG_CONTRACT.

## Example 1: test_cascade_attention (COINCIDENTALLY_CORRECT)

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

**Why it's coincidentally correct:** Compares batch=1 vs batch=64 via exact string equality without `VLLM_BATCH_INVARIANT`. No strong contract (Not Strong #6). Reference is self-generated, not a refreshable golden. PyTorch issue 182700 showed this breaks.

### Phase 1 output

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
  VERDICT: COINCIDENTALLY_CORRECT
  CODE_SNIPPET: |
    prompts = [example_system_message + prompt] * 64
    responses = llm.generate(prompts, sampling_params)
    for response in responses:
        assert response.outputs[0].text == ref_output
```

### Phase 2 output (AGREE)

```
CANDIDATE: test_cascade_attention
  PHASE_1_CLASSIFICATION: COINCIDENTALLY_CORRECT
  PHASE_1_VERDICT: COINCIDENTALLY_CORRECT
  REVIEW: AGREE
  FILE: tests/v1/e2e/general/test_cascade_attention.py
  LINE: 43
  COMPARISON: batch=1 vs batch=64
  ORACLE: exact text equality (assert .text ==)
  HELPER: direct assertion
  BATCH_INVARIANT_ENABLED: no
  CODE_PATH_VERIFIED: no
  FIXTURES: @create_new_process_for_each_test()
  C1_WEAK_ORACLE: agree — exact string == on generated text
  C2_REALISTIC_BREAKAGE: agree — cuBLAS kernel selection changes with batch size
  C3_NO_UPDATE_PATH: agree — reference is self-generated, not a refreshable golden
  C4_NO_STRONG_CONTRACT: agree — Not Strong #6: batch size invariance without BI mode
  CLASSIFICATION: COINCIDENTALLY_CORRECT
  VERDICT: COINCIDENTALLY_CORRECT
  CODE_SNIPPET: |
    prompts = [example_system_message + prompt] * 64
    for response in responses:
        assert response.outputs[0].text == ref_output
```

## Example 2: test_cpu_offload (STRONG_CONTRACT — Phase 2 reclassifies)

Shows Phase 2 catching a Phase 1 error. Phase 1 classified as COINCIDENTALLY_CORRECT, Phase 2 identifies Strong Contract #4.

### Phase 1 output (incorrect)

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
  C2_REALISTIC_BREAKAGE: yes — different loading paths may use different kernels
  C3_NO_UPDATE_PATH: yes — no golden to refresh
  C4_NO_STRONG_CONTRACT: yes — no contract found
  CLASSIFICATION: COINCIDENTALLY_CORRECT
  VERDICT: COINCIDENTALLY_CORRECT
  CODE_SNIPPET: |
    compare_two_settings(model, base_args, offload_args)
```

### Phase 2 output (RECLASSIFY)

```
CANDIDATE: test_cpu_offload
  PHASE_1_CLASSIFICATION: COINCIDENTALLY_CORRECT
  PHASE_1_VERDICT: COINCIDENTALLY_CORRECT
  REVIEW: RECLASSIFY — Phase 1 missed Strong Contract #4
  FILE: tests/basic_correctness/test_cpu_offload.py
  LINE: 42
  COMPARISON: normal loading vs CPU offload loading
  ORACLE: exact dict equality (compare_two_settings)
  HELPER: compare_two_settings
  BATCH_INVARIANT_ENABLED: no
  CODE_PATH_VERIFIED: no
  FIXTURES: none relevant
  C1_WEAK_ORACLE: agree — exact dict equality
  C2_REALISTIC_BREAKAGE: disagree — data movement uses identical kernels, no numeric divergence
  C3_NO_UPDATE_PATH: agree — no golden to refresh
  C4_NO_STRONG_CONTRACT: disagree — Strong Contract #4: CPU offload must not change math
  CLASSIFICATION: STRONG_CONTRACT
  VERDICT: NOT_COINCIDENTALLY_CORRECT
  CODE_SNIPPET: |
    compare_two_settings(model, base_args, offload_args)
```

## What NOT to do

Do not write Phase 2 output like this:

```
### Candidate 1: test_cascade_attention

**What's compared:** batch=1 vs batch=64...

**Decision tree:**
1. **Strong contract?** No...
2. **Update path?** No...

test_cascade_attention → COINCIDENTALLY_CORRECT | reason: ...
```

This is prose. Every candidate must be a structured block with all fields. No `###` headers, no `**bold labels:**`, no numbered decision trees per candidate.
