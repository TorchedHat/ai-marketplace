# Phase 2 Output Example

This example shows the correct output format for the review-agent. Every candidate must use this exact structured block format. No prose, no markdown headers, no decision tree narratives.

## Correct Output

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
  C2_REALISTIC_BREAKAGE: agree — PyTorch #182700, cuBLAS kernel selection changes with batch size
  C3_NO_UPDATE_PATH: agree — reference is self-generated at batch=1, not a refreshable golden
  C4_NO_STRONG_CONTRACT: agree — Not Strong #6: batch size invariance without BI mode
  CLASSIFICATION: COINCIDENTALLY_CORRECT
  VERDICT: COINCIDENTALLY_CORRECT
  CODE_SNIPPET: |
    prompts = [example_system_message + prompt] * 64
    responses = llm.generate(prompts, sampling_params)
    for response in responses:
        assert response.outputs[0].text == ref_output

CANDIDATE: test_cpu_offload
  PHASE_1_CLASSIFICATION: COINCIDENTALLY_CORRECT
  PHASE_1_VERDICT: COINCIDENTALLY_CORRECT
  REVIEW: RECLASSIFY — Phase 1 missed Strong Contract #4: data restoration must not change model math
  FILE: tests/basic_correctness/test_cpu_offload.py
  LINE: 42
  COMPARISON: normal loading vs CPU offload loading
  ORACLE: exact dict equality (compare_two_settings)
  HELPER: compare_two_settings
  BATCH_INVARIANT_ENABLED: no
  CODE_PATH_VERIFIED: no
  FIXTURES: none relevant
  C1_WEAK_ORACLE: agree — exact dict equality via compare_two_settings
  C2_REALISTIC_BREAKAGE: disagree — data movement/restoration uses identical kernels, no numeric divergence path
  C3_NO_UPDATE_PATH: agree — no golden to refresh
  C4_NO_STRONG_CONTRACT: disagree — Strong Contract #4: CPU offload restoration must not change model math
  CLASSIFICATION: STRONG_CONTRACT
  VERDICT: NOT_COINCIDENTALLY_CORRECT
  CODE_SNIPPET: |
    compare_two_settings(model, base_args, offload_args)

# Evidence Summary
Test files in scope: 2
Candidates analyzed: 2

| Classification | Count | Action |
|---|---|---|
| COINCIDENTALLY_CORRECT | 1 | Needs fixing |
| STRONG_CONTRACT | 1 | Remove from list |

Phase 1 agreed: 1
Phase 1 reclassified: 1 (COINCIDENTALLY_CORRECT → STRONG_CONTRACT, missed Strong Contract #4)
```

## What NOT to do

Do not write output like this:

```
### Candidate 1: test_cascade_attention

**What's compared:** batch=1 vs batch=64...

**Decision tree:**
1. **Strong contract?** No...
2. **Update path?** No...

test_cascade_attention → COINCIDENTALLY_CORRECT | reason: ...
```

This is prose analysis, not structured output. Every candidate must be a single structured block with all fields present.
