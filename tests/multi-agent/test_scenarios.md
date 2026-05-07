# Multi-Agent Test Scenarios

Test cases for validating coordinator routing, specialist responses, and report synthesis.

## Test 1: Simple Lowering Addition

**User Task:** "Add support for torch.special.bessel_j0"

### Expected Routing
- Coordinator analyzes: implementation task, Inductor domain
- Suggests: inductor-expert
- User confirms
- Spawns inductor-expert

### Expected Specialist Report

```json
{
  "specialist": "inductor-expert",
  "task": "Add support for torch.special.bessel_j0",
  "confidence": "high",
  "findings": {
    "key_insight": "bessel_j0 is element-wise, use Pointwise IR node",
    "relevant_files": [
      "torch/_inductor/lowering.py:1500",
      "torch/_inductor/ir.py:234"
    ],
    "related_concepts": ["Pointwise", "ops.bessel_j0", "@register_lowering"],
    "ir_node_type": "Pointwise"
  },
  "guidance": "...",
  "code_example": "@register_lowering(aten.special.bessel_j0)\ndef bessel_j0_lowering(x):\n    return make_pointwise(ops.bessel_j0)(x)",
  "next_steps": [
    "Add lowering in torch/_inductor/lowering.py",
    "Use Pointwise.create or make_pointwise helper",
    "Test with torch.compile"
  ]
}
```

### Expected Coordinator Synthesis

```markdown
## Summary
Add bessel_j0 support by registering a Pointwise lowering in Inductor.

## Implementation Approach

1. Add lowering in torch/_inductor/lowering.py:1500
   Use @register_lowering(aten.special.bessel_j0) decorator

2. Implement using Pointwise IR node
   Call make_pointwise(ops.bessel_j0)(x) for element-wise operation

3. Test with torch.compile
   Verify fusion with TORCH_LOGS=fusion

## Code Example
[code from specialist]

---
*Source: inductor-expert*
```

### Validation Criteria
- ✅ Coordinator correctly identifies Inductor domain
- ✅ Only inductor-expert suggested (no dynamo-expert)
- ✅ User confirmation requested before spawning
- ✅ Synthesis includes file:line references
- ✅ Summary is concise (2-3 sentences)

---

## Test 2: Graph Break Debugging

**User Task:** "Why does my custom dataclass cause graph breaks?"

### Expected Routing
- Coordinator analyzes: debugging + Dynamo domain
- Suggests: torch-compile-ai (if logs available) + dynamo-expert
- User confirms
- Queries torch-compile-ai first, then spawns dynamo-expert

### Expected Flow

**Step 1: Debug Tracer MCP Query**
```
Coordinator: Do you have TORCH_LOGS=graph_breaks output I can analyze?

User: [provides log or "no"]

Coordinator: [queries torch-compile-ai parse_graph_breaks if log available]
```

**Step 2: Dynamo Expert Report**
```json
{
  "specialist": "dynamo-expert",
  "task": "Why does custom dataclass cause graph breaks?",
  "confidence": "high",
  "findings": {
    "key_insight": "Custom dataclasses need VariableTracker subclass to avoid graph breaks",
    "relevant_files": [
      "torch/_dynamo/variables/user_defined.py:45",
      "torch/_dynamo/trace_rules.py:150"
    ]
  },
  "guidance": "Create a VariableTracker subclass...",
  "next_steps": [
    "Create MyDataclassVariable(UserDefinedObjectVariable)",
    "Implement var_getattr for field access",
    "Register in trace_rules.py"
  ]
}
```

### Expected Coordinator Synthesis

```markdown
## Summary
Your custom dataclass causes graph breaks because Dynamo doesn't know how to track it. Add a VariableTracker subclass to enable tracing.

## Analysis
[If torch-compile-ai was queried:]
The graph break occurs at: [file:line from parsed logs]
Reason: [break reason from logs]

## Solution

1. Create VariableTracker subclass in torch/_dynamo/variables/
2. Implement var_getattr for field access
3. Register type in torch/_dynamo/trace_rules.py:150

[code example]

---
*Sources: torch-compile-ai, dynamo-expert*
```

### Validation Criteria
- ✅ Coordinator suggests both torch-compile-ai and dynamo-expert
- ✅ Queries MCP before spawning subagent
- ✅ Synthesis combines both sources
- ✅ Clearly attributes which source provided what info

---

## Test 3: Multi-Domain Design Task

**User Task:** "Design approach for implementing torch.linalg.det (determinant)"

### Expected Routing
- Coordinator analyzes: design task, cross-domain (Dynamo + Inductor)
- Suggests: dynamo-expert + inductor-expert (parallel)
- User confirms
- Spawns both specialists in parallel

### Expected Dynamo Expert Report

```json
{
  "specialist": "dynamo-expert",
  "task": "Design approach for torch.linalg.det - Dynamo considerations",
  "confidence": "medium",
  "findings": {
    "key_insight": "Determinant is a standard tensor operation, no special Dynamo handling needed unless custom decomposition",
    "relevant_files": ["torch/_dynamo/variables/tensor.py:TensorVariable"]
  },
  "guidance": "linalg.det should trace normally as a TensorVariable method call. Only add custom handling if you need to decompose det into simpler ops at graph capture time.",
  "next_steps": [
    "Verify det traces without graph breaks (likely already works)",
    "If decomposition needed, add to torch/_decomp/decompositions.py"
  ]
}
```

### Expected Inductor Expert Report

```json
{
  "specialist": "inductor-expert",
  "task": "Design approach for torch.linalg.det - Inductor lowering",
  "confidence": "medium",
  "findings": {
    "key_insight": "Determinant is complex linear algebra, use ExternKernel to call cuBLAS/LAPACK rather than custom IR",
    "relevant_files": [
      "torch/_inductor/lowering.py:linear algebra ops",
      "torch/_inductor/kernel/mm.py:matrix operations"
    ],
    "ir_node_type": "ExternKernel"
  },
  "guidance": "For linalg.det, register lowering that creates ExternKernel calling aten.linalg.det (which uses cuBLAS/LAPACK). Don't attempt to decompose into Inductor IR - numerical stability and performance require optimized library.",
  "next_steps": [
    "Register @register_lowering(aten.linalg.det)",
    "Return ExternKernel that calls library implementation",
    "Test with torch.compile on GPU and CPU"
  ],
  "performance_notes": "Library implementations are 10-100x faster than manual IR for dense linear algebra."
}
```

### Expected Coordinator Synthesis

```markdown
## Summary
Implement linalg.det by ensuring it traces cleanly in Dynamo (likely already works) and adding an Inductor lowering that delegates to optimized libraries (cuBLAS/LAPACK).

## Design Approach

**Dynamo Stage (Graph Capture):**
- det should trace as standard TensorVariable method
- No special handling needed unless custom decomposition required
- Verify no graph breaks with test case

**Inductor Stage (Lowering):**
- Use ExternKernel IR node to call library implementation
- Do NOT decompose into manual IR (performance + numerical stability)
- Register lowering for aten.linalg.det

## Implementation Steps

1. **Verify Dynamo tracing** (torch/_dynamo/variables/tensor.py)
   Test that det traces without graph breaks
   
2. **Add Inductor lowering** (torch/_inductor/lowering.py)
   Register @register_lowering(aten.linalg.det)
   Return ExternKernel calling library

3. **Test on GPU and CPU**
   Verify correctness and performance vs eager mode

## Performance Notes
Library implementations are 10-100x faster than manual decomposition. ExternKernel is the right choice for dense linear algebra.

---
*Sources: dynamo-expert, inductor-expert (consulted in parallel)*
```

### Validation Criteria
- ✅ Coordinator suggests both specialists
- ✅ Notes that they can run in parallel
- ✅ Synthesis combines perspectives from both domains
- ✅ Clearly delineates Dynamo vs Inductor considerations
- ✅ Unified implementation steps spanning both stages

---

## Test 4: API Lookup (No Subagent)

**User Task:** "What are the parameters for Pointwise.__init__?"

### Expected Routing
- Coordinator analyzes: API lookup, simple query
- Suggests: steering-mcp only
- User confirms
- Queries steering-mcp directly (no subagent spawn)

### Expected Flow

```
Coordinator: Based on your task "Pointwise parameters", I recommend:
1. steering-mcp - for API signature lookup

Should I proceed?

User: yes

Coordinator: [calls steering-mcp query_api_docs("torch._inductor.ir.Pointwise.__init__")]
```

### Expected MCP Response

```json
{
  "signature": "Pointwise.__init__(device, dtype, inner_fn, ranges)",
  "parameters": {
    "device": "torch.device - target device",
    "dtype": "torch.dtype - output dtype",
    "inner_fn": "Callable - element-wise computation",
    "ranges": "List[Expr] - iteration space"
  },
  "file": "torch/_inductor/ir.py:234"
}
```

### Expected Coordinator Response

```markdown
## Pointwise.__init__ Parameters

**Signature:** `Pointwise.__init__(device, dtype, inner_fn, ranges)`

**Parameters:**
- `device` (torch.device) - Target device for computation
- `dtype` (torch.dtype) - Output data type
- `inner_fn` (Callable) - Element-wise computation function
- `ranges` (List[Expr]) - Iteration space dimensions

**Defined at:** torch/_inductor/ir.py:234

---
*Source: steering-mcp*
```

### Validation Criteria
- ✅ Coordinator identifies as simple lookup
- ✅ Only steering-mcp suggested (no subagents)
- ✅ Direct MCP query, no spawn
- ✅ Response formatted clearly
- ✅ File reference included

---

## Test 5: Fusion Debugging

**User Task:** "Why aren't my convolution and ReLU fusing? Here's my TORCH_LOGS output: [log]"

### Expected Routing
- Coordinator analyzes: debugging, fusion, has logs
- Suggests: torch-compile-ai + inductor-expert
- User confirms
- Queries torch-compile-ai first, then spawns inductor-expert with findings

### Expected Flow

**Step 1: Parse Fusion Logs**
```
Coordinator: [calls torch-compile-ai parse_fusion_decisions with user's log]
```

**MCP Response:**
```
Fusion Analysis:

Failed Fusions:
- conv2d + relu
  Reason: ExternKernel (cuDNN conv) cannot vertically fuse
  Suggestion: Enable epilogue_fusion or cudnn_fusion config
  
Location: [file:line from log]
```

**Step 2: Inductor Expert Guidance**
```json
{
  "specialist": "inductor-expert",
  "task": "Why conv+relu not fusing [with debug-tracer findings]",
  "confidence": "medium",
  "findings": {
    "key_insight": "Convolution is ExternKernel calling cuDNN, needs epilogue fusion pattern",
    "relevant_files": ["torch/_inductor/kernel/conv.py:conv_epilogue"]
  },
  "guidance": "Enable config.epilogue_fusion to match conv+relu epilogue pattern...",
  "next_steps": [
    "Set config.epilogue_fusion = True",
    "Verify pattern matches conv.py epilogue",
    "Re-run with TORCH_LOGS=fusion to confirm"
  ]
}
```

### Expected Coordinator Synthesis

```markdown
## Summary
Your conv+relu isn't fusing because convolution is an ExternKernel (cuDNN) that doesn't participate in standard vertical fusion. Enable epilogue fusion to match this pattern.

## Analysis
The fusion logs show:
- Failed fusion: conv2d + relu
- Reason: ExternKernel cannot vertically fuse
- Location: [file:line]

## Solution

1. Enable epilogue fusion config
   ```python
   import torch._inductor.config as config
   config.epilogue_fusion = True
   config.cudnn_fusion = True
   ```

2. Verify your pattern matches recognized epilogues
   Check torch/_inductor/kernel/conv.py for supported patterns

3. Re-compile and verify fusion
   Run with TORCH_LOGS=fusion to confirm fusion occurs

## Performance Impact
Conv+ReLU fusion saves 10-20% by avoiding intermediate buffer write.

---
*Sources: torch-compile-ai, inductor-expert*
```

### Validation Criteria
- ✅ Coordinator parses logs first with torch-compile-ai
- ✅ Passes parsed findings to inductor-expert for deeper analysis
- ✅ Synthesis combines both sources clearly
- ✅ Actionable config changes provided
- ✅ Verification steps included

---

## Success Metrics

### Routing Accuracy
Target: 80%+ correct specialist suggestions

Measure:
- Count correct suggestions / total test cases
- Correct = task could be solved by suggested specialists
- No unnecessary specialists suggested

### Context Efficiency
Target: 60%+ reduction vs loading all skills

Measure:
- Single specialist: ~50KB vs ~150KB all skills = 67% reduction ✓
- Multi specialist: ~90KB vs ~150KB = 40% reduction ✓
- API lookup (MCP only): ~20KB vs ~150KB = 87% reduction ✓

### Synthesis Quality
Target: Coherent, actionable guidance

Criteria:
- ✅ Summary is 2-3 sentences
- ✅ Steps are sequential and actionable
- ✅ Sources are clearly attributed
- ✅ File:line references included
- ✅ Code examples when relevant

### User Experience
Target: Clear, transparent, empowering

Criteria:
- ✅ Routing suggestions are understandable
- ✅ User confirmation always requested
- ✅ Progressive disclosure (summary → details)
- ✅ User can override routing
- ✅ Sources are transparent

---

## Running Tests

### Manual Testing
1. Load coordinator-concise.md as system prompt
2. Provide test task
3. Verify routing suggestion matches expected
4. Confirm routing
5. Mock specialist response (or spawn real subagent)
6. Verify synthesis quality

### Automated Validation (Future)
```python
# tests/multi-agent/test_routing.py
def test_simple_lowering_routing():
    task = "Add support for torch.special.bessel_j0"
    routing = coordinator.analyze_task(task)
    assert routing == ["inductor-expert"]
    assert "lowering" in routing.reasoning
```

---

## Iteration Plan

After manual testing:
1. **Measure routing accuracy** - Track suggestions vs actual needs
2. **Refine decision tree** - Adjust keyword matching
3. **Improve synthesis templates** - Standardize output format
4. **Add edge cases** - Test ambiguous tasks, multi-domain overlaps
5. **Optimize prompts** - Reduce coordinator/specialist prompt size
