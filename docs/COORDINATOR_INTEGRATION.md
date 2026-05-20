# Coordinator Agent Integration with Orchestrator

## Overview

The coordinator agent has been updated to emit `handoff_request` JSON when delegating tasks to specialist agents. The orchestration layer validates these handoffs in real-time.

## How It Works

### 1. Coordinator Emits Handoff Request

When routing to a specialist, the coordinator agent outputs structured JSON:

```json
{
  "type": "handoff_request",
  "from_agent": "coordinator-agent",
  "to_agent": "dynamo-debugger-agent",
  "task": {
    "type": "debug_graph_break",
    "issue": "x.item() causes graph break",
    "context": {
      "code_snippet": "def fn(x): return x[x.item()]"
    }
  },
  "expected_deliverable": "structured_json",
  "priority": "high"
}
```

### 2. Orchestrator Validates

The orchestration layer validates the handoff against:
- ✅ **Schema**: Matches `schemas/handoff_request.json`
- ✅ **Allowlist**: `to_agent` is in coordinator's `callable_agents` list

### 3. Specialist Responds

The specialist agent (e.g., dynamo-debugger-agent) returns structured JSON matching its response schema (e.g., `dynamo_response.json`).

### 4. Orchestrator Validates Response

The orchestration layer validates the response and logs metrics:

```
[ORCHESTRATOR] Validating response from dynamo-debugger-agent
[ORCHESTRATOR] ✓ Schema valid
[ORCHESTRATOR] coordinator → dynamo-debugger (125ms) ✓
```

## Usage in Claude Code

### Interactive Mode (Current Implementation)

**Step 1: User asks question**
```
User: "Why does len() cause a graph break?"
```

**Step 2: Load coordinator (via Agent tool or skill)**
```
In Claude Code conversation, the coordinator skill is loaded automatically
or you can explicitly invoke it.
```

**Step 3: Coordinator analyzes and emits handoff**

The coordinator outputs:
```markdown
I'll route this to the dynamo-debugger-agent for analysis.

**Handoff Request:**
```json
{
  "type": "handoff_request",
  "from_agent": "coordinator-agent",
  "to_agent": "dynamo-debugger-agent",
  "task": {
    "type": "debug_graph_break",
    "issue": "len() causes graph break"
  },
  "expected_deliverable": "structured_json",
  "priority": "high"
}
```

Would you like me to proceed with this delegation?
```

**Step 4: Validate handoff (optional, for development)**

```bash
# Copy the JSON and validate it
python scripts/validate_handoff.py '<paste JSON here>'
```

**Step 5: User confirms, specialist invoked**

User says "yes" → Claude Code Agent tool spawns dynamo-debugger-agent

**Step 6: Coordinator synthesizes response**

Coordinator receives specialist response and synthesizes it for the user.

## Validation Script Usage

### Quick Validation

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai

# Validate from argument
python scripts/validate_handoff.py '{
  "type": "handoff_request",
  "from_agent": "coordinator-agent",
  "to_agent": "dynamo-debugger-agent",
  "task": {"type": "debug_graph_break", "issue": "test"},
  "expected_deliverable": "structured_json",
  "priority": "high"
}'
```

### Validate from stdin

```bash
echo '{
  "type": "handoff_request",
  "from_agent": "coordinator-agent",
  "to_agent": "dynamo-debugger-agent",
  "task": {"type": "debug_graph_break", "issue": "test"},
  "expected_deliverable": "structured_json",
  "priority": "high"
}' | python scripts/validate_handoff.py -
```

### Expected Output

**Valid handoff:**
```
[ORCHESTRATOR] Initialized (5 agents, 5 schemas)
[...initialization output...]

✅ Handoff valid
   Schema: ✓
   Allowlist: ✓
```

**Invalid handoff (permissive mode):**
```
[ORCHESTRATOR] ⚠️  Warnings (allowing anyway):
[ORCHESTRATOR]     Allowlist: coordinator-agent cannot call unknown-agent

✅ Handoff valid  # Still valid in permissive mode!
   Schema: ✓
   Allowlist: ✗

Warnings:
  - Allowlist: coordinator-agent cannot call unknown-agent
```

## Bisect-First Workflow Example

Complete example showing the recommended debugging workflow:

### User Request
```
User: "My code fails to compile: def fn(x): return x[x.item()]"
```

### Coordinator Response (Step 1: Load compile-overview)

```markdown
I'll help debug this compilation issue using the bisect-first workflow.

**Step 1**: Load compile-overview skill for pipeline context
**Step 2**: Run bisector to identify failing stage
**Step 3**: Route to appropriate specialist based on bisect results

Let me start by running the bisector...
```

### Coordinator Emits Bisector Handoff

```json
{
  "type": "handoff_request",
  "from_agent": "coordinator-agent",
  "to_agent": "bisector-agent",
  "task": {
    "type": "bisect_failure",
    "issue": "compilation failure",
    "context": {
      "code_snippet": "def fn(x): return x[x.item()]"
    }
  },
  "expected_deliverable": "bisection_result",
  "priority": "high"
}
```

**Validation:**
```
[ORCHESTRATOR] coordinator → bisector
[ORCHESTRATOR] ✓ Schema valid
[ORCHESTRATOR] ✓ Allowlist valid
```

### Bisector Result

```json
{
  "backend": "eager",
  "subsystem": null,
  "operation": null,
  "message": "Graph break at Dynamo stage"
}
```

### Coordinator Emits Dynamo Debugger Handoff

```json
{
  "type": "handoff_request",
  "from_agent": "coordinator-agent",
  "to_agent": "dynamo-debugger-agent",
  "task": {
    "type": "debug_graph_break",
    "issue": "Graph break on x.item()",
    "context": {
      "code_snippet": "def fn(x): return x[x.item()]",
      "bisect_result": "backend='eager'"
    }
  },
  "expected_deliverable": "structured_json",
  "priority": "high"
}
```

**Validation:**
```
[ORCHESTRATOR] coordinator → dynamo-debugger
[ORCHESTRATOR] ✓ Schema valid
[ORCHESTRATOR] ✓ Allowlist valid
```

### Dynamo Debugger Response

```json
{
  "specialist": "dynamo-debugger-agent",
  "version": "1.0.0",
  "task": "debug graph break on x.item()",
  "confidence": "high",
  "insight": "Graph breaks on tensor.item() because it forces eager execution",
  "files": ["torch/_dynamo/variables/tensor.py:450"],
  "guidance": "...",
  "code": "...",
  "steps": [...],
  "dependencies": [],
  "pitfalls": [...],
  "skill_references": ["pytorch-dynamo/SKILL.md"]
}
```

**Validation:**
```
[ORCHESTRATOR] Validating response from dynamo-debugger-agent
[ORCHESTRATOR] ✓ Schema valid
[ORCHESTRATOR] coordinator → dynamo-debugger (125ms) ✓
```

### Coordinator Synthesis

```markdown
## Summary
Graph breaks on `tensor.item()` because it's data-dependent and forces eager execution. Dynamo can't trace operations where control flow depends on tensor values.

## Implementation
1. Avoid `.item()` in compiled code - use symbolic shapes instead
2. If you need the value, move it outside the compiled function
3. Alternative: use `torch.compile(..., dynamic=True)` for dynamic shapes

## Code
```python
# Instead of:
def fn(x):
    return x[x.item()]  # Breaks graph

# Do:
def fn(x, idx):
    return x[idx]  # Pass index as argument

compiled_fn = torch.compile(fn)
```

---
*Sources: bisector-agent, dynamo-debugger-agent*
*Handoffs validated: 2/2 ✓*
```

## Agent Names Reference

All agents use the "debugger" naming convention:

- `coordinator-agent` - Orchestrator
- `dynamo-debugger-agent` - Dynamo stage specialist
- `inductor-debugger-agent` - Inductor stage specialist
- `aot-debugger-agent` - AOT Autograd specialist
- `bisector-agent` - Failure isolation specialist

## Callable Agents Matrix

| From Agent | Can Call |
|------------|----------|
| coordinator-agent | dynamo-debugger-agent, inductor-debugger-agent, aot-debugger-agent, bisector-agent |
| dynamo-debugger-agent | coordinator-agent, inductor-debugger-agent |
| inductor-debugger-agent | coordinator-agent, dynamo-debugger-agent |
| aot-debugger-agent | coordinator-agent, dynamo-debugger-agent, inductor-debugger-agent |
| bisector-agent | coordinator-agent, dynamo-debugger-agent, inductor-debugger-agent, aot-debugger-agent |

## Troubleshooting

### Handoff validation fails

**Problem:** Orchestrator reports allowlist violation

**Solution:** Check `managed-agent-cookbooks/*/agent.yaml` - ensure `to_agent` is in `from_agent`'s `callable_agents` list

### Schema validation fails

**Problem:** Orchestrator reports schema violation

**Solution:** Ensure handoff JSON matches `schemas/handoff_request.json` structure. Required fields:
- `type` (must be "handoff_request")
- `from_agent`
- `to_agent`
- `task` (with `type` and `issue`)
- `expected_deliverable`

### Response validation fails

**Problem:** Orchestrator reports response schema violation

**Solution:** Ensure specialist response matches its schema (e.g., `dynamo_response.json`). Required fields vary by specialist.

## Future Enhancements

### Automatic Validation Hook

**Status:** Not implemented (manual validation only)

**Planned:** Hook into Claude Code to automatically validate handoffs without manual script invocation

### Strict Enforcement Mode

**Status:** Only permissive mode available

**Planned:** Configurable strict mode that blocks invalid handoffs instead of warning

### Metrics Dashboard

**Status:** Console logging only

**Planned:** Persistent metrics tracking and visualization
