# Orchestration Layer - torch-compile Multi-Agent System

## Overview

The orchestration layer validates handoff requests and agent responses for the torch-compile multi-agent debugging system. It operates in **permissive mode** by default: logging warnings for invalid handoffs but allowing them to proceed.

## Architecture

```
User Question
    ↓
Coordinator Agent (uses compile-overview skill)
    ├─ Analyzes task
    ├─ Determines routing (bisect-first workflow)
    └─ Emits handoff_request JSON
    ↓
Orchestrator (validation layer)
    ├─ Validates handoff_request schema ⚠️ (warn if invalid)
    ├─ Checks allowlist graph ⚠️ (warn if not in callable_agents)
    └─ Logs to console: [ORCHESTRATOR] coordinator → dynamo-debugger
    ↓
Claude Code Agent Tool
    ├─ Spawns specialist agent (e.g., dynamo-debugger-agent)
    └─ Returns structured JSON response
    ↓
Orchestrator (response validation)
    ├─ Validates response schema ⚠️ (warn if invalid)
    └─ Logs: [ORCHESTRATOR] coordinator → dynamo-debugger (125ms) ✓
    ↓
Coordinator synthesizes response for user
```

## Components

### 1. Agent Allowlist (`scripts/agent_allowlist.py`)

Builds and validates callable_agents graph from agent.yaml manifests.

**Functions:**
- `build_allowlist(agent_manifest_dir)` - Parse agent.yaml files, extract callable_agents
- `validate_handoff_allowlist(handoff, allowlist)` - Check if handoff is in allowlist

**Example:**
```python
from pathlib import Path
from scripts.agent_allowlist import build_allowlist

allowlist = build_allowlist(Path("managed-agent-cookbooks"))
# {
#     "coordinator-agent": ["dynamo-debugger-agent", "inductor-debugger-agent", ...],
#     "dynamo-debugger-agent": ["coordinator-agent", "inductor-debugger-agent"],
#     ...
# }
```

### 2. Schema Validator (`scripts/orchestrate_schemas.py`)

Loads and validates JSON schemas for handoff requests and agent responses.

**Functions:**
- `load_schemas(schema_dir)` - Load all *.json schemas
- `validate_handoff_request(handoff, schemas)` - Validate against handoff_request.json
- `validate_response(response, agent_name, schemas)` - Validate against stage-specific schema

**Example:**
```python
from pathlib import Path
from scripts.orchestrate_schemas import load_schemas, validate_response

schemas = load_schemas(Path("schemas"))
response = {"specialist": "dynamo-debugger-agent", "version": "1.0.0", ...}
valid, errors = validate_response(response, "dynamo-debugger-agent", schemas)
```

### 3. Core Orchestrator (`scripts/orchestrate.py`)

Main orchestration class that ties everything together.

**Class:** `TorchCompileOrchestrator`

**Methods:**
- `validate_handoff(handoff)` - Validate handoff request (schema + allowlist)
- `validate_agent_response(response, agent_name)` - Validate agent response
- `log_handoff(from_agent, to_agent, latency_ms, success, warnings)` - Log to console

**Example:**
```python
from pathlib import Path
from scripts.orchestrate import TorchCompileOrchestrator

orchestrator = TorchCompileOrchestrator(
    Path("managed-agent-cookbooks"),
    Path("schemas")
)

# Validate handoff
handoff = {
    "type": "handoff_request",
    "from_agent": "coordinator-agent",
    "to_agent": "dynamo-debugger-agent",
    "task": {...},
    "expected_deliverable": "structured_json",
    "priority": "high"
}

result = orchestrator.validate_handoff(handoff)
# {"valid": True, "warnings": [], "schema_valid": True, "allowlist_valid": True}
```

## Validation Modes

### Permissive Mode (Default)

- Validates handoffs and responses
- Logs warnings for violations
- **Allows all handoffs to proceed**
- Development-friendly

**Console output:**
```
[ORCHESTRATOR] ⚠️  Warnings (allowing anyway):
[ORCHESTRATOR]     Allowlist: dynamo-debugger-agent cannot call bisector-agent
```

### Strict Mode (Not Implemented Yet)

- Validates handoffs and responses
- **Blocks invalid handoffs**
- Returns error to coordinator
- Production-ready (future)

## Usage

### Running Orchestrator in Console Mode

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
python scripts/orchestrate.py --mode console
```

**Output:**
```
[ORCHESTRATOR] Initialized (5 agents, 5 schemas)
[ORCHESTRATOR] Allowlist: coordinator → 4 agents
[ORCHESTRATOR] Allowlist: dynamo-debugger → 2 agents
[ORCHESTRATOR] Allowlist: inductor-debugger → 2 agents
[ORCHESTRATOR] Allowlist: aot-debugger → 3 agents
[ORCHESTRATOR] Allowlist: bisector → 4 agents
[ORCHESTRATOR] Validation mode: PERMISSIVE (warnings only)
[ORCHESTRATOR] Ready for handoff validation
```

### Validating Handoff from JSON

```bash
echo '{
  "type": "handoff_request",
  "from_agent": "coordinator-agent",
  "to_agent": "dynamo-debugger-agent",
  "task": {"type": "debug_graph_break", "issue": "test"},
  "expected_deliverable": "structured_json",
  "priority": "high"
}' | python scripts/orchestrate.py --mode validate
```

### Running Tests

```bash
pytest tests/test_orchestrate.py -v
```

## Bisect-First Workflow

The orchestration layer supports the bisect-first debugging workflow:

```
1. User: "My code fails to compile: def fn(x): return x[x.item()]"
   ↓
2. Coordinator loads compile-overview skill
   → Explains pipeline, recommends bisect-first
   ↓
3. Coordinator loads compile-bisect skill
   → Runs bisector: backend='eager' (Dynamo stage)
   ↓
4. Coordinator emits handoff_request:
   {
     "from_agent": "coordinator-agent",
     "to_agent": "dynamo-debugger-agent",
     "task": {
       "type": "debug_graph_break",
       "issue": "x.item() causes graph break",
       "context": {"bisect_result": "backend='eager'", ...}
     }
   }
   ↓
5. Orchestrator validates:
   [ORCHESTRATOR] Validating handoff: coordinator → dynamo-debugger
   [ORCHESTRATOR] ✓ Schema valid
   [ORCHESTRATOR] ✓ Allowlist valid
   ↓
6. Claude Code spawns dynamo-debugger-agent
   → Agent has pytorch-dynamo + compile-trace-dynamo skills
   → Returns structured JSON with diagnosis
   ↓
7. Orchestrator validates response:
   [ORCHESTRATOR] Validating response from dynamo-debugger-agent
   [ORCHESTRATOR] ✓ Schema valid
   [ORCHESTRATOR] coordinator → dynamo-debugger (125ms) ✓
   ↓
8. Coordinator synthesizes response for user
```

## Console Output Reference

### Initialization

```
[ORCHESTRATOR] Initialized (5 agents, 5 schemas)
[ORCHESTRATOR] Allowlist: <agent> → <count> agents
[ORCHESTRATOR] Validation mode: PERMISSIVE (warnings only)
[ORCHESTRATOR] Ready for handoff validation
```

### Valid Handoff

```
[ORCHESTRATOR] coordinator → dynamo-debugger (125ms) ✓
```

### Invalid Handoff (Permissive Mode)

```
[ORCHESTRATOR] ⚠️  Warnings (allowing anyway):
[ORCHESTRATOR]     Schema: 'task' is a required property
[ORCHESTRATOR]     Allowlist: dynamo-debugger-agent cannot call bisector-agent
```

### Invalid Response (Permissive Mode)

```
[ORCHESTRATOR] ⚠️  Response warnings from dynamo-debugger-agent:
[ORCHESTRATOR]     'version' is a required property
```

## Agent Names

The system supports two naming conventions for compatibility:

**YAML Files (actual names):**
- `coordinator-agent`
- `dynamo-debugger-agent`
- `inductor-debugger-agent`
- `aot-debugger-agent`
- `bisector-agent`

**Schemas (legacy names, supported as aliases):**
- `dynamo-expert-agent` → maps to `dynamo-debugger-agent`
- `inductor-expert-agent` → maps to `inductor-debugger-agent`

Both naming conventions work for validation.

## Troubleshooting

### "Unknown agent" error

**Cause:** Agent name doesn't exist in allowlist  
**Fix:** Check agent.yaml files in `managed-agent-cookbooks/`, ensure agent name matches

### "Cannot call" warning

**Cause:** Handoff not in callable_agents list  
**Fix:** Update agent.yaml to add target agent to callable_agents

### "Schema not found" error

**Cause:** Missing JSON schema file  
**Fix:** Ensure all schema files exist in `schemas/` directory

### Tests failing

**Cause:** Agent names mismatch or schema changes  
**Fix:**
```bash
# Check agent names
ls managed-agent-cookbooks/*/agent.yaml | xargs head -1

# Verify schemas
ls schemas/*.json

# Re-run sync if needed
python scripts/sync-agent-skills.py
```

## Future Enhancements

### Strict Enforcement Mode

**Status:** Not implemented (permissive mode only for now)

**Planned:**
- Block invalid handoffs instead of warning
- Return errors to coordinator
- Configurable per-environment (strict in CI, permissive in dev)

### Headless Mode

**Status:** Not needed (no Anthropic API access)

**If API access obtained:**
- Programmatic orchestration API
- Auto-routing without user confirmation
- CI/CD integration

### Advanced Telemetry

**Status:** Console logging only

**Planned:**
- JSON metrics export
- Persistent logging to files
- Dashboard visualization
- Performance analytics

## References

- **Plan:** `/home/dev/.claude/plans/calm-sparking-wreath.md`
- **Agent Manifests:** `managed-agent-cookbooks/*/agent.yaml`
- **Schemas:** `schemas/*.json`
- **Tests:** `tests/test_orchestrate.py`
- **Financial-services pattern:** https://github.com/anthropics/financial-services
