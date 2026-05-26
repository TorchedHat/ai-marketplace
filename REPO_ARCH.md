# Repository Architecture

## Overview

Multi-agent development system for PyTorch torch.compile, organized using Anthropic's vertical plugin pattern. Provides steering MCP server, stage-specific skills, and structured agent definitions for debugging across all compilation stages (Dynamo, AOT Autograd, Inductor).

**Design Philosophy**:
- **Vertical Organization** - Domain-based structure (not type-based)
- **Agent Formalization** - Structured definitions with JSON schemas
- **Direct Log Interpretation** - Claude reads TORCH_LOGS directly with skill guidance
- **Skill Composition** - Agents bundle relevant skills automatically

## Project Structure

```
torch-compile-ai/
├── vertical-plugins/           # Phase 1: Skills organized by compilation stage (source of truth)
│   ├── dynamo-debugger/
│   │   ├── skills/            # compile-trace-dynamo, pytorch-dynamo
│   │   ├── prompts/dynamo-expert.md
│   │   └── README.md
│   ├── inductor-debugger/
│   │   ├── skills/            # compile-trace-inductor, pytorch-inductor
│   │   ├── prompts/inductor-expert.md
│   │   └── README.md
│   ├── aot-debugger/
│   │   └── skills/compile-trace-aot/
│   └── bisector/
│       └── skills/compile-bisect/
│
├── agent-plugins/              # Phase 2: Structured agent definitions
│   ├── coordinator-agent/
│   │   ├── agents/coordinator.md       # 5-section agent definition
│   │   └── skills/                     # Synced from coordinator/
│   ├── dynamo-debugger-agent/
│   │   ├── agents/dynamo-expert.md
│   │   └── skills/                     # Synced from vertical-plugins/
│   ├── inductor-debugger-agent/
│   │   ├── agents/inductor-expert.md
│   │   └── skills/
│   ├── aot-debugger-agent/
│   │   ├── agents/aot-expert.md
│   │   └── skills/
│   └── bisector-agent/
│       ├── agents/bisector.md
│       └── skills/
│
├── coordinator/                # Coordinator vertical (special case)
│   ├── skills/compile-overview/
│   ├── prompts/coordinator.md
│   └── README.md
│
├── managed-agent-cookbooks/    # Phase 2: Deployment manifests
│   ├── coordinator-agent/agent.yaml
│   ├── dynamo-debugger-agent/agent.yaml
│   ├── inductor-debugger-agent/agent.yaml
│   ├── aot-debugger-agent/agent.yaml
│   └── bisector-agent/agent.yaml
│
├── schemas/                    # Phase 2: Structured output schemas
│   ├── handoff_request.json
│   ├── coordinator_routing.json
│   ├── dynamo_response.json
│   ├── inductor_response.json
│   └── aot_response.json
│
├── scripts/                    # Phase 2: Automation
│   ├── sync-agent-skills.py   # Sync skills to agent bundles
│   └── validate-skills.py     # Lint and validate
│
├── tests/                      # Tests
│   └── multi-agent/           # Multi-agent test scenarios
└── examples/                   # Example usage
```

## Architecture

### Vertical Plugin Organization (Phase 1)

Skills are organized by **compilation stage** rather than file type:

**Benefits:**
- Each compilation stage's components grouped together
- Clear boundaries between Dynamo, AOT, Inductor concerns
- Easy to see what belongs to each stage
- Skills symlinked to agent bundles (single source of truth)

**Source of Truth:** `vertical-plugins/` and `coordinator/`
**Agent Bundles:** `agent-plugins/*/skills/` (symlinked from vertical-plugins)
**Backward Compatibility:** `.claude/skills/` symlinks to vertical-plugins

### Agent Formalization (Phase 2)

Structured agent definitions following Anthropic's 5-section pattern:

1. **Frontmatter** - Metadata, tools, callable_agents
2. **Identity** - Persona, scope boundaries
3. **Deliverables** - Structured JSON output
4. **Workflow** - Step-by-step process
5. **Guardrails** - NEVER/ALWAYS constraints

**Agent-to-Agent Communication:**
- `handoff_request.json` - Task routing between agents
- Stage-specific response schemas (dynamo, inductor, aot)
- `coordinator_routing.json` - Routing decisions

### Execution Flow

```
User Request
    ↓
Coordinator Agent (compile-overview skill)
    ├─ Analyzes task and suggests routing
    ├─ Confirms delegation with user
    └─ Routes to specialists
    ↓
    ├─→ steering-mcp (API lookups)
    ├─→ bisector-agent (automated failure isolation)
    ├─→ dynamo-debugger-agent (Dynamo analysis)
    ├─→ inductor-debugger-agent (Inductor analysis)
    └─→ aot-debugger-agent (AOT Autograd analysis)
    ↓
Synthesized Response (structured JSON)
```

### MCP Server

**Steering MCP** - PyTorch API documentation and semantic search:
- Query steering guidance for task approaches
- Query API docs for function signatures
- Indexed modules: torch/_dynamo, torch/_functorch, torch/_inductor

Skills guide Claude to read TORCH_LOGS output and debug files directly rather than using intermediate parsers.

**External Dependencies** (runtime):
- `acp-steering-mcp` - Steering MCP server

**External Dependencies** (dev):
- `pytest>=8.0.0` - Testing
- `ruff` - Linting/formatting
- `pyright` - Type checking

## Key Components

### Skills by Stage

**Tracing Skills** - User-level debugging with TORCH_LOGS
- `compile-trace-dynamo` - Graph breaks, FX graphs, pre-grad passes
- `compile-trace-aot` - Functionalization, partitioning, post-grad passes
- `compile-trace-inductor` - Fusion decisions, IR post-fusion, kernel codegen

**Implementation Skills** - PyTorch contributor internals
- `pytorch-dynamo` - VariableTracker, guards, bytecode capture
- `pytorch-aot` - Functorch, vmap, functionalization
- `pytorch-inductor` - Lowerings, scheduling, Triton codegen

**Workflow Skills**
- `compile-overview` - Bisect-first workflow, pipeline overview
- `compile-bisect` - Automated failure isolation

### Debug Output Workflow

**Two types of torch.compile output:**

**1. Stdout logs (ephemeral):**
- TORCH_LOGS="graph_breaks" - Graph break locations and reasons
- TORCH_LOGS="fusion,schedule" - Fusion decisions
- Claude reads directly with skill guidance

**2. Debug files (persistent):**
- fx_graph_readable.py - FX graph structure
- output_code.py - Generated Triton/C++ kernels
- ir_post_fusion_*.txt - LoopBody IR
- Saved to torch_compile_debug/
- Claude reads directly with skill guidance

**Debugging Workflow:**
```python
# 1. Run code with TORCH_LOGS
stdout = bash("TORCH_LOGS='graph_breaks,fusion,output_code' python temp.py")

# 2. Show user the raw output
# Claude analyzes stdout directly using skill knowledge

# 3. Find and read debug files
debug_dir = find_latest("torch_compile_debug/run_*/")
fx_graph = read(f"{debug_dir}/fx_graph_readable.py")
output_code = read(f"{debug_dir}/output_code.py")

# 4. Claude analyzes files directly using skill knowledge
# No intermediate parsing - full context available
```

### File Locations

Real torch.compile output structure:
```
torch_compile_debug/run_<timestamp>-pid_<pid>/
└── torchinductor/
    ├── model__0_inference_0.0/
    │   ├── fx_graph_readable.py     # Dynamo FX graph
    │   ├── fx_graph_transformed.py  # After pre-grad passes
    │   ├── ir_pre_fusion.txt        # Inductor IR before fusion
    │   ├── ir_post_fusion.txt       # LoopBody after fusion
    │   └── output_code.py           # Generated Triton/C++ code
    ├── model__3_forward_4.3/        # AOT forward graph
    └── model__3_backward_6.4/       # AOT backward graph
```

## MCP Server Integration

### Steering MCP

Configured in `.mcp.json`:

```json
{
  "mcpServers": {
    "steering": {
      "command": "/home/dev/.venv/bin/acp-steering-mcp"
    }
  }
}
```

Provides semantic search over PyTorch API documentation (dynamo, functorch, inductor modules).

## Deployment Manifests

### Agent Manifests (Phase 2)

Each agent has an `agent.yaml` manifest in `managed-agent-cookbooks/`:

**Structure:**
```yaml
agent:
  definition: "../agent-plugins/<name>/agents/<name>.md"
  skills_path: "../agent-plugins/<name>/skills/"
  mcp_config:
    servers: [...]
  tools:
    allow: [...]
    deny: [...]
  callable_agents: [...]
  deployment:
    max_turns: 20
    timeout: 300
    mode: "interactive"
```

**Future:** Ready for headless deployment via Anthropic Managed Agents API

### JSON Schemas

Structured communication between agents:

- `handoff_request.json` - Agent-to-agent task handoff
- `coordinator_routing.json` - Coordinator routing decisions
- `dynamo_response.json` - Dynamo expert output format
- `inductor_response.json` - Inductor expert output format
- `aot_response.json` - AOT debugger output format

All schemas validated with `python -m json.tool`

## Installation

### Container Environment

**Persistent:** `/workspaces/` (code + PyTorch indices)
**Ephemeral:** `~/.claude/settings.json`, pip packages

### Setup Script

```bash
cd /workspaces/pytorch-devcontainers/torch-compile-ai
./setup.sh
```

The `setup.sh` script:
- Installs pip packages (fast, ~30s)
- Stores indices in `/workspaces/ai-tooling/.acp-indices/` (persists)
- Recreates `~/.claude/settings.json` on each startup
- Skills auto-discoverable via Claude Code skill system

### Manual Installation

```bash
# Install in editable mode (required for tests)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

**Requirements:**
- Python 3.10+
- mcp>=1.0.0
- pytest, ruff, pyright (dev only)

## Testing

Skills and multi-agent scenarios tested:

```bash
# All tests
pytest tests/ -v

# Multi-agent scenarios
pytest tests/multi-agent/ -v
```

## Multi-Agent System

### Coordinator Agent
**Location:** `agent-plugins/coordinator-agent/`
**Skills:** compile-overview
**Role:** Routes tasks to specialists, synthesizes results, presents unified guidance

**Capabilities:**
- Task analysis and routing
- User confirmation workflow
- Multi-specialist coordination
- Structured JSON output

### Specialist Agents

**Dynamo Debugger Agent**
**Location:** `agent-plugins/dynamo-debugger-agent/`
**Skills:** pytorch-dynamo, compile-trace-dynamo
**Specialization:** VariableTracker system, bytecode tracing, guard generation, graph breaks

**Inductor Debugger Agent**
**Location:** `agent-plugins/inductor-debugger-agent/`
**Skills:** pytorch-inductor, compile-trace-inductor
**Specialization:** Lowering registration, IR nodes, Triton codegen, fusion patterns

**AOT Debugger Agent**
**Location:** `agent-plugins/aot-debugger-agent/`
**Skills:** compile-trace-aot
**Specialization:** Functionalization, decompositions, joint graphs, partitioning

**Bisector Agent**
**Location:** `agent-plugins/bisector-agent/`
**Skills:** compile-bisect
**Specialization:** Automated failure isolation, backend/subsystem binary search

### MCP Server
- **steering**: API documentation and semantic search over PyTorch modules

### Tool Allowlists

**Coordinator:** Read, Bash, MCP steering
**Experts** (dynamo, inductor, aot): Read, Bash, MCP steering
**Bisector:** Read, Bash (needs to run bisector command)

## Skill Sync & Validation

### Sync Script
`scripts/sync-agent-skills.py` syncs skills from vertical-plugins/ to agent-plugins/

**Mappings:**
```python
{
    "coordinator-agent": ["compile-overview"],
    "dynamo-debugger-agent": ["pytorch-dynamo", "compile-trace-dynamo"],
    "inductor-debugger-agent": ["pytorch-inductor", "compile-trace-inductor"],
    "aot-debugger-agent": ["compile-trace-aot"],
    "bisector-agent": ["compile-bisect"],
}
```

**Usage:**
```bash
python scripts/sync-agent-skills.py
```

### Validation Script
`scripts/validate-skills.py` validates YAML frontmatter, cross-references, and circular dependencies

**Usage:**
```bash
python scripts/validate-skills.py
```

## Benefits

### Context Efficiency
- Vertical organization reduces context size by loading only relevant skills
- Coordinator loads minimal routing logic without domain skills
- Specialists load only their domain-specific knowledge
- Agents can run in parallel when tasks are independent

### Design Philosophy
- Simple, focused parsers aligned with IR levels
- Agents have clear responsibility boundaries
- Skills are composable and reusable
- Automation reduces manual sync burden

## Development Workflow

### Skill Development

1. **Edit source skills** in `vertical-plugins/` or `coordinator/`
2. **Run sync script** to update agent bundles:
   ```bash
   python scripts/sync-agent-skills.py
   ```
3. **Validate** frontmatter and cross-references:
   ```bash
   python scripts/validate-skills.py
   ```

### Agent Development

1. **Edit agent definitions** in `agent-plugins/*/agents/`
2. **Update schemas** in `schemas/` if changing output format
3. **Validate schemas**:
   ```bash
   python -m json.tool schemas/<schema>.json
   ```

### Test Development

1. **Write test** in `tests/multi-agent/`
2. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

## Reorganization History

### Phase 1: Vertical Organization (Complete ✅)
**Date:** 2026-05-19
**Goal:** Reorganize from horizontal (type-based) to vertical (domain-based)

**Changes:**
- Created `vertical-plugins/` with stage-specific directories
- Moved skills from `.claude/skills/` to verticals
- Created symlinks in `.claude/skills/` for backward compatibility
- Added README.md to each vertical explaining purpose

**See:** `/workspaces/pytorch-devcontainers/specs/agentic-workflow/REORGANIZATION-SUMMARY.md`

### Phase 2: Agent Formalization (Complete ✅)
**Date:** 2026-05-19
**Goal:** Add structured agent definitions and automation

**Phase 2A - Manifests & Schemas:**
- Created `agent-plugins/` with 5-section agent definitions
- Created `managed-agent-cookbooks/` with agent.yaml manifests
- Created `schemas/` with JSON schemas for structured output
- Added tool allowlists and callable_agents

**Phase 2B - Automation:**
- Created `scripts/sync-agent-skills.py` for skill syncing
- Created `scripts/validate-skills.py` for validation
- All 6 skills synced to agent bundles
- All schemas validated

**See:** `/workspaces/pytorch-devcontainers/specs/agentic-workflow/PHASE2-IMPLEMENTATION-SUMMARY.md`

### Phase 3: Future Enhancements (Deferred ⏸️)

**Not Implemented:**
- Orchestration layer (scripts/orchestrate.py)
- Headless deployment to Claude API
- Advanced skill composition
- User-teachable workflows
- Performance telemetry

**See:** `/workspaces/pytorch-devcontainers/specs/agentic-workflow/phase-3-plan.md`

## Code Quality

- ✅ **Type hints**: Modern Python 3.10+ annotations
- ✅ **Google docstrings**: Args/Returns documented
- ✅ **TDD**: Parser tests with realistic torch.compile output
- ✅ **Linted**: ruff + pyright compliant
- ✅ **Validated**: All skill frontmatter and schemas validated
