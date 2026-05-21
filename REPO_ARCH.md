# Repository Architecture

## Overview

Multi-agent development system for PyTorch torch.compile, organized using Anthropic's vertical plugin pattern. Provides MCP tools, stage-specific skills, and structured agent definitions for debugging across all compilation stages (Dynamo, AOT Autograd, Inductor).

**Design Philosophy**:
- **Vertical Organization** - Domain-based structure (not type-based)
- **Agent Formalization** - Structured definitions with JSON schemas
- **Simple Parsers** - Aligned with torch.compile IR levels
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
├── analyzers/                  # 9 MCP parser implementations
│   ├── __init__.py
│   ├── dynamo_parsers.py      # Dynamo stage (3 parsers)
│   ├── aot_parsers.py         # AOT stage (3 parsers)
│   └── inductor_parsers.py    # Inductor stage (3 parsers)
│
├── server.py                   # MCP server entry point
├── tests/                      # Tests
│   ├── analyzers/             # Parser unit tests
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
    ├─→ torch-compile-ai (parse stdout/files)
    │   └─→ 9 parsers aligned with IR levels
    ├─→ bisector-agent (automated failure isolation)
    ├─→ dynamo-debugger-agent (Dynamo analysis)
    ├─→ inductor-debugger-agent (Inductor analysis)
    └─→ aot-debugger-agent (AOT Autograd analysis)
    ↓
Synthesized Response (structured JSON)
```

### 9 MCP Tools (Aligned with IR Levels)

**Dynamo Stage:**
1. `parse_graph_breaks(log_content)` - TORCH_LOGS="graph_breaks" stdout
2. `parse_fx_graph(graph_content)` - fx_graph_readable.py file
3. `parse_pre_grad_passes(before, after)` - FX graph before/after files

**AOT Autograd Stage:**
4. `parse_aot_joint_graph(graph_content)` - joint graph file
5. `parse_aot_graphs(forward, backward)` - forward/backward graph files
6. `parse_post_grad_passes(log_content)` - post-grad optimization logs

**Inductor Stage:**
7. `parse_fusion_decisions(log_content)` - TORCH_LOGS="fusion,schedule" stdout
8. `parse_ir_post_fusion(ir_content)` - ir_post_fusion_*.txt file
9. `parse_output_code(code_content)` - output_code.py file

### Module Dependencies

```
server.py → all analyzer modules (imports 9 parsers)
tests/ → analyzers (imports for testing)
analyzers/__init__.py → all analyzer modules (re-exports)

analyzers/
├── dynamo_parsers.py (no internal dependencies)
├── aot_parsers.py (no internal dependencies)
└── inductor_parsers.py (no internal dependencies)
```

**External Dependencies** (runtime):
- `mcp>=1.0.0` - MCP protocol types only
- Standard library: `re`, `collections`, `asyncio`

**External Dependencies** (dev):
- `pytest>=8.0.0`, `pytest-asyncio>=0.23.0` - Testing

## Key Components

### Parsers by Stage

**Dynamo Stage (FX Graph Generation)** - `analyzers/dynamo_parsers.py`
- `parse_graph_breaks` - Parse TORCH_LOGS="graph_breaks" stdout, categorize by type
- `parse_fx_graph` - Parse fx_graph_readable.py, count operations, identify patterns
- `parse_pre_grad_passes` - Compare before/after FX graphs, detect optimizations

**AOT Autograd Stage (Training Mode)** - `analyzers/aot_parsers.py`
- `parse_aot_joint_graph` - Parse joint forward+backward graph file
- `parse_aot_graphs` - Parse separate forward/backward graph files
- `parse_post_grad_passes` - Parse post-grad optimization logs

**Inductor Stage (Lowering and Codegen)** - `analyzers/inductor_parsers.py`
- `parse_fusion_decisions` - Parse TORCH_LOGS="fusion,schedule" stdout
- `parse_ir_post_fusion` - Parse ir_post_fusion_*.txt (LoopBody IR)
- `parse_output_code` - Parse output_code.py (Triton/C++ kernels)

### Data Flow

**Two types of torch.compile output:**

**1. Stdout logs (ephemeral):**
- TORCH_LOGS="graph_breaks" → parse_graph_breaks
- TORCH_LOGS="fusion,schedule" → parse_fusion_decisions
- Only available during code execution
- Tracing-agent must parse immediately before returning

**2. Debug files (persistent):**
- fx_graph_readable.py → parse_fx_graph
- output_code.py → parse_output_code
- Saved to torch_compile_debug/
- Can be read and parsed later

**Tracing-Agent Workflow:**
```python
# 1. Run code with TORCH_LOGS
stdout = bash("TORCH_LOGS='graph_breaks,fusion,output_code' python temp.py")

# 2. Parse stdout immediately (ephemeral)
findings = {}
if "graph_breaks" in flags:
    findings["graph_breaks"] = parse_graph_breaks(stdout)

# 3. Find debug directory
debug_dir = find_latest("torch_compile_debug/run_*/")

# 4. Read and parse files (persistent)
if "output_code" in flags:
    code = read(f"{debug_dir}/output_code.py")
    findings["kernel"] = parse_output_code(code)

# 5. Return structured findings
return {"parsed_findings": findings, "debug_dir": debug_dir}
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

### Running the Server

```bash
# Start MCP server
python server.py
```

### Tool Invocation

MCP clients call tools with JSON parameters:

```json
{
  "name": "parse_graph_breaks",
  "arguments": {
    "log_content": "Graph break: print(...)\n  Reason: ..."
  }
}
```

Server routes to `dynamo_parsers.parse_graph_breaks(log_content)` and returns formatted string.

### Claude Code Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "torch-compile-ai": {
      "command": "python",
      "args": ["/workspaces/pytorch-devcontainers/torch-compile-ai/server.py"],
      "cwd": "/workspaces/pytorch-devcontainers/torch-compile-ai",
      "env": {
        "PYTHONPATH": "/workspaces/pytorch-devcontainers/torch-compile-ai"
      }
    }
  }
}
```

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

All parsers tested with real or realistic torch.compile output:

```bash
# All tests
pytest tests/analyzers/ -v

# Specific stage
pytest tests/analyzers/test_dynamo_parsers.py -v

# Specific test
pytest tests/analyzers/test_inductor_parsers.py::TestParseOutputCode::test_triton_kernel -v
```

**Test Requirements:**
- Use realistic stdout/file content
- Test both success and error cases
- Verify output format and key information

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

### MCP Servers
- **torch-compile-ai**: 9 parsers (this repository)
- **steering-mcp**: API documentation and code navigation

### Tool Allowlists

**Coordinator:** Read, Bash, MCP tools (steering + debug-tracer)
**Experts** (dynamo, inductor, aot): Read, MCP steering only (no Write/Edit)
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

### Parser Development

1. **Write test** in `tests/analyzers/`
2. **Implement parser** in `analyzers/`
3. **Run tests**:
   ```bash
   pytest tests/analyzers/ -v
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
