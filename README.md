# Multi-Agent Development System for torch.compile

Anthropic-pattern multi-agent debugging system for PyTorch compiler development.

**Architecture:**
- **Claude Code Plugin** - Self-contained plugin with all torch.compile debugging capabilities
- **Skills** - 8 skills organized by compilation stage (Dynamo, AOT, Inductor)
- **Agents** - 5 specialized sub-agents (coordinator, dynamo-expert, aot-expert, inductor-expert, bisector)
- **MCP Server** - Steering (PyTorch API documentation and semantic search) bundled in plugin
- **Source Organization** - Vertical plugins directory structure preserved for development

**Features:**
- 🔍 **Intelligent Routing** - Bisect-first workflow routes to stage-specific skills
- 📚 **Steering Integration** - Semantic search over PyTorch Dynamo/Inductor APIs
- 🔄 **Skill Composition** - Implementation + tracing skills per stage
- 🛠️ **Bisector Integration** - Automated failure isolation

## Quick Start

### Install and Use

```bash
# Clone the plugin
git clone https://github.com/morrison-turnansky/torch-compile-ai.git
cd torch-compile-ai

# Start Claude Code with the plugin
claude --plugin-dir .

# Skills are now available
/compile-overview
/pytorch-dynamo
/compile-trace-inductor

# Or just ask naturally:
#   "Debug this graph break: def fn(x): return x[x.item()]"
#   "Why isn't my reduction fusing?"
#   "Show me the Triton kernel for: def fn(x): return x.relu()"
```

**Auto-setup:** Plugin automatically installs dependencies and indexes PyTorch on first use via SessionStart hook.

See **[INSTALL.md](INSTALL.md)** for detailed installation instructions and troubleshooting.

## Installation

### Quick Install

```bash
claude plugin install https://github.com/pytorch/pytorch-devcontainers/tree/main/torch-compile-ai
```

On first use, the plugin automatically:
- ✅ Installs `acp-steering-mcp` package
- ✅ Indexes PyTorch modules (dynamo, inductor, functorch)
- ✅ Configures MCP server for API documentation

### Manual Install (Development)

```bash
./setup.sh
```

For detailed installation instructions, troubleshooting, and configuration options, see **[INSTALL.md](INSTALL.md)**.

## Usage

### Skill Discovery

All skills are **auto-discoverable** via Claude Code's plugin system. Skills are namespaced under `torch-compile-ai:`:

```bash
# Meta-skills (workflow guidance)
/torch-compile-ai:compile-overview         # Pipeline overview & bisect-first workflow
/torch-compile-ai:compile-bisect          # Automated failure isolation

# Tracing skills (user-level debugging)
/torch-compile-ai:compile-trace-dynamo    # Debug Dynamo: graph breaks, FX graphs, TORCH_LOGS
/torch-compile-ai:compile-trace-aot       # Debug AOT: functionalization, decompositions, partitioning
/torch-compile-ai:compile-trace-inductor  # Debug Inductor: fusion, scheduling, codegen

# Implementation skills (PyTorch contributors)
/torch-compile-ai:pytorch-dynamo          # Dynamo internals: VariableTracker, guards, C++ runtime
/torch-compile-ai:pytorch-aot             # AOT/Functorch internals: vmap, functionalization, partitioning
/torch-compile-ai:pytorch-inductor        # Inductor internals: lowerings, scheduling, Triton codegen

# Or just ask naturally - Claude loads skills automatically based on context
```

### Example Queries

**User-level debugging (uses tracing skills):**
```
Why does this graph break? def fn(x): return x[x.item()]
Show me the fusion decisions for this model
Parse these TORCH_LOGS and explain what happened
```

**API lookup (uses steering MCP):**
```
What are the parameters for Pointwise.__init__?
How do I use SymInt in C++ code?
Show me the signature for torch._dynamo.variables.VariableTracker.call_method
```

Skills guide Claude to read raw TORCH_LOGS output directly rather than using intermediate parsers.

**Implementation work (uses pytorch-* skills):**
```
How do I implement a new VariableTracker type?
How do I add functionalization support for my op?
Where do I add a lowering for my custom op?
Explain the C++ guard evaluation tree
Show me how vmap batching works internally
```

See `tests/multi-agent/test_scenarios.md` for complete examples.

## Structure

```
torch-compile-ai/
├── .claude-plugin/                # Plugin metadata
│   └── plugin.json               # Plugin manifest (name, version, MCP config)
│
├── skills/                        # All skills (plugin-discoverable)
│   ├── compile-overview/         # Meta: Pipeline overview & bisect-first workflow
│   ├── compile-bisect/           # Meta: Failure isolation
│   ├── pytorch-dynamo/           # Implementation: VariableTracker, guards, C++ runtime
│   ├── compile-trace-dynamo/     # Tracing: TORCH_LOGS, graph breaks, FX graphs
│   ├── pytorch-aot/              # Implementation: vmap, functorch, AOT internals
│   ├── compile-trace-aot/        # Tracing: functionalization, partitioning, TORCH_LOGS
│   ├── pytorch-inductor/         # Implementation: lowerings, scheduler, Triton
│   └── compile-trace-inductor/   # Tracing: fusion, scheduling, kernels
│
├── agents/                        # Sub-agent definitions (plugin-discoverable)
│   ├── coordinator.md            # Workflow coordinator
│   ├── dynamo-expert.md          # Dynamo debugging specialist
│   ├── aot-expert.md             # AOT debugging specialist
│   ├── inductor-expert.md        # Inductor debugging specialist
│   └── bisector.md               # Bisection specialist
│
├── vertical-plugins/              # Source organization (development)
│   ├── dynamo-debugger/
│   ├── aot-debugger/
│   ├── inductor-debugger/
│   └── bisector/
│
├── agent-plugins/                 # Agent definitions (future)
├── schemas/                       # JSON schemas for structured output
├── scripts/                       # Automation
└── tests/                         # pytest tests
```

**Skill Organization:**
- **Tracing skills** (`compile-trace-*`): User-level debugging, TORCH_LOGS interpretation
- **Implementation skills** (`pytorch-*`): Contributor-level, internals and architecture
- **Meta skills** (`compile-overview`, `compile-bisect`): Workflow guidance

**Plugin Structure:**
- Skills at `skills/` directory are auto-discovered by Claude Code
- Agents at `agents/` directory provide specialized sub-agents
- MCP server configured in `.claude-plugin/plugin.json` mcpServers section
- Plugin manifest at `.claude-plugin/plugin.json`

See **[REPO_ARCH.md](REPO_ARCH.md)** for detailed architecture.

## MCP Server

### Steering (Semantic API Search)

Indexes PyTorch Dynamo and Inductor modules for semantic API documentation:

**Query steering guidance:**
```
How do I handle pytree operations in VariableTracker?
```

**Query API docs:**
```
What's the signature for torch._inductor.ir.Pointwise.__init__?
```

**Indexed modules:**
- `torch/_dynamo/` - Dynamo internals (bytecode capture, VariableTracker, guards)
- `torch/_functorch/` - Functorch/AOT internals (vmap, functionalization, partitioning)
- `torch/_inductor/` - Inductor internals (lowerings, fusion, codegen)

Steering provides context about when/how to use APIs, common patterns, and architectural guidance.

## Development

**Run tests:**
```bash
pytest tests/ -v
```

**Create/update skill symlinks (Phase 2):**
```bash
python scripts/sync-agent-skills.py
```

This creates symlinks from `agent-plugins/*/skills/` to `vertical-plugins/*/skills/`, ensuring a single source of truth.

**Update agent prompts:**
Edit `agent-plugins/*/agents/*.md` to adjust agent behavior

**Update skills:**
Edit `vertical-plugins/*/skills/*/` source files directly (symlinks ensure changes are visible everywhere)

**Re-index after PyTorch updates:**
```bash
./setup.sh
```

## Documentation

- **[REPO_ARCH.md](REPO_ARCH.md)** - Architecture, structure, and reorganization history
- **[CLAUDE.md](CLAUDE.md)** - Code guidelines and testing
- **[/specs/agentic-workflow/](../../specs/agentic-workflow/)** - Multi-agent workflow implementation
  - `REORGANIZATION-SUMMARY.md` - Phase 1: Vertical organization (✅ complete)
  - `PHASE2-IMPLEMENTATION-SUMMARY.md` - Phase 2: Agent formalization (✅ complete)
  - `phase-3-plan.md` - Phase 3: Future enhancements (⏸️ deferred)

## Troubleshooting

**Steering returns no results:**
```bash
# Check if indices exist
ls ~/.acp/repos/*/

# Re-index PyTorch
./setup.sh
```

**Skills not loading:**
```bash
# Verify plugin is configured
cat /workspaces/pytorch-devcontainers/.claude/settings.json

# Verify plugin structure
ls -la /workspaces/pytorch-devcontainers/torch-compile-ai/.claude-plugin/
ls -la /workspaces/pytorch-devcontainers/torch-compile-ai/skills/

# Check if skills are in plugin
find /workspaces/pytorch-devcontainers/torch-compile-ai/skills -name "SKILL.md"
```

**Plugin not recognized:**
```bash
# Re-run setup to regenerate settings
./setup.sh

# Manually verify plugin.json is valid JSON
cat /workspaces/pytorch-devcontainers/torch-compile-ai/.claude-plugin/plugin.json | python -m json.tool
```

## License

Part of the PyTorch devcontainer tooling.
