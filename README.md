# Multi-Agent Development System for torch.compile

Anthropic-pattern multi-agent debugging system for PyTorch compiler development.

**Architecture:**
- **Vertical Plugins** - Skills organized by compilation stage (Dynamo, AOT, Inductor) - **single source of truth**
- **Agent Plugins** - Structured agents with manifests (coordinator, experts, bisector) - skills are symlinked from vertical plugins
- **MCP Servers** - Debug output parsers + steering (PyTorch API documentation)
- **Skill Symlinks** - Agent plugin skills symlinked from vertical sources (single source of truth)

**Features:**
- 🔍 **Intelligent Routing** - Bisect-first workflow routes to stage-specific skills
- 📚 **Steering Integration** - Semantic search over PyTorch Dynamo/Inductor APIs
- 🔄 **Skill Composition** - Implementation + tracing skills per stage
- 🛠️ **Bisector Integration** - Automated failure isolation
- 📋 **Debug Parsers** - 9 MCP tools for parsing TORCH_LOGS output

## Quick Start

```bash
# 1. Run setup (one-time per container startup)
cd /workspaces/pytorch-devcontainers/torch-compile-ai
./setup.sh

# 2. Skills are auto-discoverable - just ask!
# Examples:
#   "Debug this graph break: def fn(x): return x[x.item()]"
#   "Why isn't my reduction fusing?"
#   "Show me the Triton kernel for: def fn(x): return x.relu()"
#   "How do I implement a new VariableTracker?"
```

**Note**: Skills are discoverable through the Claude Code skill system. Setup includes PyTorch API indexing for steering-based semantic search.

## Installation

```bash
./setup.sh
```

This script:
- Installs pip packages (`acp-steering-mcp`, `torch-compile-ai`)
- **Indexes PyTorch** (dynamo + inductor) for steering API documentation
- Configures MCP servers in `~/.claude/settings.json`:
  - `debug-tracer` - 9 parsers for TORCH_LOGS output
  - `steering` - Semantic search over PyTorch APIs

## Usage

### Skill Discovery

All skills are **auto-discoverable** via Claude Code's skill system. Use slash commands or ask natural questions:

```bash
# Meta-skills (workflow guidance)
/compile-overview         # Pipeline overview & bisect-first workflow
/compile-bisect          # Automated failure isolation

# Tracing skills (user-level debugging)
/compile-trace-dynamo    # Debug Dynamo: graph breaks, FX graphs, TORCH_LOGS
/compile-trace-aot       # Debug AOT: functionalization, decompositions, partitioning
/compile-trace-inductor  # Debug Inductor: fusion, scheduling, codegen

# Implementation skills (PyTorch contributors)
/pytorch-dynamo          # Dynamo internals: VariableTracker, guards, C++ runtime
/pytorch-aot             # AOT/Functorch internals: vmap, functionalization, partitioning
/pytorch-inductor        # Inductor internals: lowerings, scheduling, Triton codegen

# Or just ask naturally - skills load automatically based on context
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
├── vertical-plugins/              # Skills organized by stage (source of truth)
│   ├── dynamo-debugger/
│   │   ├── skills/
│   │   │   ├── compile-trace-dynamo/  # User-level: TORCH_LOGS, graph breaks, FX graphs
│   │   │   └── pytorch-dynamo/        # Implementation: VariableTracker, guards, C++ runtime
│   │   ├── prompts/dynamo-expert.md
│   │   └── README.md
│   ├── aot-debugger/
│   │   ├── skills/
│   │   │   ├── compile-trace-aot/     # User-level: functionalization, partitioning, TORCH_LOGS
│   │   │   └── pytorch-aot/           # Implementation: vmap, functorch, AOT internals
│   │   ├── prompts/aot-expert.md
│   │   └── README.md
│   ├── inductor-debugger/
│   │   ├── skills/
│   │   │   ├── compile-trace-inductor/  # User-level: fusion, scheduling, kernels
│   │   │   └── pytorch-inductor/        # Implementation: lowerings, scheduler, Triton
│   │   ├── prompts/inductor-expert.md
│   │   └── README.md
│   └── bisector/skills/compile-bisect/         # Failure isolation
│
├── coordinator/skills/compile-overview/  # Bisect-first workflow & pipeline overview
│
├── agent-plugins/                 # Agent definitions (Phase 2)
│   ├── coordinator-agent/
│   ├── dynamo-debugger-agent/
│   ├── inductor-debugger-agent/
│   ├── aot-debugger-agent/
│   └── bisector-agent/
│
├── schemas/                       # JSON schemas for structured output (Phase 2)
│   ├── handoff_request.json
│   ├── dynamo_response.json
│   └── inductor_response.json
│
├── scripts/                       # Automation (Phase 2)
│   ├── sync-agent-skills.py       # Create symlinks from vertical-plugins to agent bundles
│   └── validate-skills.py         # Lint and validate
│
├── analyzers/                     # 9 MCP debug parsers (Python)
├── server.py                      # MCP server: debug-tracer + steering
└── tests/                         # pytest tests
```

**Skill Organization:**
- **Tracing skills** (`compile-trace-*`): User-level debugging, TORCH_LOGS interpretation
- **Implementation skills** (`pytorch-*`): Contributor-level, internals and architecture
- **Meta skills** (`compile-overview`, `compile-bisect`): Workflow guidance

See **[REPO_ARCH.md](REPO_ARCH.md)** for detailed architecture.

## MCP Servers

### Debug Tracer (9 Parsers)

Parses TORCH_LOGS output and debug artifacts:

| Parser | Input | Output |
|--------|-------|--------|
| `parse_fx_graph` | `fx_graph_readable.py` | FX graph structure |
| `parse_graph_breaks` | `TORCH_LOGS=graph_breaks` stdout | Break locations & reasons |
| `parse_aot_joint_graph` | AOT joint graph file | Forward+backward graph |
| `parse_aot_graphs` | AOT forward/backward files | Partitioned graphs |
| `parse_post_grad_passes` | `TORCH_LOGS=post_grad_graphs` | Post-grad transformations |
| `parse_fusion_decisions` | `TORCH_LOGS=fusion,schedule` | Fusion decisions |
| `parse_ir_post_fusion` | `ir_post_fusion_*.txt` | LoopBody IR |
| `parse_output_code` | `output_code.py` | Generated kernels |

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
pytest tests/analyzers/ -v
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

**MCP server not starting:**
```bash
python server.py  # Test manually
cat ~/.claude/settings.json  # Verify MCP config includes debug-tracer and steering
```

**Steering returns no results:**
```bash
# Check if indices exist
ls ~/.acp-indices/pytorch-*/

# Re-index PyTorch
./setup.sh
```

**Debug parsers returning errors:**
```bash
# Test individual parser
python -c "from analyzers.dynamo_parsers import parse_graph_breaks; print(parse_graph_breaks('test'))"

# Run parser tests
pytest tests/analyzers/ -v -s
```

**Skills not loading:**
```bash
# Verify skills are linked correctly
ls -la /workspaces/pytorch-devcontainers/.claude/skills/

# Check skill rules
cat /workspaces/pytorch-devcontainers/.claude/skills/skill-rules.json
```

## License

Part of the PyTorch devcontainer tooling.
