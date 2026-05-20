# Multi-Agent Development System for torch.compile

Anthropic-pattern multi-agent debugging system for PyTorch compiler development.

**Architecture:**
- **Vertical Plugins** - Skills organized by compilation stage (Dynamo, AOT, Inductor)
- **Agent Plugins** - Structured agents with manifests (coordinator, experts, bisector)
- **MCP Servers** - 9 parsers for debug output + API documentation
- **Skill Sync** - Automated sync from vertical sources to agent bundles

**Features:**
- 🔍 **Intelligent Routing** - Coordinator delegates to stage-specific experts
- 📋 **Structured Output** - JSON schemas for agent responses
- 🔄 **Skill Composition** - Agents bundle relevant skills automatically
- 🛠️ **Bisector Integration** - Automated failure isolation

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
```

**First run:** ~10-15 minutes (includes PyTorch indexing)
**Subsequent:** ~30 seconds

**Note**: Skills are always discoverable through the Claude Code skill system. No need to load prompts manually - the system routes automatically.

## Installation

```bash
./setup.sh
```

This script:
- Installs pip packages (`acp-steering-mcp`, `torch-compile-ai`)
- Indexes PyTorch (dynamo + inductor) on first run
- Configures MCP servers in `~/.claude/settings.json`

## Usage

### Skill Discovery

All skills are **auto-discoverable** via Claude Code's skill system. Use slash commands or ask natural questions:

```bash
# Slash commands
/compile-overview         # Pipeline overview
/compile-bisect          # Automated failure isolation
/pytorch-dynamo          # Dynamo implementation knowledge
/pytorch-inductor        # Inductor implementation knowledge

# Or just ask naturally - skills load automatically
"Why does len() cause a graph break?"
"Show me the fusion decisions for this model"
"How do I trace through AOT Autograd?"
```

### Example Queries

**Debug graph break:**
```
Why does this graph break? def fn(x): return x[x.item()]
```

**API lookup:**
```
What are the parameters for Pointwise.__init__?
```

**Performance:**
```
Parse fusion logs and explain why ops aren't fusing
```

See `tests/multi-agent/test_scenarios.md` for complete examples.

## Structure

```
torch-compile-ai/
├── vertical-plugins/              # Skills organized by stage (source of truth)
│   ├── dynamo-debugger/
│   │   ├── skills/compile-trace-dynamo/, pytorch-dynamo/
│   │   ├── prompts/dynamo-expert.md
│   │   └── README.md
│   ├── inductor-debugger/
│   │   ├── skills/compile-trace-inductor/, pytorch-inductor/
│   │   ├── prompts/inductor-expert.md
│   │   └── README.md
│   ├── aot-debugger/skills/compile-trace-aot/
│   └── bisector/skills/compile-bisect/
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
│   ├── sync-agent-skills.py       # Sync skills to agent bundles
│   └── validate-skills.py         # Lint and validate
│
├── analyzers/                     # 9 MCP parsers (Python)
├── server.py                      # MCP server entry point
└── tests/                         # pytest tests
```

See **[REPO_ARCH.md](REPO_ARCH.md)** for detailed architecture.

## Development

**Run tests:**
```bash
pytest tests/analyzers/ -v
```

**Sync skills (Phase 2):**
```bash
python scripts/sync-agent-skills.py
```

**Update agent prompts:**
Edit `agent-plugins/*/agents/*.md` to adjust agent behavior

**Update skills:**
Edit `vertical-plugins/*/skills/*/` source files, then run sync script

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
cat ~/.claude/settings.json  # Verify config
```

**Steering returns no results:**
```bash
ls /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/  # Check indices
./setup.sh  # Re-run setup
```

**Tests failing:**
```bash
pytest tests/analyzers/ -v -s
```

## License

Part of the PyTorch devcontainer tooling.
