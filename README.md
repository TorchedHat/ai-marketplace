# Multi-Agent Development System for torch.compile

A specialized multi-agent architecture for PyTorch compiler development, combining debug log parsing with API documentation lookup.

## Status: ✅ Complete

- **torch-compile-ai MCP:** 9 parsers aligned with IR levels, 16 tests passing
- **Steering MCP:** Indexed (dynamo: 1,208 funcs, inductor: 2,457 funcs)
- **Multi-Agent Prompts:** Coordinator + 3 specialists (tracing + 2 experts)
- **Configuration:** Automated setup script for container environments

## Quick Start

```bash
# 1. Run setup (one-time per container startup)
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh

# 2. Start Claude Code and load coordinator
# In Claude: "Load the coordinator prompt from .../prompts/coordinator-concise.md"

# 3. Describe your PyTorch issue naturally
# The coordinator will suggest tools/specialists and work with you interactively
```

**First run:** ~10-15 minutes (includes PyTorch indexing)  
**Subsequent runs:** ~30 seconds

**See [GETTING_STARTED.md](GETTING_STARTED.md) for interactive examples and workflow.**

## What This Provides

### 1. torch-compile-ai MCP (9 Parsers)

Parse torch.compile debug output aligned with IR levels:

**Dynamo Stage (3 parsers)**
- Graph breaks, FX graphs, pre-grad passes

**AOT Autograd Stage (3 parsers)**
- Joint graph, partitioned graphs, post-grad passes

**Inductor Stage (3 parsers)**
- Fusion decisions, IR post-fusion (LoopBody), output code (Triton/C++)

### 2. Steering MCP

API documentation and code navigation:
- **torch._dynamo:** 1,208 functions, 647 classes
- **torch._inductor:** 2,457 functions, 1,122 classes

### 3. Multi-Agent Prompts

Coordinator + specialist architecture:
- **Coordinator** (126 lines): Routes tasks, synthesizes results, confirms with user
- **Dynamo Expert** (95 lines): VariableTracker, guards, graph breaks
- **Inductor Expert** (131 lines): Lowerings, IR nodes, Triton, fusion

**Context efficiency:** 60-70% reduction vs loading all skills

## Architecture

```
User Query
    ↓
Coordinator (routing + synthesis)
    ↓
    ├─→ steering-mcp (API lookups)
    ├─→ torch-compile-ai (parse logs)
    ├─→ dynamo-expert (Dynamo analysis)
    └─→ inductor-expert (Inductor analysis)
    ↓
Synthesized Response
```

## Container Environment

**Persistent:** `/workspaces/` (code + PyTorch indices)  
**Ephemeral:** `~/.claude/settings.json`, pip packages

The `setup.sh` script:
- Stores indices in `/workspaces/ai-tooling/.acp-indices/` (persists)
- Recreates `~/.claude/settings.json` on each startup
- Re-installs pip packages (fast, ~30s)

## Usage

### 1. Start System

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

### 2. Load Coordinator

In Claude Code:
```
Load prompts/coordinator-concise.md
```

### 3. Ask Questions

**Example 1: Debug Graph Break**
```
Parse guards from torch_compile_debug/run_*/torchdynamo/debug.log 
and explain why we're getting graph breaks on tensor.item()
```

**Example 2: API Lookup**
```
What are the parameters for Pointwise.__init__?
```

**Example 3: Performance Analysis**
```
Why isn't my reduction fusing with the pointwise op before it?
```

See `tests/multi-agent/test_scenarios.md` for 5 complete test scenarios.

## Files

```
torch-compile-ai/
├── setup.sh                          # Automated setup (run on container startup)
├── server.py                         # MCP server entry point
├── analyzers/                          # 9 parser implementations (3 modules)
│   ├── dynamo_guards.py
│   ├── dynamo_graph.py
│   ├── aot_forward_graph.py
│   ├── inductor_output_code.py
│   └── ...
├── prompts/                          # Multi-agent prompts
│   ├── coordinator-concise.md        # Main routing agent
│   ├── dynamo-expert-concise.md      # Dynamo specialist
│   └── inductor-expert-concise.md    # Inductor specialist
├── tests/
│   ├── parsers/                        # 16 unit tests (mirrors source)
│   └── multi-agent/                  # 5 end-to-end scenarios
└── docs/
    ├── INSTALLATION.md               # Detailed setup guide
    ├── CURRENT_STATUS.md             # Implementation status
    └── ...
```

## Performance

### Context Efficiency
- MCP-only query: ~10KB (87% reduction vs all skills)
- Single specialist: ~60KB (60% reduction)
- Multi-specialist: ~110KB (27% reduction)

### Query Latency
- Steering API lookup: <1s
- Debug log parse: 1-3s
- Specialist analysis: 5-10s

## Development

### Run Tests

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
python -m pytest tests/analyzers/ -v
```

### Re-index PyTorch

After pulling PyTorch updates:

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh  # Re-runs indexing if needed
```

### Update Prompts

Edit `prompts/*.md` to adjust:
- Routing rules (coordinator)
- Output format (specialists)
- Tool selection logic

## Documentation

- **Installation:** `docs/INSTALLATION.md` - Detailed setup guide
- **Architecture:** `REPO_ARCH.md` - Repository structure and design
- **Code Guidelines:** `CLAUDE.md` - Code style and testing requirements
- **Getting Started:** `GETTING_STARTED.md` - Interactive workflow examples
- **Test Scenarios:** `tests/multi-agent/test_scenarios.md` - 5 end-to-end tests

## Troubleshooting

### MCP Server Not Starting

```bash
# Test manually
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
python server.py

# Check settings
cat ~/.claude/settings.json
```

### Steering Returns No Results

```bash
# Verify indices exist
ls /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/

# Re-run setup
./setup.sh
```

### Tests Failing

```bash
# Run with verbose output
python -m pytest tests/analyzers/ -v -s
```

## Code Quality

- ✅ **Type hints**: Modern Python 3.10+ annotations
- ✅ **Google docstrings**: Args/Returns documented
- ✅ **TDD**: 16 tests, all passing
- ✅ **Linted**: ruff + pyright compliant

## License

Part of the PyTorch devcontainer tooling.
