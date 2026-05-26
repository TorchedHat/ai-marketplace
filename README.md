# torch-compile-ai

Multi-agent debugging system for PyTorch torch.compile development. Provides stage-specific skills, specialized agents, and semantic API search for Dynamo, AOT Autograd, and Inductor.

## What This Repo Provides

**8 Debugging Skills:**
- Meta skills: `compile-overview`, `compile-bisect`
- Tracing skills: `compile-trace-dynamo`, `compile-trace-aot`, `compile-trace-inductor`
- Implementation skills: `pytorch-dynamo`, `pytorch-aot`, `pytorch-inductor`

**5 Specialized Agents:**
- Coordinator, dynamo-expert, aot-expert, inductor-expert, bisector

**MCP Server Integration:**
- Semantic search over PyTorch Dynamo, Inductor, and Functorch APIs
- Auto-indexed on first use

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/morrison-turnansky/torch-compile-ai.git
cd torch-compile-ai

# Start Claude Code with the plugin
claude --plugin-dir .
```

The plugin automatically (via SessionStart hook):
- ✅ Installs required dependencies (`acp-steering-mcp`)
- ✅ Indexes PyTorch modules on first use
- ✅ Configures MCP server
- ✅ Loads all 8 skills and 5 agents

### Prerequisites

1. **Claude Code** installed and configured
2. **Python environment** with `uv` available
3. **PyTorch source** (optional, for API documentation):
   - If available at `/workspaces/pytorch-devcontainers/pytorch` or `~/pytorch`, indexing happens automatically
   - Otherwise, set `PYTORCH_SRC` environment variable

### Verify Installation

After starting Claude Code:

```bash
# Skills should be available
# Type "/" to see autocomplete - look for compile-overview, pytorch-dynamo, etc.

# Check that setup ran successfully
# Look for console output about:
#   - acp-steering-mcp installation
#   - PyTorch module indexing
```

### Future: Marketplace Install

Once the repo is public:

```bash
claude plugin marketplace add https://raw.githubusercontent.com/morrison-turnansky/torch-compile-ai/main/.claude-marketplace/marketplace.json
claude plugin install torch-compile-ai
```

## Usage

### Slash Commands

When using `--plugin-dir`:
```bash
/compile-overview
/pytorch-dynamo
/compile-trace-inductor
```

When installed via marketplace:
```bash
/torch-compile-ai:compile-overview
/torch-compile-ai:pytorch-dynamo
```

### Natural Language

Skills load automatically based on context:
```
"How do I debug a graph break in torch.compile?"
"Why isn't my reduction fusing?"
"Show me the Triton kernel for this code"
```

### Example Queries

**User-level debugging:**
```
Why does this graph break? def fn(x): return x[x.item()]
Show me the fusion decisions for this model
Parse these TORCH_LOGS and explain what happened
```

**API lookup:**
```
What are the parameters for Pointwise.__init__?
How do I use SymInt in C++ code?
```

**Implementation work:**
```
How do I implement a new VariableTracker type?
Where do I add a lowering for my custom op?
```

## Troubleshooting

### Plugin Not Loading

```bash
cd /path/to/torch-compile-ai
ls -la .claude-plugin/plugin.json  # Should exist
claude --plugin-dir .
```

### Skills Not Found

```bash
# Should show 8 SKILL.md files
find skills -name "SKILL.md"
```

### MCP Server Errors

```bash
# Check if steering is installed
which acp-steering-mcp

# If not, install manually
uv pip install git+https://github.com/ambient-code/steering.git
```

### Indexing Failed

```bash
# Check PyTorch source is available
ls -la ~/pytorch/torch/_dynamo
ls -la /workspaces/pytorch-devcontainers/pytorch/torch/_dynamo

# If in a different location, set PYTORCH_SRC
export PYTORCH_SRC=/path/to/pytorch
```

If PyTorch source is not available, API documentation will be limited but core debugging functionality still works.

### Re-index PyTorch

```bash
# Remove existing indices
rm -rf ~/.acp/repos/dynamo ~/.acp/repos/inductor ~/.acp/repos/functorch

# Restart Claude Code to trigger re-indexing
claude --plugin-dir .
```

## Documentation

- **[REPO_ARCH.md](REPO_ARCH.md)** - Architecture, structure, and development guide
- **[CLAUDE.md](CLAUDE.md)** - Code guidelines and testing

## License

Part of the PyTorch devcontainer tooling.
