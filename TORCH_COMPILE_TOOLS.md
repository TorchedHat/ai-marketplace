# torch.compile Debugging Tools

Multi-agent debugging system for PyTorch torch.compile development. Provides stage-specific skills, specialized agents, and semantic API search for Dynamo, AOT Autograd, and Inductor.

## Available Tools

### Skills (10)

**torch.compile debugging:**
- `compile-overview` - torch.compile pipeline architecture and stages reference
- `compile-bisect` - Automatic bisection to find failing operations
- `compile-trace-dynamo` - Debug Dynamo graph capture and tracing
- `compile-trace-aot` - Debug AOT Autograd functionalization and gradients
- `compile-trace-inductor` - Debug Inductor lowering and codegen

**PyTorch implementation:**
- `pytorch-dynamo` - Dynamo implementation guidance and patterns
- `pytorch-aot` - AOT Autograd implementation guidance
- `pytorch-inductor` - Inductor implementation guidance

**Skill/agent development:**
- `skill-writer` - Create new Claude Code skills
- `agent-writer` - Create specialized agent definitions

### Specialized Agents (4)

- `compile-debug` - Multi-stage compilation debugging coordinator
- `dynamo-expert` - Dynamo graph capture and tracing specialist
- `aot-expert` - AOT Autograd functionalization and gradient specialist
- `inductor-expert` - Inductor lowering and codegen specialist

### MCP Server Integration

**steering** - Semantic search over PyTorch modules:
- Dynamo (`torch/_dynamo`)
- Inductor (`torch/_inductor`)
- Functorch (`torch/_functorch`)
- Auto-indexed on first use from PyTorch source

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
/ai-marketplace:compile-overview
/ai-marketplace:pytorch-dynamo
```

### Natural Language

Skills load automatically based on context:
```
"How do I debug a graph break in torch.compile?"
"Why isn't my reduction fusing?"
"Show me the Triton kernel for this code"
```

### Example Queries

**torch.compile debugging:**
```
Why does this graph break? def fn(x): return x[x.item()]
Show me the fusion decisions for this model
Parse these TORCH_LOGS and explain what happened
Bisect this compilation failure to find the exact failing op
```

**PyTorch API lookup (via steering MCP):**
```
What are the parameters for Pointwise.__init__?
How do I use SymInt in C++ code?
Show me FakeTensor usage patterns
```

**PyTorch implementation work:**
```
How do I implement a new VariableTracker type?
Where do I add a lowering for my custom op?
What's the AOT partitioning algorithm?
```

**Skill/agent development:**
```
/skill-writer - Create a new Claude Code skill
/agent-writer - Create a specialized agent definition
```

## Prerequisites

1. **Claude Code** installed and configured
2. **Python environment** with `uv` available
3. **PyTorch source** (optional, for API documentation):
   - If available at `/workspaces/pytorch-devcontainers/pytorch` or `~/pytorch`, indexing happens automatically
   - Otherwise, set `PYTORCH_SRC` environment variable

## Automatic Setup

The plugin automatically (via SessionStart hook):
- ✅ Installs required dependencies (`acp-steering-mcp`)
- ✅ Indexes PyTorch modules on first use
- ✅ Configures steering MCP server
- ✅ Loads all 10 skills and 4 agents

## Troubleshooting

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

## Architecture

See [REPO_ARCH.md](REPO_ARCH.md) for detailed architecture, including:
- Stage-specific debugging workflow
- Agent delegation patterns
- Skill trigger mechanisms
- MCP server integration
