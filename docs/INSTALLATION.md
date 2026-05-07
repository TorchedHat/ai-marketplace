# Multi-Agent System Installation Guide

Complete setup guide for the torch.compile multi-agent development system in containerized environments.

## Overview

This system provides:
- **torch-compile-ai**: Parse torch.compile debug logs (13 parsers across 4 stages)
- **acp-steering-mcp**: API documentation and code navigation
- **Multi-agent prompts**: Coordinator + specialist prompts for routing and synthesis

## Container Environment

**IMPORTANT:** Only `/workspaces/` persists as a volume. Everything else (home directory, pip packages) is destroyed on container restart.

**Persistent Storage:**
- All code: `/workspaces/pytorch-devcontainers/`
- AI tooling: `/workspaces/pytorch-devcontainers/ai-tooling/`
- PyTorch indices: `/workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/`

**Recreated on Startup:**
- Python packages (pip install)
- `~/.claude/settings.json` (MCP configuration)

## Quick Start (Automated)

Run the setup script on container startup:

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

This script:
1. Installs Python packages (acp-steering-mcp, torch-compile-ai)
2. Creates indices directory in `/workspaces/` (persists)
3. Indexes PyTorch modules (if not already indexed)
4. Configures `~/.claude/settings.json` (recreated each startup)
5. Verifies setup and tests MCP servers

**First run:** ~10-15 minutes (includes indexing)
**Subsequent runs:** ~30 seconds (indices already exist)

## Manual Installation

If you prefer step-by-step setup:

### 1. Install Python Packages

```bash
# Install torch-compile-ai
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
pip install -e .

# Install acp-steering-mcp from GitHub (for API lookups)
pip install "git+https://github.com/ambient-code/steering.git"
```

This installs:
- `torch-compile-ai` (editable mode)
- `mcp` (Model Context Protocol)
- `acp-steering-mcp` (for API lookups, optional)

Verify:

```bash
which acp-steering-mcp repomap
# Should output paths to installed binaries
```

### 2. Index PyTorch Modules

Create indices in `/workspaces/` (persists across container restarts):

```bash
# Navigate to PyTorch source
cd /workspaces/pytorch-devcontainers/pytorch

# Create persistent index directory
mkdir -p /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices

# Index torch._dynamo (~2-3 minutes)
repomap ./torch/_dynamo --repo-name dynamo --verbose
mv ~/.acp/repos/dynamo /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/

# Index torch._inductor (~5-8 minutes)
repomap ./torch/_inductor --repo-name inductor --verbose
mv ~/.acp/repos/inductor /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/
```

**Expected Results:**
- `/workspaces/.../ai-tooling/.acp-indices/dynamo/`: 1,208 functions, 647 classes
- `/workspaces/.../ai-tooling/.acp-indices/inductor/`: 2,457 functions, 1,122 classes

Verify:

```bash
ls /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/
# Should show: dynamo/ inductor/

cat /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/dynamo/steering.json | grep -A 5 statistics
```

### 3. Configure MCP Servers

Create `~/.claude/settings.json` (must be recreated on each container startup):

```json
{
  "skipDangerousModePermissionPrompt": true,
  "mcpServers": {
    "steering": {
      "command": "acp-steering-mcp",
      "env": {
        "STEERING_REPOS_PATH": "/workspaces/pytorch-devcontainers/ai-tooling/.acp-indices"
      }
    },
    "debug-tracer": {
      "command": "python",
      "args": [
        "/workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai/server.py"
      ],
      "cwd": "/workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai",
      "env": {
        "PYTHONPATH": "/workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai"
      }
    }
  }
}
```

**Critical Paths:**
- `STEERING_REPOS_PATH`: Must point to persistent indices in `/workspaces/`
- `args[0]` and `cwd`: Must point to ai-tooling in `/workspaces/`

### 5. Verify MCP Servers

Start Claude Code and verify MCP tools are available:

```bash
# In Claude Code session
User: What MCP tools are available?

# Expected output should include:
# torch-compile-ai tools:
#   - parse_dynamo_guards
#   - parse_dynamo_graph
#   - parse_aot_forward_graph
#   - parse_aot_backward_graph
#   - parse_inductor_post_grad_graph
#   - parse_inductor_output_code
#   - parse_compiled_module
#   - parse_fx_graph_code
#   - parse_fx_graph_sizevars
#   - parse_fx_graph_cache_lookup
#   - find_graph_breaks
#   - find_recompiles
#   - analyze_guards
#
# steering-mcp tools:
#   - query_api_docs
#   - query_class_hierarchy
#   - list_symbols
```

Test torch-compile-ai:

```bash
# Create a test debug directory
TORCH_LOGS="+dynamo,+inductor" python -c "import torch; torch.compile(lambda x: x + 1)(torch.randn(10))"

# In Claude Code
User: Parse the guards from torch_compile_debug/run_2024_01_01_00_00_00_000000/torchdynamo/debug.log
```

Test steering-mcp:

```bash
# In Claude Code
User: What are the parameters for Pointwise.__init__?
```

## Directory Structure

After installation:

```
/workspaces/pytorch-devcontainers/        # PERSISTENT VOLUME
├── ai-tooling/
│   ├── .acp-indices/                    # PERSISTENT - PyTorch indices
│   │   ├── dynamo/
│   │   │   ├── steering.json            # Index metadata
│   │   │   ├── _.md                     # Root module (16,262 lines)
│   │   │   ├── variables.md             # variables/ module (7,572 lines)
│   │   │   └── ...                      # Other modules
│   │   └── inductor/
│   │       ├── steering.json            # Index metadata
│   │       ├── _.md                     # Root module (22,486 lines)
│   │       ├── codegen.md               # codegen/ module (15,891 lines)
│   │       └── ...                      # Other modules
│   │
│   └── torch-compile-ai/
│       ├── setup.sh                     # Automated setup script
│       ├── server.py                    # MCP server entry point
│       ├── analyzers/                     # 13 parser implementations
│       ├── prompts/                     # Coordinator + specialist prompts
│       │   ├── coordinator-concise.md   # Main routing agent (126 lines)
│       │   ├── dynamo-expert-concise.md # Dynamo specialist (95 lines)
│       │   └── inductor-expert-concise.md # Inductor specialist (131 lines)
│       ├── tests/                       # Parser tests + scenarios
│       │   ├── analyzers/                 # 23 parser unit tests
│       │   └── multi-agent/             # 5 end-to-end scenarios
│       └── docs/                        # This file + guides
│
└── pytorch/                             # PyTorch source (for indexing)
    ├── torch/_dynamo/
    └── torch/_inductor/

~/.claude/                               # EPHEMERAL - recreated on startup
└── settings.json                        # MCP configuration (from setup.sh)
```

## Using the Multi-Agent System

### Load Coordinator Prompt

In Claude Code, load the coordinator:

```bash
# Option 1: Load via file
User: Load the coordinator prompt from /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai/prompts/coordinator-concise.md

# Option 2: Manual paste
# Copy the contents of coordinator-concise.md and paste into Claude Code
```

### Test Scenarios

Try one of the test scenarios from `tests/multi-agent/test_scenarios.md`:

**Scenario 1: Simple Lowering Question**
```
User: How does Pointwise handle broadcasting in the Inductor IR?

Expected Routing:
1. steering-mcp - API lookup for Pointwise class
2. inductor-expert - explain broadcasting logic

Expected Synthesis:
- API signature for Pointwise.__init__
- Broadcasting explanation from inductor-expert
- Code example showing broadcast handling
```

**Scenario 2: Debug Graph Break**
```
User: Parse guards from torch_compile_debug/run_*/torchdynamo/debug.log 
      and explain why we're getting graph breaks on dynamic shapes

Expected Routing:
1. torch-compile-ai - parse_dynamo_guards
2. dynamo-expert - analyze guard failures

Expected Synthesis:
- Guard analysis from torch-compile-ai
- Root cause from dynamo-expert
- Recommended fix
```

See `tests/multi-agent/test_scenarios.md` for all 5 scenarios.

## Maintenance

### Re-indexing After PyTorch Updates

```bash
cd /workspaces/pytorch-devcontainers/pytorch
git pull

# Re-index both modules (overwrites existing indices in /workspaces/)
cd /workspaces/pytorch-devcontainers/pytorch
repomap ./torch/_dynamo --repo-name dynamo --verbose
mv ~/.acp/repos/dynamo /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/

repomap ./torch/_inductor --repo-name inductor --verbose
mv ~/.acp/repos/inductor /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/
```

### Updating Prompts

Prompts are in `prompts/`:
- `coordinator-concise.md` - Main routing logic
- `dynamo-expert-concise.md` - Dynamo specialist
- `inductor-expert-concise.md` - Inductor specialist

Edit these files to adjust routing rules or output formats.

### Monitoring Index Size

```bash
du -sh /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/dynamo
du -sh /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/inductor
du -sh /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/
```

Current sizes:
- dynamo: ~784KB
- inductor: ~1.6MB
- Total: ~2.4MB

## Troubleshooting

### MCP Server Not Starting

**Symptom:** "debug-tracer MCP server not available"

**Fix:**
```bash
# Test server manually
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
python server.py
# Should start and wait for input

# Check settings.json paths
cat ~/.claude/settings.json | grep -A 10 debug-tracer

# Verify Python path
which python
python -c "import sys; print(sys.path)"
```

### Steering Queries Return No Results

**Symptom:** "Symbol not found" for valid PyTorch classes

**Fix:**
```bash
# Check indices exist in persistent storage
ls /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/dynamo/
ls /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/inductor/

# Verify index has content
cat /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/dynamo/steering.json

# Verify settings.json points to correct path
cat ~/.claude/settings.json | grep STEERING_REPOS_PATH
# Should show: /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices

# Re-run setup.sh if paths are wrong
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

### Parser Tests Failing

**Symptom:** Tests fail with "file not found" or parsing errors

**Fix:**
```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai

# Run tests with verbose output
python -m pytest tests/analyzers/ -v

# Check test fixtures exist
ls tests/fixtures/

# Regenerate fixtures if needed
TORCH_LOGS="+dynamo,+inductor" python tests/generate_fixtures.py
```

## Performance Metrics

### Context Efficiency

Compared to loading all skills (~150KB):
- MCP-only query: ~10KB (87% reduction)
- Single specialist: ~60KB (60% reduction)
- Multi-specialist: ~110KB (27% reduction)

### Prompt Size

- Standard prompts: 733 lines (~15KB)
- Concise prompts: 352 lines (~7KB)
- Reduction: 52%

### Query Latency

- Steering MCP API lookup: <1s
- Debug tracer parse: 1-3s (depends on log size)
- Specialist analysis: 5-10s (with context loading)

## Next Steps

1. Load coordinator prompt
2. Run test scenarios
3. Measure routing accuracy
4. Iterate on prompts based on results

## References

- Debug Tracer MCP: `docs/DEBUG_TRACER.md`
- Steering Setup: `docs/STEERING_SETUP.md`
- Test Scenarios: `tests/multi-agent/test_scenarios.md`
- Prompt Comparison: `docs/PROMPT_COMPARISON.md`
- Configuration: `docs/CONFIGURATION.md`
