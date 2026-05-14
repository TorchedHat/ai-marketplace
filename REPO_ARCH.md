# Repository Architecture

## Overview

MCP server providing 9 parsers for torch.compile debug output across all pipeline stages (Dynamo, AOT Autograd, Inductor).

**Design Philosophy**: Simple parsers aligned with torch.compile IR levels, supporting both TORCH_LOGS stdout and debug file content.

## Project Structure

```
torch-compile-ai/
├── analyzers/                  # 9 parser implementations
│   ├── __init__.py            # Re-exports all parsers
│   ├── dynamo_parsers.py      # Dynamo stage (3 parsers)
│   ├── aot_parsers.py         # AOT stage (3 parsers)
│   └── inductor_parsers.py    # Inductor stage (3 parsers)
├── server.py                   # MCP server entry point
├── tests/analyzers/              # Parser unit tests
│   ├── test_dynamo_parsers.py
│   └── test_inductor_parsers.py
├── prompts/                    # Multi-agent prompts
│   ├── coordinator-concise.md
│   ├── tracing-agent-concise.md
│   ├── dynamo-expert-concise.md
│   └── inductor-expert-concise.md
└── docs/                       # User documentation
```

## Architecture

### Execution Flow

```
User Request
    ↓
Coordinator (routing + synthesis)
    ↓
    ├─→ steering (API lookups)
    ├─→ torch-compile-ai (parse stdout/files)
    │   └─→ 9 parsers aligned with IR levels
    ├─→ tracing-agent (generate debug output)
    ├─→ dynamo-expert (Dynamo analysis)
    └─→ inductor-expert (Inductor analysis)
    ↓
Synthesized Response
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
      "args": ["/workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai/server.py"],
      "cwd": "/workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai",
      "env": {
        "PYTHONPATH": "/workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai"
      }
    }
  }
}
```

## Installation

### Container Environment

**Persistent:** `/workspaces/` (code + PyTorch indices)  
**Ephemeral:** `~/.claude/settings.json`, pip packages

### Setup Script

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

The `setup.sh` script:
- Installs pip packages (fast, ~30s)
- Stores indices in `/workspaces/ai-tooling/.acp-indices/` (persists)
- Recreates `~/.claude/settings.json` on each startup

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

### Coordinator
Routes tasks to specialists, synthesizes results, presents unified guidance.

### Specialists
- **tracing-agent**: Generate debug output, parse stdout/files, return structured findings
- **dynamo-expert**: VariableTracker, guards, graph breaks
- **inductor-expert**: Lowerings, IR nodes, Triton, fusion

### MCP Servers
- **torch-compile-ai**: 9 parsers (this repository)
- **steering-mcp**: API documentation and code navigation

## Performance

### Context Efficiency
- MCP-only query: ~10KB (87% reduction vs all skills)
- Single specialist: ~60KB (60% reduction)
- Multi-specialist: ~110KB (27% reduction)

### Query Latency
- Parsing: <1s for stdout, 1-3s for large files
- Specialist analysis: 5-10s

## Code Quality

- ✅ **Type hints**: Modern Python 3.10+ annotations
- ✅ **Google docstrings**: Args/Returns documented
- ✅ **TDD**: 8 tests, all passing
- ✅ **Linted**: ruff + pyright compliant
