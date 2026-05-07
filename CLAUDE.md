# Debug Tracer MCP

## Purpose

MCP server for parsing torch.compile debug output across all compilation pipeline stages (Dynamo, AOT Autograd, Inductor). Provides 13 specialized tools for analyzing FX graphs, fusion decisions, lowering transformations, Triton codegen, and cross-stage operation tracing.

**Design Philosophy**: Test-driven development with strong typing and real data verification.

## Code Guidelines

**TDD**: All functions require a corresponding test. Python tests use Pytest. Follow Red-Green-Refactor phases.

**Code Style**:

Python: ruff · pyright type checker · Google-style docstrings (no types in docstring body) · module-level docstrings on every .py file · docstrings on every test function · functions <200 lines (prefer <100)

**Type Safety**:
- Use modern Python 3.10+ type hints (list, dict, set, tuple - not List, Dict, Set, Tuple)
- Full type annotations on all function signatures
- No `Any` unless absolutely necessary
- Specific exception types in except blocks (no bare `except`)

**Import Style**:
- Avoid local imports
- Use explicit module imports (e.g., `from analyzers.dynamo import parse_graph_breaks`)
- Import from modules, not package `__init__.py`

**Documentation**:
- Google-style docstrings with Args and Returns sections
- No type information in docstrings (types are in hints)
- Module-level docstrings explaining purpose
- Test docstrings explaining what is being tested

**Error Handling**:
- Avoid try/except unless for I/O operations or external API calls
- Let errors propagate for debugging
- Use specific FileNotFoundError, ValueError, etc. instead of generic Exception

**Code Organization**:
- One test file per source file (tests/analyzers/test_dynamo_trace.py for analyzers/dynamo_trace.py)
- Test classes named Test<FunctionName> (e.g., TestParseGraphBreaks)
- Keep functions focused and under 200 lines

## Architecture Overview

### Execution Flow

```
MCP Server (server.py)
    ↓
    ├─ Dynamo Stage Tools
    │  ├─ parse_graph_breaks() → TORCH_LOGS analysis
    │  ├─ analyze_fx_graph() → FX graph structure
    │  └─ analyze_pre_grad_passes() → Pre-grad optimizations
    │
    ├─ AOT Autograd Stage Tools
    │  ├─ analyze_functionalization() → Inplace → functional
    │  ├─ analyze_joint_graph() → Forward/backward graphs
    │  ├─ analyze_partitioning() → F/B split + saved activations
    │  └─ analyze_post_grad_passes() → Post-grad optimizations
    │
    ├─ Inductor Stage Tools
    │  ├─ parse_fusion_decisions() → Fusion log parsing
    │  ├─ analyze_triton_codegen() → Triton kernel analysis
    │  ├─ analyze_lowering() → ATen → IR mapping
    │  └─ analyze_loopbody() → ops.* operation analysis
    │
    └─ Cross-Stage Tools
       ├─ trace_operation() → Trace op through all 5 stages
       └─ search_ir() → Regex search across stages
```

### Module Dependencies

```
analyzers/
├─ __init__.py (re-exports all functions)
├─ dynamo_trace.py (no internal dependencies)
├─ aot_trace.py (no internal dependencies)
├─ inductor_trace.py (uses pathlib.Path)
└─ cross_stage_trace.py (no internal dependencies)

tests/analyzers/
├─ test_dynamo_trace.py → analyzers.dynamo_trace
├─ test_aot_trace.py → analyzers.aot_trace
├─ test_inductor_trace.py → analyzers.inductor_trace
└─ test_cross_stage_trace.py → analyzers.cross_stage_trace
```

## Key Components

### Parsers by Stage

**Dynamo Stage (FX Graph Generation)** - `analyzers/dynamo_trace.py`
- `parse_graph_breaks` - Parse TORCH_LOGS="graph_breaks" output, categorize by type, suggest fixes
- `analyze_fx_graph` - Analyze fx_graph_readable.py structure, count operations, identify patterns
- `analyze_pre_grad_passes` - Detect pre-grad optimizations (constant folding, canonicalization)

**AOT Autograd Stage (Training Mode)** - `analyzers/aot_trace.py`
- `analyze_functionalization` - Verify inplace ops converted to functional
- `analyze_joint_graph` - Analyze forward/backward graph structure
- `analyze_partitioning` - Show forward/backward split and saved activations
- `analyze_post_grad_passes` - Detect post-grad optimizations (CSE, DCE)

**Inductor Stage (Lowering and Codegen)** - `analyzers/inductor_trace.py`
- `parse_fusion_decisions` - Parse TORCH_LOGS="fusion" output, explain success/failure
- `analyze_triton_codegen` - Analyze Triton kernels (tiling, memory, performance)
- `analyze_lowering` - Map ATen ops to Inductor IR nodes
- `analyze_loopbody` - Analyze ops.* operations in ir_post_fusion.txt

**Cross-Stage Tools** - `analyzers/cross_stage_trace.py`
- `trace_operation` - Trace operation through all 5 stages (FX → AOT → IR → LoopBody → Triton)
- `search_ir` - Regex search across all stages with line numbers

### Data Flow

1. **torch.compile debug output** → stored in `torch_compile_debug/run_*/torchinductor/`
2. **MCP tool call** → server.py routes to appropriate analyzer
3. **Analyzer parses files** → extracts information via regex/parsing
4. **Formatted result** → returned as markdown string

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
    │   └── output_code.py           # Generated Triton code
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

Server routes to `analyzers.dynamo.parse_graph_breaks(log_content)` and returns formatted string.

### Claude Code Configuration

Add to Claude Code settings.json:

```json
{
  "mcpServers": {
    "debug-tracer": {
      "command": "python",
      "args": ["/path/to/torch-compile-ai/server.py"]
    }
  }
}
```

## Testing

All parsers tested with real torch.compile output:

```bash
# All tests
pytest tests/analyzers/ -v

# Specific stage
pytest tests/analyzers/test_dynamo_trace.py -v

# Specific test
pytest tests/analyzers/test_inductor_trace.py::TestAnalyzeLowering::test_lowering_basic -v
```

**Test Requirements**:
- Use real torch.compile debug files in torch_compile_debug/
- Mock only external dependencies (never mock internal functions)
- Test both success and error cases
- Verify output format and key information

## Installation

```bash
# Install in editable mode (required for tests)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

**Requirements**:
- Python 3.10+
- mcp>=1.0.0
- pytest, ruff, pyright (dev only)

## Design Decisions

### Async Functions for Consistency
All parsers are async (`async def`) even though most don't do I/O, for consistency with MCP async tool handlers.

### String Returns
All parsers return formatted markdown strings (not JSON) for human-readable output in Claude conversations.

### Real File Parsing Only
No synthetic data in tests - all tests use actual torch.compile debug output to ensure parsers work with real data.

### Test-First Development
Write test before implementation to validate requirements and ensure testability.

### Strong Typing Required
All function signatures fully typed. No `Any` without justification. Use modern type hints (list vs List).
