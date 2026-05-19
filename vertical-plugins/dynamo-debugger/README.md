# Dynamo Debugger Vertical

Debugging tools for PyTorch Dynamo compilation stage (bytecode capture, FX graph, graph breaks).

## Skills

- **`compile-trace-dynamo/`** - How to trace Dynamo stage
  - TORCH_LOGS configuration for graph_breaks, dynamo, pre_grad_graphs
  - Interpreting FX graph files (fx_graph_readable.py, fx_graph_transformed.py)
  - Understanding graph break reasons and debugging patterns
  - Pre-grad fusion patterns (Conv-BN, split-cat elimination)

- **`pytorch-dynamo/`** - Implementation knowledge
  - VariableTracker system architecture
  - Guard generation and symbolic shapes
  - Bytecode tracing mechanics
  - Graph break debugging patterns
  - Multiple sub-guides: ARCHITECTURE.md, GUARD.md, DEBUGGING-GUIDE.md, etc.

## Prompts

- **`dynamo-expert.md`** - Specialist agent for Dynamo analysis
  - Structured JSON output format
  - Knowledge base: Steering MCP (API docs) + Skills (deep implementation)
  - Scope: Graph capture, VariableTracker, guards, graph breaks

## MCP Tools

Uses **steering-mcp** for API documentation lookups:
- `query_api_docs` - Function signatures and docstrings
- `query_steering` - When/why/how guidance

## When to Use This Vertical

- Graph breaks in compilation
- Unsupported Python operations
- FX graph construction issues
- Pre-grad fusion not working (Conv-BN, etc.)
- Dynamo-specific debugging

## Related Verticals

- **Next stage**: `aot-debugger/` (AOT Autograd transformations)
- **Next stage**: `inductor-debugger/` (if skipping AOT, going directly to backend)
- **Routing**: `bisector/` determines if issue is in Dynamo stage
