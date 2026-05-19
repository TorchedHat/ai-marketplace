# Inductor Debugger Vertical

Debugging tools for PyTorch Inductor compiler backend (IR lowering, fusion, scheduling, Triton/C++ codegen).

## Skills

- **`compile-trace-inductor/`** - How to trace Inductor stage
  - TORCH_LOGS configuration for fusion, schedule, ir_post_fusion, output_code
  - config.trace.enabled for IR debug files
  - Interpreting Inductor IR files (ir_*.txt, ir_post_fusion_*.txt, output_code.py)
  - Understanding fusion decisions and scheduling
  - Debugging kernel generation and performance

- **`pytorch-inductor/`** - Implementation knowledge
  - Lowering registration patterns
  - IR node architecture (Pointwise, Reduction, etc.)
  - Triton codegen and template system
  - Fusion patterns and scheduling
  - Multiple sub-guides: TRITON-CODEGEN.md, COMMON-PATTERNS.md, etc.

## Prompts

- **`inductor-expert.md`** - Specialist agent for Inductor analysis
  - Structured JSON output format
  - Knowledge base: Steering MCP (API docs) + Skills (deep implementation)
  - Scope: Lowerings, IR nodes, Triton codegen, fusion patterns

## MCP Tools

Uses **steering-mcp** for API documentation lookups:
- `query_api_docs` - Function signatures (Pointwise.__init__, register_lowering, etc.)
- `query_steering` - When/why/how guidance

## When to Use This Vertical

- Missing operator lowerings
- Fusion not happening (ops not fusing into single kernel)
- Wrong dtype or semantics in generated code
- Slow kernel performance
- Triton compilation errors
- Understanding generated kernels

## Related Verticals

- **Previous stage**: `dynamo-debugger/` (FX graph capture)
- **Previous stage**: `aot-debugger/` (AOT transformations)
- **Routing**: `bisector/` determines if issue is in Inductor stage (backend='inductor')

## Subsystems

When bisector identifies `backend='inductor'`, it may further narrow down:
- `subsystem='lowerings'` → Missing operator lowering
- `subsystem='pre_grad_passes'` → Pre-grad optimization issue
- `subsystem='post_grad_passes'` → Post-grad optimization issue
- `subsystem='cudagraphs'` → CUDA graphs wrapper issue
