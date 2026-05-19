# AOT Debugger Vertical

Debugging tools for PyTorch AOT Autograd stage (functionalization, decompositions, graph transformations).

## Skills

- **`compile-trace-aot/`** - How to trace AOT stage
  - TORCH_LOGS configuration for aot, aot_graphs, aot_joint_graph, post_grad_graphs
  - Interpreting AOT graph files (model__*__joint_*.py, model__*__forward_*.py, model__*__backward_*.py)
  - Understanding functionalization and decompositions
  - Debugging partitioning and recomputation strategies
  - Post-grad fusion patterns (GEMM, attention)

## Prompts

Currently no dedicated AOT expert prompt (fewer issues at this stage, simpler patterns).

## MCP Tools

Uses **steering-mcp** for API documentation lookups when needed.

## When to Use This Vertical

- Training-specific issues (gradients, backward pass)
- Functionalization problems
- Decomposition issues
- OOM during training (check partitioning/recomputation)
- Post-grad pattern matching not working (GEMM fusion, etc.)

## Related Verticals

- **Previous stage**: `dynamo-debugger/` (FX graph capture)
- **Next stage**: `inductor-debugger/` (backend compilation)
- **Routing**: `bisector/` determines if issue is in AOT stage (backend='aot_*')

## Notes

AOT stage runs for both training and inference (not training-only). The joint graph creation and backward-specific transformations only happen when gradients are required.
