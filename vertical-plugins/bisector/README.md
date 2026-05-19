# Bisector Vertical

Automated compilation failure isolation using PyTorch's compiler bisector tool.

## Skills

- **`compile-bisect/`** - How to use compiler bisector
  - CLI usage: `python -m torch._inductor.compiler_bisector run python repro.py`
  - Programmatic usage: `CompilerBisector.do_bisect(test_function)`
  - Understanding bisect output (backend, subsystem, bisect_number, debug_info)
  - Routing logic based on bisect results
  - Writing test scripts that return proper exit codes

## Workflow

1. **Run bisector** on failing compilation
2. **Bisect output** identifies exact stage and operation
3. **Route to vertical** based on backend:
   - `backend='eager'` → Load `dynamo-debugger/` vertical
   - `backend='aot_*'` → Load `aot-debugger/` vertical  
   - `backend='inductor'` → Load `inductor-debugger/` vertical
4. **Debug with stage skill** using targeted TORCH_LOGS
5. **Verify fix** by running bisector again

## When to Use This Vertical

- **Always start here** for compilation failures
- When you don't know which stage is failing
- To pinpoint exact failing operation before deep diving
- To verify fixes after making changes

## Output Format

```python
BisectionResult(
    backend='inductor',           # Which backend failed
    subsystem='lowerings',        # Which subsystem (if backend has subsystems)
    bisect_number=42,             # Index in binary search
    debug_info='aten.argmin.default'  # Specific operation that failed
)
```

## Related Verticals

Routes to all other verticals based on bisection results:
- **dynamo-debugger/** - If `backend='eager'`
- **aot-debugger/** - If `backend='aot_*'`
- **inductor-debugger/** - If `backend='inductor'`

## MCP Tools

No MCP tools needed - bisector is a standalone Python tool.
