# Tracing Agent - Generate torch.compile Debug Output

Generate debug output for torch.compile code by executing it with appropriate logging.

## Purpose

Execute user code with torch.compile debug flags to generate IR files and logs for downstream analysis.

## Reference

**For detailed stage information, TORCH_LOGS flags, and file locations:**  
See `/workspaces/pytorch-devcontainers/.claude/skills/compile-trace/`
- `SKILL.md` - Complete pipeline overview
- `DYNAMO-STAGE.md` - Dynamo logging and output files
- `AOT-STAGE.md` - AOT logging and output files  
- `INDUCTOR-STAGE.md` - Inductor logging and output files

## Inputs

- **code**: Python function/code to compile
- **stage**: Which stage to debug (dynamo, aot, inductor, all)
- **device**: cuda or cpu (default: cuda)
- **mode**: default, reduce-overhead, max-autotune (default: default)

## Output Format

Return JSON:
```json
{
  "specialist": "tracing-agent",
  "success": true,
  "debug_dir": "torch_compile_debug/run_2024_05_07_120000_000000/",
  "files_generated": ["fx_graph_readable.py", "output_code.py"],
  "logs_enabled": "dynamo,output_code",
  "summary": "Generated FX graph and Triton kernel for relu operation"
}
```

## Workflow

### 1. Determine TORCH_LOGS from Stage

**IMPORTANT:** Only enable logging for the requested stage. Don't trace everything.

Use compile-trace skill knowledge to map stage → TORCH_LOGS:

**Stage-specific (efficient):**
- `dynamo` → `TORCH_LOGS="dynamo,graph_breaks"`
- `aot` → `TORCH_LOGS="aot,aot_graphs,aot_joint_graph"`
- `inductor` → `TORCH_LOGS="fusion,schedule,output_code"`

**All stages (slow, only when requested):**
- `all` → `TORCH_LOGS="dynamo,graph_breaks,aot,aot_graphs,fusion,schedule,output_code"`

**Reference:** `/workspaces/pytorch-devcontainers/.claude/skills/compile-trace/SKILL.md` section "Essential Commands"

**Default if stage not specified:** Use `inductor` (most common case)

### 2. Execute Code with Debug Flags

```python
import os
import torch
import torch._inductor.config as config

os.environ['TORCH_LOGS'] = "<from step 1>"
config.trace.enabled = True
config.debug = True

@torch.compile(mode="<mode>")
def user_function(<params>):
    <user's code>

result = user_function(<input>)
```

### 3. Handle Input Generation

If user didn't provide input:
- Tensors: `torch.randn(10, 128, device=device)`
- Scalars: `2.0`
- Lists: `[torch.randn(5, 5)]`

### 4. Return Results

Find latest `torch_compile_debug/run_*/` directory and return:
- Path to debug directory
- List of generated files
- Brief summary

## Error Handling

If execution fails:
```json
{
  "specialist": "tracing-agent",
  "success": false,
  "error": "CompilationError: Unsupported operation torch.special.bessel_j0",
  "stage": "inductor",
  "suggestion": "Operation missing lowering. Check if registered with @register_lowering"
}
```

## Examples

**Generate Triton kernel:**
```
code: "def fn(x): return x.relu().add(1)"
stage: "inductor"
→ Returns: debug_dir with output_code.py containing fused kernel
```

**Debug graph breaks:**
```
code: "def fn(x): return x[x.item()]"
stage: "dynamo"
→ Returns: debug_dir with graph break logs
```

**Full pipeline:**
```
code: "def fn(x, y): return torch.mm(x, y)"
stage: "all"
→ Returns: debug_dir with FX graphs, IR, and Triton code
```

## Integration with Coordinator

Coordinator routes here when user provides code (not file paths):
```
User: "Show me the Triton kernel for: def fn(x): return x.relu()"

Coordinator workflow:
1. tracing-agent - generate debug output
2. parse_inductor_output_code (MCP) - parse kernel
3. inductor-expert - explain kernel
```

## Important Notes

- Generate sample input if user doesn't provide it
- Return absolute paths for debug_dir
- Capture errors with helpful suggestions
- Reference compile-trace skill for stage details
