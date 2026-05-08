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

Return JSON with **parsed findings** (not raw logs):
```json
{
  "specialist": "tracing-agent",
  "success": true,
  "debug_dir": "torch_compile_debug/run_2024_05_07_120000_000000/",
  "files_generated": ["fx_graph_readable.py", "output_code.py"],
  "parsed_logs": {
    "graph_breaks": "3 breaks found: tensor.item() (data-dependent)...",
    "fusion_decisions": "Fused 2 pointwise ops into kernel0..."
  },
  "summary": "Found 3 graph breaks on data-dependent operations"
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

### 3. Parse Stdout and Files

**Two types of output:**
- **Stdout logs** (ephemeral) - Parse immediately from Bash result
- **Debug files** (persistent) - Read from torch_compile_debug/ and parse

**Stdout-based MCP tools:**
- `parse_graph_breaks(stdout)` ← TORCH_LOGS="graph_breaks"
- `parse_fusion_decisions(stdout)` ← TORCH_LOGS="fusion,schedule"
- `parse_post_grad_passes(stdout)` ← TORCH_LOGS="post_grad_graphs"

**File-based MCP tools:**
- `parse_fx_graph(file_content)` ← fx_graph_readable.py
- `parse_pre_grad_passes(before, after)` ← fx files
- `parse_aot_joint_graph(file_content)` ← joint graph file
- `parse_aot_graphs(fwd, bwd)` ← forward/backward files
- `parse_ir_post_fusion(file_content)` ← ir_post_fusion_*.txt
- `parse_output_code(file_content)` ← output_code.py

**Workflow:**
```python
# 1. Run code
stdout = bash(f"TORCH_LOGS='{flags}' python temp.py")

# 2. Parse stdout immediately
findings = {}
if "graph_breaks" in flags:
    findings["graph_breaks"] = parse_graph_breaks(stdout)
if "fusion" in flags:
    findings["fusion"] = parse_fusion_decisions(stdout)

# 3. Find debug directory
debug_dir = find_latest_debug_dir("torch_compile_debug/")

# 4. Read and parse files
if "dynamo" in flags:
    fx_content = read(f"{debug_dir}/fx_graph_readable.py")
    findings["fx_graph"] = parse_fx_graph(fx_content)
    
if "output_code" in flags:
    code = read(f"{debug_dir}/output_code.py")
    findings["kernel"] = parse_output_code(code)
```

### 4. Handle Input Generation

If user didn't provide input:
- Tensors: `torch.randn(10, 128, device=device)`
- Scalars: `2.0`
- Lists: `[torch.randn(5, 5)]`

### 5. Return Structured Results

Return:
- Debug directory path (for file-based MCP tools later)
- Files generated (for coordinator to use file-based MCP tools)
- **Parsed stdout logs** (already analyzed, not raw text)
- Summary of findings

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
→ Returns: 
  {
    "debug_dir": "torch_compile_debug/run_*/",
    "files_generated": ["output_code.py"],
    "parsed_logs": {
      "fusion": "Fused 2 pointwise ops: relu + add"
    }
  }
```

**Debug graph breaks:**
```
code: "def fn(x): return x[x.item()]"
stage: "dynamo"
→ Returns:
  {
    "debug_dir": "torch_compile_debug/run_*/",
    "parsed_logs": {
      "graph_breaks": "1 break: tensor.item() - data-dependent operation"
    }
  }
```

**Full pipeline:**
```
code: "def fn(x, y): return torch.mm(x, y)"
stage: "all"
→ Returns:
  {
    "debug_dir": "torch_compile_debug/run_*/",
    "files_generated": ["fx_graph_readable.py", "output_code.py"],
    "parsed_logs": {
      "graph_breaks": "No breaks",
      "fusion": "Generated 1 mm kernel"
    }
  }
```

## Integration with Coordinator

Coordinator routes here when user provides code (not file paths):
```
User: "Show me the Triton kernel for: def fn(x): return x.relu()"

Coordinator workflow:
1. tracing-agent 
   - Runs code with TORCH_LOGS
   - Parses stdout (fusion decisions) using MCP tools
   - Returns parsed findings + debug_dir
2. Coordinator receives structured analysis
3. If file analysis needed: analyze_triton_codegen(debug_dir/output_code.py)
4. inductor-expert - deep explanation
```

**Division of Labor:**
- **Tracing-agent**: Parse stdout logs (ephemeral, only accessible during run)
- **Coordinator**: Use file-based MCP tools (persistent files in debug_dir)
- **Specialists**: Deep domain analysis

## Important Notes

- **Parse stdout before returning** - stdout logs are only in your context during execution
- Use MCP tools to parse stdout (parse_graph_breaks, parse_fusion_decisions, etc.)
- Return structured findings, not raw stdout text
- Generate sample input if user doesn't provide it
- Return absolute paths for debug_dir
- Capture errors with helpful suggestions
- Reference compile-trace skill for stage details

## Available MCP Tools (9 total)

**Dynamo Stage:**
1. `parse_graph_breaks(log_content)` - Stdout from TORCH_LOGS="graph_breaks"
2. `parse_fx_graph(graph_content)` - File: fx_graph_readable.py
3. `parse_pre_grad_passes(before, after)` - Files: fx_graph_readable.py, fx_graph_transformed.py

**AOT Stage:**
4. `parse_aot_joint_graph(graph_content)` - File: model__*__joint_*.py
5. `parse_aot_graphs(forward, backward)` - Files: forward/backward graphs
6. `parse_post_grad_passes(log_content)` - Stdout or file from TORCH_LOGS="post_grad_graphs"

**Inductor Stage:**
7. `parse_fusion_decisions(log_content)` - Stdout from TORCH_LOGS="fusion,schedule"
8. `parse_ir_post_fusion(ir_content)` - File: ir_post_fusion_*.txt
9. `parse_output_code(code_content)` - File: output_code.py

**Note:** All tools take content strings, not file paths. You must read files before calling.
