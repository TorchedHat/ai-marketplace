# Getting Started

Quick guide to using the multi-agent system for PyTorch debugging.

## One-Time Setup

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

**First run:** ~10-15 minutes (includes PyTorch indexing)  
**Subsequent runs:** ~30 seconds

This installs dependencies and configures MCP servers.

## Using the System

### 1. Start Claude Code

Just start a normal Claude Code session.

### 2. Load the Coordinator

```
Load the coordinator prompt from /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai/prompts/coordinator-concise.md
```

### 3. Describe Your Problem

Talk naturally - the coordinator will suggest tools and specialists.

## Example Sessions

### Example 1: Debug Graph Breaks

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: I'm getting graph breaks when I use tensor.item() 
      in my model. Debug logs are at:
      torch_compile_debug/run_2024_05_07_120000_000000/
      
      Can you help me understand why and how to fix it?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Coordinator: Based on your task, I recommend:
1. torch-compile-ai - parse_dynamo_guards to analyze guards
2. dynamo-expert - explain tensor.item() graph breaks

Should I proceed?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: yes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Coordinator calls tools and returns synthesized answer]

Guard Analysis:
• 3 graph breaks found on tensor.item()
• Reason: "data-dependent operation" 
• Location: model.py:45

Why This Happens:
tensor.item() reads a value at compile time, which 
creates a data dependency. The compiler can't trace 
through this without knowing the runtime value.

Fix Options:
1. Remove .item() - if you don't need the scalar
2. Use torch.compiler.assume_constant_result() - if 
   the value is truly constant
3. Accept the break - if it's unavoidable

Would you like me to show how to apply fix #2?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: yes, show me the code
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Coordinator continues conversation, shows code example]
```

### Example 3: Debug Fusion Issue (User Provides Code)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: Why isn't this fusing?

def slow_fn(x):
    return x.sum(dim=-1).relu()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Coordinator: I recommend:
1. tracing-agent - generate debug output with fusion logs
2. parse_inductor_output_code (torch-compile-ai) - check kernels
3. inductor-expert - analyze fusion

Proceed?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: yes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kernel Analysis:
• 2 separate kernels generated (not fused)
• Reduction (sum) + Pointwise (relu)

Issue:
Different iteration spaces prevent fusion. Your 
sum() reduces to shape (10,) but relu expects (10, 128).

Fix:
Add .unsqueeze(-1) after sum to match shapes:

```python
def fast_fn(x):
    return x.sum(dim=-1, keepdim=True).relu()
```

Now fuses to 1 kernel (2x faster).
```

### Example 3: Adding New Lowering

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: I need to add a lowering for torch.special.bessel_j0
      What's the pattern I should follow?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Coordinator: I recommend:
1. steering-mcp - look up register_lowering and Pointwise API
2. inductor-expert - explain lowering pattern

Proceed?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: yes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[API lookup + expert guidance]

API Reference:
@register_lowering(torch.ops.aten.bessel_j0)
def bessel_j0_lowering(x):
    # Returns IR node

Lowering Pattern:
For elementwise math operations like bessel_j0:

1. Use @register_lowering decorator
2. Create Pointwise IR node
3. Implement in ops.bessel_j0 (Triton)

Here's the complete pattern:
[Shows code example with file locations]
```

### Example 4: Trace Operation Through Pipeline

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: Trace relu through the compilation pipeline.
      Debug logs at torch_compile_debug/run_2024_05_07_*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Coordinator: I recommend:
1. trace_operation (torch-compile-ai) - trace through all 5 stages

Proceed?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: yes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Operation Trace: relu

Stages Found: 5/5

Dynamo (FX): ✓ Found (torch.ops.aten.relu.default)
AOT (Joint): ✓ Found (aten.relu)
Inductor (IR): ✓ Found (ops.relu)
LoopBody: ✓ Found (ops.relu)
Triton: ✓ Found (tl.relu)

Pipeline Summary:
Python source → FX Graph → AOT Autograd → 
Inductor IR → LoopBody → Triton Kernel

Your relu operation successfully flows through all stages!
```

### Example 5: Understanding Code

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: What's a VariableTracker and when should I create 
      a new subclass?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Coordinator: I recommend:
1. steering-mcp - query VariableTracker class hierarchy
2. dynamo-expert - explain usage patterns and when to subclass

Proceed?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: yes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Class hierarchy + expert explanation]

VariableTracker Hierarchy:
VariableTracker (base)
├── TensorVariable (torch.Tensor)
├── ConstantVariable (constants)
├── UserDefinedObjectVariable (custom classes)
└── ... (47 subclasses total)

When to Subclass:
Create a new VariableTracker subclass when you need to
track a Python type with special symbolic execution behavior.

Decision Tree:
• Tracking torch.Tensor? Use TensorVariable
• Tracking constant? Use ConstantVariable
• Tracking new type (e.g., torch.special.BezelFunc)?
  → Create BezelFuncVariable

Key Methods to Override:
• as_python_constant() - for compile-time evaluation
• reconstruct() - for bytecode generation
• call_method() - for method calls

Example: [Shows code]
```

## How It Works

**Interactive Workflow:**
1. You describe problem naturally
2. Coordinator suggests tools/specialists
3. You approve (or modify) the suggestion
4. Coordinator executes and synthesizes results
5. You continue conversation, ask follow-ups

**You're Always in Control:**
- Approve suggestions before execution
- Redirect if routing is wrong
- Ask follow-ups and dive deeper
- Request code changes

## What's Available

### MCP Tools (torch-compile-ai)
- `parse_dynamo_guards` - Parse guard failures
- `parse_dynamo_graph` - Analyze FX graph structure
- `parse_aot_forward_graph` - Parse AOT forward graph
- `parse_aot_backward_graph` - Parse AOT backward graph
- `parse_inductor_post_grad_graph` - Parse Inductor IR
- `parse_inductor_output_code` - Analyze generated Triton code
- `parse_compiled_module` - Parse compiled module info
- `parse_fx_graph_code` - Parse FX graph Python code
- `parse_fx_graph_sizevars` - Analyze size variables
- `parse_fx_graph_cache_lookup` - Parse cache lookup logs
- `find_graph_breaks` - Find all graph breaks
- `find_recompiles` - Find recompilation triggers
- `analyze_guards` - Analyze guard failures

### MCP Tools (steering-mcp)
- `query_api_docs` - Look up API signatures
- `query_class_hierarchy` - Query class relationships
- `list_symbols` - List available symbols

### Specialists
- **dynamo-expert**: VariableTracker, guards, graph breaks
- **inductor-expert**: Lowerings, IR nodes, Triton, fusion

## When to Use vs Not Use

### Use the Coordinator When:
- You have a debugging problem
- You need to understand torch.compile behavior
- You're adding new features (lowerings, etc.)
- You want synthesized answers from multiple sources

### Use Normal Claude Code When:
- Simple code questions
- General Python/PyTorch help
- You know exactly which tool you need

## Tips

1. **Be specific about log locations** - Include full paths to torch_compile_debug
2. **Describe symptoms, not solutions** - Let coordinator figure out routing
3. **Approve or redirect** - If routing looks wrong, say "no, try X instead"
4. **Ask follow-ups** - The conversation is interactive
5. **Request code examples** - Ask "show me the code" for implementations

## Troubleshooting

### Coordinator not suggesting tools?
Make sure you loaded the coordinator prompt. If you see normal Claude responses without routing suggestions, reload the prompt.

### MCP tools not available?
Run `./setup.sh` again to configure MCP servers in `~/.claude/settings.json`.

### Steering returns no results?
PyTorch might not be indexed yet. First run of `./setup.sh` takes ~10-15 min to index.

## Next Steps

1. Run `./setup.sh` (one-time setup)
2. Start Claude Code
3. Load coordinator prompt
4. Try one of the examples above with your own torch.compile issue

## Documentation

- **INSTALLATION.md** - Detailed installation guide
- **CONTAINER_SETUP.md** - Container deployment guide
- **tests/multi-agent/test_scenarios.md** - 5 test scenarios
- **prompts/** - Coordinator and specialist prompts
