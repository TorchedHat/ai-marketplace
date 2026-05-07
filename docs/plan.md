# Multi-Agent Architecture for Torch.Compile Development

# Code Quality

must be stronlgy typed. 
json can jq, python can use same linters as pytorch, ruff, flake8, pyrefly
use google no type doc strings
test driven development
avoid local imports
create a pyproject toml to track dependencies

## Problem Statement

**Current Issues:**
1. **Context bloat** - Single agent loads all skills (pytorch-dynamo + pytorch-inductor + compile-trace = 16+ files)
2. **Agent confusion** - Doesn't know which skill answers which question
3. **No parallelization** - Sequential processing, can't delegate independent work
4. **Skill awareness** - Agents struggle to navigate between skills

**Goal:** Multi-agent architecture with coordinator and specialists for better context management and task routing.

## Solution: Hybrid MCP + Subagent Architecture

```
┌─────────────────────────────────────────────────────┐
│              Coordinator Agent                       │
│  • Task analysis and routing                        │
│  • Suggests specialists (user confirms)             │
│  • Synthesizes specialist reports                   │
│  • Presents unified guidance to user                │
└─────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────────────┐
         │                                             │
         ▼                                             ▼
┌──────────────────────┐                  ┌──────────────────────┐
│   MCP Servers        │                  │   Subagent Pool      │
│   (Stateless)        │                  │   (Stateful)         │
├──────────────────────┤                  ├──────────────────────┤
│                      │                  │                      │
│ • Steering MCP       │                  │ • Dynamo Expert      │
│   - API lookups      │                  │   (pytorch-dynamo)   │
│   - Signatures       │                  │                      │
│   - Class hierarchy  │                  │ • Inductor Expert    │
│                      │                  │   (pytorch-inductor) │
│ • Debug Tracer MCP   │                  │                      │
│   - Parse TORCH_LOGS │                  │                      │
│   - Explain fusion   │                  │                      │
│   - IR search        │                  │                      │
└──────────────────────┘                  └──────────────────────┘
```

### Why This Split?

**MCP Servers (Stateless Lookups):**
- Fast, deterministic queries
- No context needed
- Cacheable responses
- Examples: "What are Pointwise.__init__ parameters?", "Why didn't nodes fuse?"

**Subagents (Stateful Reasoning):**
- Complex, multi-step tasks
- Context-aware decision making
- Needs skill documentation
- Examples: "How do I implement X?", "Design approach for feature Y"

## Components

### 1. Coordinator Agent (Main Session)

**Role:** Task analysis, routing, synthesis

**Workflow:**
1. User provides task
2. Coordinator analyzes: "This requires Inductor knowledge"
3. Suggests: "Should I consult the Inductor expert?"
4. User confirms
5. Spawns Inductor subagent OR queries MCP
6. Receives structured report
7. Synthesizes and presents to user

**Example:**
```
User: "Add support for torch.special.bessel_j0"

Coordinator:
  Analysis: New operator → Needs lowering → Inductor domain
  Suggests: "I should consult the Inductor expert for lowering patterns. Proceed?"
  User: "yes"
  → Spawns Inductor Expert subagent

Inductor Expert (isolated context):
  Loads: pytorch-inductor skill only
  Reads: COMPILE-OPERATOR-REGISTRATION.md
  Returns: {
    "file": "torch/_inductor/lowering.py",
    "pattern": "@register_lowering(aten.special.bessel_j0)",
    "ir_node": "Pointwise",
    "implementation": "def bessel_j0_lowering(x): return ops.bessel_j0(x)",
    "next_steps": ["Add FakeTensor", "Add decomposition", "Test"]
  }

Coordinator synthesizes:
  "To add bessel_j0 support:
   1. Add lowering in torch/_inductor/lowering.py using @register_lowering
   2. Use Pointwise IR node for element-wise operation
   3. Implementation: ops.bessel_j0(x)
   Would you like me to implement this?"
```

### 2. MCP Servers

#### Steering MCP (API Lookups)
**Purpose:** Fast API reference queries

**Tools:**
- `query_api_docs(symbol)` → Function signature, parameters, return type
- `query_class_hierarchy(class)` → Parent classes, subclasses
- `list_symbols(module)` → Available functions/classes

**Implementation:** Use existing steering tool from ambient-code

**Example:**
```python
# Coordinator queries:
result = query_api_docs("torch._inductor.ir.Pointwise.__init__")

# Returns:
{
  "signature": "__init__(device, dtype, inner_fn, ranges)",
  "parameters": {
    "device": "torch.device",
    "dtype": "torch.dtype",
    "inner_fn": "Callable - defines computation",
    "ranges": "List[Expr] - iteration space"
  },
  "file": "torch/_inductor/ir.py:234"
}
```

#### Debug Tracer MCP (Runtime Analysis)
**Purpose:** Parse compilation logs and explain decisions

**Tools:**
- `parse_fusion_logs(log_file)` → Fusion decisions with explanations
- `search_ir(pattern, stage)` → Find IR matching pattern
- `explain_graph_break(file, line)` → Why graph break occurred
- `trace_operation(op, input_sig)` → IR at each stage

**Implementation:** Custom MCP server

**Example:**
```python
# Coordinator queries:
result = explain_fusion("buf0_relu", "buf1_add")

# Returns:
{
  "can_fuse": true,
  "reason": "Both Pointwise with matching iteration space [10, 100]",
  "fusion_type": "vertical (producer-consumer)",
  "scheduler_code": "torch/_inductor/scheduler.py:567",
  "expected_benefit": "Eliminates intermediate buffer buf0 (4KB)"
}
```

### 3. Subagents

#### Dynamo Expert
**Loaded Skills:** pytorch-dynamo only

**Specialization:**
- VariableTracker system
- Bytecode tracing
- Guard generation
- Graph breaks

**When to Use:**
- "How do I track custom Python type?"
- "Why is this causing a graph break?"
- "How do I add opcode support?"

**Context Size:** ~50KB (just pytorch-dynamo skill)

#### Inductor Expert
**Loaded Skills:** pytorch-inductor only

**Specialization:**
- Lowering registration
- IR nodes (Pointwise, Reduction)
- Triton codegen
- Fusion patterns

**When to Use:**
- "How do I add a lowering?"
- "What IR node should I use?"
- "How do I write Triton template?"

**Context Size:** ~40KB (just pytorch-inductor skill)

## Communication Protocol

### Subagent Report Format

```json
{
  "specialist": "inductor-expert",
  "task": "How to add lowering for bessel_j0",
  "confidence": "high",
  
  "findings": {
    "file": "torch/_inductor/lowering.py",
    "pattern": "@register_lowering(aten.special.bessel_j0)",
    "ir_node_type": "Pointwise",
    "reason": "Element-wise operation with no reduction"
  },
  
  "guidance": "Use Pointwise IR node for element-wise operations. Register with @register_lowering decorator. Implementation should use ops.bessel_j0(x) for the operation.",
  
  "code_example": "...",
  
  "next_steps": [
    "Add lowering in torch/_inductor/lowering.py",
    "Add FakeTensor kernel for shape inference",
    "Test with torch.compile"
  ],
  
  "dependencies": ["FakeTensor kernel required before lowering works"],
  "references": ["COMPILE-OPERATOR-REGISTRATION.md", "COMMON-PATTERNS.md"]
}
```

### Coordinator Synthesis

Coordinator receives multiple specialist reports and:
1. **Combines findings** - Merges complementary information
2. **Resolves conflicts** - If specialists disagree, asks user
3. **Orders steps** - Sequences next_steps logically
4. **Presents summary** - High-level guidance first, details on request

## Routing Strategy

### Coordinator Decision Tree

```python
def route_task(task_description: str) -> List[Specialist]:
    """Coordinator analyzes task and suggests specialists."""
    
    specialists = []
    
    # Keyword matching
    if "VariableTracker" in task or "bytecode" in task or "graph break" in task:
        specialists.append("dynamo-expert")
    
    if "lowering" in task or "IR node" in task or "Triton" in task or "fusion" in task:
        specialists.append("inductor-expert")
    
    if "debug" in task or "why" in task.lower() or "not fusing" in task:
        specialists.append("torch-compile-ai")
    
    if "API" in task or "parameters" in task or "signature" in task:
        specialists.append("steering-mcp")
    
    # Ask user to confirm
    return confirm_routing_with_user(specialists)
```

### User Confirmation

```
Coordinator: "Based on your task 'Add bessel_j0 support', I recommend:
  1. Inductor Expert - for lowering pattern guidance
  2. Steering MCP - for API lookups if needed
  
  Should I proceed with this routing?"

User: "yes" or "just inductor" or "add debug tracer too"

Coordinator: Spawns confirmed specialists
```

## Implementation Plan

### Phase 1: MCP Servers (Week 1-2)

**1.1. Steering MCP Integration**
- ✅ Already evaluated steering tool
- Use existing `acp-steering-mcp` from ambient-code
- Configure in Claude Code settings
- Test API lookups

**Example configuration:**
```json
{
  "mcpServers": {
    "steering": {
      "command": "acp-steering-mcp",
      "env": {
        "STEERING_REPOS_PATH": "~/.acp/repos"
      }
    }
  }
}
```

**Pre-index torch.compile:**
```bash
# Run steering on torch.compile codebase
repomap ./pytorch/torch/_dynamo --repo-name dynamo --output ~/.acp/repos/dynamo
repomap ./pytorch/torch/_inductor --repo-name inductor --output ~/.acp/repos/inductor
```

**1.2. Debug Tracer MCP (New)**
- Build custom MCP server following ambient-code/mcp patterns
- Use standard `mcp` package for protocol types

**Architecture (following ambient-code/mcp pattern):**
```python
# Three-layer design
torch-compile-ai/
├── server.py       # MCP server (tool dispatch, protocol handling)
├── analyzers/      # Analysis logic (log parsing, IR indexing)
│   ├── fusion.py   # Fusion log analysis
│   ├── breaks.py   # Graph break analysis
│   └── ir.py       # IR indexing and search
└── formatters/     # Output transformation
    └── reports.py  # Structured report generation
```

**Using `mcp` package:**
- `from mcp.types import Tool, TextContent` for standard types
- Pydantic for input validation
- Server pattern from ambient-code/mcp

**Tools:**
- `parse_fusion_logs(log_content: str) -> FusionAnalysis`
  - Parse `TORCH_LOGS="fusion,schedule"` output
  - Categorize fusion decisions (success/failure reasons)
  - Return structured explanations
  
- `explain_graph_break(log_content: str, file: str, line: int) -> GraphBreakAnalysis`
  - Parse `TORCH_LOGS="graph_breaks"` output
  - Extract break location, reason, graph count
  - Return actionable fix suggestions

- `search_ir(pattern: str, stage: str) -> List[IRMatch]`
  - Index generated IR from `/tmp/torchinductor_$USER/`
  - Search by pattern, operation, stage
  - Return IR snippets with context

**Key Differences from ambient-code/mcp:**
- ✅ Use same server/formatter pattern
- ✅ Use `mcp` package types
- ✅ Use Pydantic validation
- ❌ No HTTP client layer (we parse local files, not external API)

**Deliverables:**
- `torch-compile-ai/` - MCP server implementation
- `steering-mcp/` - Integration config
- Testing harness

### Phase 2: Coordinator Prompt Engineering (Week 2)

**2.1. Coordinator System Prompt**
```markdown
You are the Coordinator for torch.compile development assistance.

Your role:
1. Analyze user tasks and determine which specialists are needed
2. Suggest routing to user for confirmation
3. Query MCP servers for lookups (steering, debug-tracer)
4. Spawn subagent specialists for complex reasoning (dynamo-expert, inductor-expert)
5. Synthesize specialist reports into unified guidance
6. Present progressive disclosure (summary first, details on request)

Available specialists:
- **dynamo-expert** (subagent): VariableTracker, bytecode, guards
- **inductor-expert** (subagent): Lowerings, IR nodes, Triton
- **steering-mcp** (MCP): API lookups, signatures
- **torch-compile-ai** (MCP): Log parsing, fusion explanations

Routing strategy:
- API lookups → steering-mcp
- "Why didn't X fuse?" → torch-compile-ai
- "How do I implement Y?" → spawn appropriate subagent
- Complex multi-domain tasks → multiple specialists

Always confirm routing with user before delegating.
```

**2.2. Specialist Prompts**

**Dynamo Expert:**
```markdown
You are a Dynamo expert loaded with pytorch-dynamo skill.
Focus: VariableTracker system, bytecode tracing, guards.

Return structured reports in this format:
{
  "specialist": "dynamo-expert",
  "task": "<original question>",
  "confidence": "high|medium|low",
  "findings": {...},
  "guidance": "...",
  "code_example": "...",
  "next_steps": [...],
  "references": [...]
}

Be concise. The coordinator will synthesize your report with others.
```

**Inductor Expert:**
```markdown
You are an Inductor expert loaded with pytorch-inductor skill.
Focus: Lowerings, IR nodes, Triton codegen, fusion.

Return structured reports in this format: {...}

Be concise. The coordinator will synthesize your report with others.
```

**Deliverables:**
- Coordinator system prompt
- Subagent system prompts
- Routing decision tree

### Phase 3: Integration & Testing (Week 3)

**3.1. Wire Up Components**
- Configure MCP servers in Claude Code
- Test coordinator → specialist flows
- Verify report synthesis

**3.2. Test Cases**

**Test 1: Simple Lowering Addition**
```
User: "Add support for torch.special.bessel_j0"

Expected Flow:
  Coordinator → Inductor Expert
  Inductor Expert → Returns lowering pattern
  Coordinator → Synthesizes and guides implementation
  (Optional) User hits issue → Debug Tracer MCP
```

**Test 2: Multi-Domain Task**
```
User: "Why does my custom type cause graph breaks?"

Expected Flow:
  Coordinator → Dynamo Expert (understand graph breaks)
  Coordinator → Debug Tracer MCP (parse actual break logs)
  Coordinator → Synthesizes: "Your type breaks because..."
```

**Test 3: Parallel Specialists**
```
User: "Design approach for torch.linalg.det support"

Expected Flow:
  Coordinator → Spawns in parallel:
    - Dynamo Expert (check if det needs special handling)
    - Inductor Expert (lowering approach)
  Coordinator → Waits for both
  Coordinator → Synthesizes unified design
```

**Deliverables:**
- Integration tests
- Example workflows
- Performance metrics (context usage, latency)

### Phase 4: Refinement (Week 4)

**4.1. Measure Effectiveness**
- Context window usage (before/after)
- Task completion accuracy
- User satisfaction

**4.2. Iterate**
- Refine routing heuristics
- Improve report synthesis
- Optimize specialist prompts

## Success Metrics

### Quantitative
- **Context reduction**: 60%+ reduction in coordinator context (no longer loads all skills)
- **Routing accuracy**: 80%+ correct specialist suggestions
- **Parallel efficiency**: 2x faster for multi-domain tasks (parallel specialists)

### Qualitative
- Coordinator doesn't get confused about which skill to reference
- Specialists provide focused, actionable guidance
- User understands which specialist is helping with what

## Key Decisions Made

### Steering Evaluation Result
- ✅ **Use steering MCP for API lookups** (fast, deterministic)
- ❌ **Don't use steering for guidance** (too generic)
- ❌ **Don't use tree-sitter** (torch.compile is pure Python, steering's AST is better)

### Architecture Choices
- ✅ **MCP for stateless** (steering, debug tracer)
- ✅ **Subagents for stateful** (dynamo expert, inductor expert)
- ✅ **Hybrid routing** (coordinator suggests, user confirms)
- ✅ **Structured reports** (enables synthesis, progressive disclosure)

### What We're NOT Building
- ❌ Tree-sitter indexing (unnecessary for pure Python)
- ❌ Custom abstraction guides (already in skills)
- ❌ Separate CompiletTalk MCP (folded into Debug Tracer MCP)

## Open Questions

1. **IR Snapshot Strategy**
   - When to index IR? (On-demand? CI only? Continuous?)
   - Storage/rotation policy for generated IR
   - How to link IR snapshots to git commits

2. **Report Format Evolution**
   - Start with JSON, iterate based on usage
   - May need different formats for different specialist types

3. **Skill Updates**
   - How to keep MCP tools in sync with skill documentation updates
   - Version coordination between skills and MCP servers

## Files to Clean Up

Once plan is finalized:
- ✅ Keep: `plan.md` (this file)
- ✅ Keep: `steering-evaluation.md` (documents why we chose MCP path)
- ❌ Delete: `tree-sitter-vs-steering.md` (tree-sitter not needed)
- ❌ Delete: `tree-sitter-prototype.py` (not building this)
- ❌ Delete: `steering-test/` and `steering-test-inductor/` (evaluation artifacts)
- ❌ Delete: `current-state.md` (info folded into this plan)
- ✅ Keep: `SUMMARY.md` (convert to implementation status tracker)

## Next Steps

1. ✅ **Finalize plan** (this document) - DONE
2. 🔜 **Build Debug Tracer MCP** (Week 1) - NEXT
   - Start with `parse_fusion_logs()` tool
   - Test with real TORCH_LOGS output
3. 🔜 **Configure Steering MCP** (Week 1)
   - Install ambient-code steering as MCP server
   - Test API queries
4. 🔜 **Design coordinator prompt** (Week 2)
   - Routing logic
   - Report synthesis patterns
5. 🔜 **Test end-to-end flow** (Week 3)
   - Simple lowering task
   - Multi-domain task
   - Parallel specialists

## Appendix: Existing Skills Inventory

### pytorch-dynamo
- COMMON-PATTERNS.md - Adding opcodes, VariableTrackers
- ARCHITECTURE.md - Dynamo internals
- DEBUGGING-GUIDE.md - Debug compilation errors
- GUARD.md - Guard system
- PYTREE-INTEGRATION.md - Pytree support
- QUICK-REFERENCE.md - Commands

### pytorch-inductor
- COMPILE-OPERATOR-REGISTRATION.md - Custom ops, lowerings
- COMMON-PATTERNS.md - Implementation patterns
- TRITON-CODEGEN.md - Triton templates
- ARCHITECTURE.md - Inductor internals

### compile-trace
- SKILL.md - Pipeline overview
- DYNAMO-STAGE.md - Dynamo debugging
- AOT-STAGE.md - AOT autograd debugging
- INDUCTOR-STAGE.md - Inductor debugging

**Total:** 16 skill files currently loaded by single agent
**After multi-agent:** Each specialist loads ~5 files, coordinator loads none
