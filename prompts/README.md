# Multi-Agent Prompts

This directory contains system prompts for the multi-agent torch.compile development architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Coordinator Agent                       │
│  • Analyzes tasks and routes to specialists         │
│  • Confirms routing with user                        │
│  • Synthesizes specialist reports                    │
│  • Presents unified guidance                         │
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
│ • steering-mcp       │                  │ • dynamo-expert      │
│   API lookups        │                  │   VariableTracker,   │
│   Signatures         │                  │   guards, bytecode   │
│                      │                  │                      │
│ • torch-compile-ai   │                  │ • inductor-expert    │
│   Log parsing        │                  │   Lowerings, IR,     │
│   IR analysis        │                  │   Triton, fusion     │
│   Fusion debugging   │                  │                      │
└──────────────────────┘                  └──────────────────────┘
```

## Prompts

### coordinator-concise.md
**Purpose:** Main agent that routes tasks to specialists

**Role:**
- Analyzes user requests
- Suggests appropriate specialists (MCP tools, subagents)
- Gets user confirmation before delegating
- Queries MCP servers for lookups
- Spawns subagents for complex reasoning
- Synthesizes reports into unified guidance

**Key Features:**
- Routing decision tree (maps keywords → tools)
- User confirmation flow
- Progressive disclosure (summary → details)
- Multi-specialist coordination
- Knows all 13 torch-compile-ai tools

### dynamo-expert-concise.md
**Purpose:** Specialist for torch._dynamo stage

**Specialization:**
- VariableTracker system
- Bytecode tracing
- Guard generation
- Graph breaks

**Knowledge:** References pytorch-dynamo skill for implementation details

**Output:** Structured JSON reports with findings, guidance, code examples

### inductor-expert-concise.md
**Purpose:** Specialist for torch._inductor backend

**Specialization:**
- Lowering registration
- IR nodes (Pointwise, Reduction, etc.)
- Triton codegen
- Fusion patterns
- Performance optimization

**Knowledge:** References pytorch-inductor skill + compile-trace/INDUCTOR-STAGE.md

**Output:** Structured JSON reports with IR specifics and performance analysis

### tracing-agent-concise.md
**Purpose:** Generate torch.compile debug output from user code

**Specialization:**
- Wraps code with @torch.compile
- Sets appropriate TORCH_LOGS flags
- Executes to generate debug files
- Returns debug directory path

**Knowledge:** References compile-trace skill for stage details

**Output:** JSON with debug_dir path and files generated

## Usage

### For Coordinator Agent

Load `coordinator-concise.md` as system prompt when acting as the main agent. The coordinator:

1. Receives user task
2. Analyzes and suggests specialists (MCP tools, subagents)
3. Confirms with user
4. Delegates to specialists
5. Synthesizes reports
6. Presents guidance

Example routing:
```
User: "Add support for torch.special.bessel_j0"

Coordinator: Based on your task, I recommend:
1. query_api_docs (steering-mcp) - look up Pointwise API
2. inductor-expert - explain lowering pattern

Should I proceed?

User: yes

Coordinator: [queries MCP, spawns inductor-expert, synthesizes]
```

### For Specialist Subagents

Load the appropriate specialist prompt when spawned by the coordinator:
- `dynamo-expert-concise.md` - Dynamo stage issues
- `inductor-expert-concise.md` - Inductor stage issues
- `tracing-agent-concise.md` - Generate debug output

Specialists:
1. Receive focused task
2. Consult loaded skill documentation
3. Generate structured JSON report
4. Return to coordinator

Example:
```json
{
  "specialist": "inductor-expert",
  "task": "How to add lowering for bessel_j0",
  "confidence": "high",
  "findings": {...},
  "guidance": "...",
  "code_example": "...",
  "next_steps": [...],
  "references": [...]
}
```

## Design Principles

### Context Efficiency
**Problem:** Loading all skills (dynamo + inductor + compile-trace) bloats context
**Solution:** Specialists load only their domain skill, coordinator has no skills

**Result:**
- Coordinator: ~20KB (no skills, just routing logic)
- Dynamo expert: ~50KB (pytorch-dynamo skill only)
- Inductor expert: ~40KB (pytorch-inductor skill only)
- Total: 60-70% context reduction vs. loading all skills upfront

### Clear Responsibility Boundaries

**Coordinator:**
- Routes but doesn't implement
- Synthesizes but doesn't specialize
- Orchestrates but doesn't execute

**Specialists:**
- Deep domain knowledge
- Focused reports
- No coordination logic

**MCP Servers:**
- Stateless lookups
- Fast, cacheable
- No reasoning required

### User Agency

**Always confirm routing** - Users must approve specialist consultation
**Progressive disclosure** - Summary first, details on request
**Transparent sourcing** - Always cite which specialist provided info

## Routing Examples

### Simple Lowering
Task: "Add bessel_j0"
→ inductor-expert only

### Graph Break Debugging
Task: "Why does my custom type break?"
→ torch-compile-ai (parse logs) + dynamo-expert (explain + fix)

### Architecture Design
Task: "Design approach for torch.linalg.det"
→ dynamo-expert + inductor-expert (parallel)

### API Lookup
Task: "What are Pointwise.__init__ parameters?"
→ steering-mcp only (no subagent needed)

## Testing

See `../tests/multi-agent/` for test scenarios validating:
- Routing accuracy
- Report synthesis
- User confirmation flows
- Parallel specialist coordination
- Conflict resolution

## References

- **Plan:** `../docs/plan.md` - Full multi-agent architecture design
- **Status:** `../docs/CURRENT_STATUS.md` - Implementation progress
- **Skills:** `~/.claude/skills/pytorch-{dynamo,inductor}/` - Domain documentation loaded by specialists
