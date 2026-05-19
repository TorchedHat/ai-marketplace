# Coordinator

Orchestrates torch.compile debugging workflow by routing to appropriate vertical specialists.

## Skills

- **`compile-overview/`** - Meta-skill providing pipeline overview
  - Complete compilation pipeline diagram (Dynamo → AOT → Inductor)
  - Bisect-first workflow recommendation
  - Symptom-to-stage mapping
  - Routing guidance to vertical-specific skills

## Prompts

- **`coordinator.md`** - Main orchestration agent
  - Analyzes user debugging tasks
  - Suggests appropriate specialists (bisector, stage-specific experts)
  - Routes to MCP tools when needed
  - Synthesizes specialist reports into unified guidance

## Workflow

1. **Receive** user compilation issue
2. **Analyze** - Is it a failure or exploration?
3. **Bisect** (if failure) - Guide user through compiler bisector
4. **Route** based on bisect result:
   - `backend='eager'` → Dynamo vertical
   - `backend='aot_*'` → AOT vertical
   - `backend='inductor'` → Inductor vertical
5. **Synthesize** - Combine findings from specialists
6. **Present** - Unified guidance with file:line references

## MCP Tools

Uses both MCP servers:
- **steering-mcp** - API documentation lookups across all PyTorch modules
- **torch-compile-ai** (optional) - 9 parsers for debug output (can be disabled)

## Related Components

Routes to all verticals:
- **bisector/** - Automated failure isolation
- **dynamo-debugger/** - Dynamo stage issues
- **aot-debugger/** - AOT stage issues
- **inductor-debugger/** - Inductor stage issues

## Progressive Disclosure

- **Toolkit mode**: User manually loads skills and uses MCP tools
- **Orchestrated mode**: Coordinator loads skills and calls tools automatically
