# Agents Guide

## Available Agents

### coordinator
**Role:** Orchestrates debugging workflow
**Skills:** compile-overview
**Purpose:** Routes tasks to specialist agents

Routes to:
- dynamo-expert (graph breaks, guards)
- aot-expert (functionalization, partitioning)
- inductor-expert (fusion, codegen)
- bisector (automated isolation)

### dynamo-expert
**Role:** Dynamo debugging specialist
**Skills:** pytorch-dynamo, compile-trace-dynamo
**Purpose:** Debug bytecode capture and FX graph construction

Handles:
- Graph break diagnosis
- VariableTracker issues
- Guard generation
- Symbolic shapes

### aot-expert
**Role:** AOT Autograd specialist
**Skills:** pytorch-aot, compile-trace-aot
**Purpose:** Debug functionalization and gradient computation

Handles:
- Functionalization errors
- Decomposition issues
- Joint graph problems
- Partitioning failures

### inductor-expert
**Role:** Inductor compiler specialist
**Skills:** pytorch-inductor, compile-trace-inductor
**Purpose:** Debug lowering and kernel generation

Handles:
- Fusion decisions
- Scheduling issues
- Triton codegen
- Memory optimization

### bisector
**Role:** Automated failure isolation
**Skills:** compile-bisect
**Purpose:** Binary search to find failing operation

Workflow:
1. Tries backends: eager → aot_eager → inductor
2. Identifies exact failing stage
3. Routes to appropriate expert

## Agent Structure

Each agent has:

```yaml
---
name: agent-name
skills:
  - skill-one
  - skill-two
callable_agents:
  - other-agent
parent_agent: coordinator-agent
---

# Agent Identity
Persona and scope...

# Deliverables
Expected output format...

# Workflow
Step-by-step process...

# Guardrails
NEVER/ALWAYS constraints...
```

## Using Agents

Agents are invoked automatically by Claude when delegating work:
- Claude uses Agent tool to spawn specialists
- Agents load their skills automatically
- Agents can call other agents per callable_agents list

## Adding New Agents

1. Create file in `agents/`:
```bash
touch agents/my-agent.md
```

2. Add frontmatter and content:
```markdown
---
name: my-agent
skills:
  - relevant-skill
callable_agents:
  - other-agent
---

# Agent Prompt
...
```

3. Add to plugin.json agents array:
```json
"agents": [
  ...
  "./agents/my-agent.md"
]
```

4. Restart Claude Code

## Best Practices

- Keep agents focused on one domain
- Declare all required skills
- Define clear callable_agents boundaries
- Provide structured output format
- Include workflow steps
- Set clear guardrails
