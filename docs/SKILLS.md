# Skills Guide

## Available Skills

### Meta Skills (Workflow Guidance)

**compile-overview**
- Entry point for torch.compile debugging
- Explains pipeline: Dynamo → AOT → Inductor
- Recommends bisect-first workflow

**compile-bisect**
- Automated failure isolation
- Binary search through compilation pipeline
- Routes to stage-specific skills

### Tracing Skills (User-Level Debugging)

**compile-trace-dynamo**
- Debug Dynamo stage
- TORCH_LOGS interpretation
- Graph breaks, FX graphs, guards
- Pre-grad passes

**compile-trace-aot**
- Debug AOT Autograd stage
- Functionalization, decompositions
- Joint graphs, partitioning
- Post-grad passes

**compile-trace-inductor**
- Debug Inductor stage
- Fusion decisions, scheduling
- IR lowering, kernel generation
- Triton/C++ codegen

### Implementation Skills (Contributor-Level)

**pytorch-dynamo**
- Dynamo internals
- VariableTracker system
- Guard generation, symbolic shapes
- C++ runtime, bytecode tracing

**pytorch-aot**
- AOT/Functorch internals
- vmap, functionalization
- Joint graph construction
- Activation checkpointing

**pytorch-inductor**
- Inductor internals
- IR nodes, lowerings
- Scheduler, fusion heuristics
- Triton template system

## Using Skills

### Via Slash Command

When using `--plugin-dir`:
```bash
/compile-overview
/pytorch-dynamo
```

When installed via marketplace:
```bash
/torch-compile-ai:compile-overview
/torch-compile-ai:pytorch-dynamo
```

### Via Natural Language

Skills load automatically based on context:
```
"How do I debug a graph break?"
"Why isn't my reduction fusing?"
"Show me the VariableTracker for this case"
```

## Skill Structure

Each skill directory contains:

```
skill-name/
└── SKILL.md
    ├── --- (YAML frontmatter)
    │   name: skill-name
    │   description: Brief description
    │   ---
    └── # Skill Content
        - When to use
        - Key concepts
        - Common patterns
        - Examples
        - Troubleshooting
```

## Adding New Skills

1. Create directory in `skills/`:
```bash
mkdir skills/my-skill
```

2. Create `SKILL.md` with frontmatter:
```markdown
---
name: my-skill
description: What this skill helps with
---

# My Skill

Content here...
```

3. Restart Claude Code - auto-discovered!

## Best Practices

- Keep skills focused on one area
- Include concrete examples
- Reference actual PyTorch code locations
- Link to related skills
- Provide debugging workflows
