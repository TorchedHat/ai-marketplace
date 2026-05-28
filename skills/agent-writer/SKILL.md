---
name: agent-writer
description: Create and update agent definition files following Anthropic best practices. Use when writing agent files, creating subagents, designing agent systems, or troubleshooting agent discovery and delegation issues.
---

# Agent Writer

Create well-structured agent definition files for Claude Code following Anthropic's best practices for subagents and multi-agent systems.

## Quick Start

**Creating a new agent:**
1. Determine the agent's single, focused purpose
2. Choose appropriate scope (project, user, or plugin)
3. Define tool allowlist (principle of least privilege)
4. Write clear identity and workflow instructions
5. Validate frontmatter and test delegation

## When to Use This Skill

- Creating a new agent definition file
- Updating existing agent files to follow best practices
- Designing multi-agent systems with coordinators and specialists
- Troubleshooting agent discovery or delegation issues
- Converting workflows or prompts into specialized agents
- Implementing handoff protocols between agents

## Agent File Structure

Agent definitions use Markdown files with YAML frontmatter:

```markdown
---
name: agent-name
description: When to use this agent
tools:
  allowed:
    - Read
    - Grep
    - Glob
model: sonnet
---

# Agent Identity

You are a [role] specializing in [domain].

## Workflow

1. [Step 1]
2. [Step 2]
3. [Step 3]

## Deliverables

Provide output in this format:
- [Structure]

## Guardrails

**NEVER:**
- [Anti-pattern 1]

**ALWAYS:**
- [Best practice 1]
```

### File Locations

| Scope | Location | Use When |
|-------|----------|----------|
| Project | `.claude/agents/` | Agent is codebase-specific, version controlled |
| User | `~/.claude/agents/` | Personal agent used across all projects |
| Plugin | `<plugin>/agents/` | Distributing via plugin system |

**Priority:** Managed > CLI > Project > User > Plugin

## Essential Frontmatter Fields

### Required Fields

**name:**
- Lowercase letters and hyphens only
- Max 64 characters
- Must be unique across scope

```yaml
name: code-reviewer
name: torch-compile-ai:bisector-agent  # Plugin namespace
```

**description:**
- Critical for automatic delegation
- Include what it does AND when to use it
- Use specific trigger words

```yaml
# Good - specific with triggers
description: "Expert code review specialist. Proactively reviews code for quality, security, and maintainability. Use immediately after writing or modifying code."

# Bad - too vague
description: "Helper agent"
```

### Common Optional Fields

**tools:** - Principle of least privilege
```yaml
tools:
  allowed:
    - Read
    - Grep
    - Glob
```

**model:** - Strategic assignment
```yaml
model: opus      # High-stakes reviews
model: haiku     # Fast exploration
```

**skills:** - Preload context
```yaml
skills:
  - compile-bisect
  - pytorch-inductor
```

**For complete field reference, see [reference.md](reference.md)**

## Writing Effective Descriptions

**Formula:** `[What it does] + [When to use it] + [Key triggers]`

✅ **Good:**
```yaml
description: "Compiler bisector specialist for automatically isolating compilation failures. Use when debugging torch.compile errors, crashes, or incorrect output to identify which backend/subsystem fails."
```

❌ **Too vague:**
```yaml
description: "Helper agent"
```

**Tips:**
- Include specific operations ("bisect", "review", "debug")
- Mention domains ("torch.compile", "code", "SQL")
- Add context clues ("Use when...", "Use immediately after...")
- Include "use proactively" for automatic invocation

## System Prompt Structure

### Recommended Pattern

```markdown
# Agent Name

## Identity

You are a [role] specializing in [domain]. Your role is to:
- [Primary responsibility 1]
- [Primary responsibility 2]

**Scope**: [What's in scope]
**Not in scope**: [What to delegate]

## Deliverables

Return [format] in this structure:
[Expected output format]

## Workflow

1. **[Phase 1]**
   - [Steps]
   - [Expected outcome]

2. **[Phase 2]**
   - [Steps]
   - [Expected outcome]

## Guardrails

**NEVER:**
- [Anti-pattern with reason]

**ALWAYS:**
- [Best practice with reason]
```

## Tool Access Patterns

### Principle of Least Privilege

Grant **only** the tools necessary for the agent's purpose.

**Read-Only Analysis:**
```yaml
tools:
  allowed:
    - Read
    - Grep
    - Glob
```

**Code Modification:**
```yaml
tools:
  allowed:
    - Read
    - Edit
    - Write
    - Grep
    - Glob
```

### Tools Never Available to Subagents

- `Agent` (no nesting)
- `AskUserQuestion`
- `EnterPlanMode`
- `ExitPlanMode`
- `ScheduleWakeup`

## Multi-Agent Patterns

### Coordinator Pattern

Routes tasks to specialists and synthesizes responses:

```yaml
callable_agents:
  - specialist-1
  - specialist-2
parent_agent: null
```

### Specialist Pattern

Deep expertise in specific domain:

```yaml
callable_agents:
  - coordinator-agent
parent_agent: coordinator-agent
```

**For detailed patterns, see [reference.md](reference.md)**

## Context Isolation

**Critical:** Subagents start with **fresh context**.

**Subagents receive:**
- Their own system prompt
- The delegation prompt from parent
- Project CLAUDE.md files
- Preloaded skills

**Subagents do NOT receive:**
- Parent conversation history
- Parent tool call results

**Best Practice:** Include necessary context in delegation prompt.

```markdown
# Bad - assumes prior knowledge
"Continue with the approach we discussed"

# Good - provides full context
"Implement OAuth 2.0 with JWT tokens. Files: src/auth/login.py, src/auth/tokens.py. Requirements: 1hr token expiry, refresh support."
```

## Common Anti-Patterns

❌ **Vague descriptions** - "Helper agent"
✅ **Specific descriptions** - "Compiler bisector for torch.compile failures"

❌ **Overly broad scope** - One agent doing everything
✅ **Focused purpose** - Single, specific capability

❌ **Excessive tool access** - Read-only agent with Write/Edit
✅ **Least privilege** - Only necessary tools

❌ **Missing context** - "Debug the issue"
✅ **Full context** - "Debug auth failure in login.py. Error: 'Invalid token'. Stack trace: [...]"

## Validation Checklist

### Frontmatter
- [ ] `name` is lowercase, hyphens only, max 64 chars
- [ ] `description` is specific and < 1024 chars
- [ ] YAML is valid (no tabs, correct indentation)
- [ ] `tools` uses least privilege
- [ ] `model` matches task complexity

### System Prompt
- [ ] Clear identity and scope
- [ ] Explicit deliverable format
- [ ] Step-by-step workflow
- [ ] NEVER/ALWAYS guardrails
- [ ] Concrete examples

### Testing
- [ ] Description triggers automatic delegation
- [ ] Agent receives necessary context
- [ ] Agent produces expected format

## Examples

**See [examples.md](examples.md) for complete agent definitions:**
- Read-Only Specialist (code-reviewer)
- Coordinator Agent (debug-coordinator)
- Bisection Specialist (bisector-agent)

## Advanced Topics

**See [reference.md](reference.md) for:**
- Complete frontmatter field reference
- Worktree isolation patterns
- Security best practices (permission modes, hooks)
- Handoff protocols and JSON schemas
- Multi-agent coordination patterns

## Output Format

When creating an agent with this skill, I will:

1. **Ask clarifying questions:**
   - Agent's focused purpose
   - When it should be used
   - Required tools
   - Scope (project/user/plugin)
   - Multi-agent system role

2. **Create agent definition with:**
   - Valid YAML frontmatter
   - Clear identity and scope
   - Step-by-step workflow
   - Explicit deliverable format
   - NEVER/ALWAYS guardrails
   - Concrete examples

3. **Validate against requirements:**
   - Frontmatter syntax
   - Tool access (least privilege)
   - Description specificity
   - Context isolation awareness

4. **Provide testing instructions:**
   - Example trigger queries
   - Expected delegation behavior
   - Verification steps

The result will be a complete, working agent definition following all Anthropic best practices.
