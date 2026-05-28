# Agent Writer - Complete Reference

Detailed documentation for all agent frontmatter fields, advanced patterns, and best practices.

## Complete Frontmatter Field Reference

### Required Fields

**name:**
- Unique identifier using lowercase letters and hyphens
- Max 64 characters
- Used as agent type in tool calls and hooks
- Must be unique across scope

```yaml
name: code-reviewer
name: torch-compile-ai:bisector-agent  # Plugin namespace
```

**description:**
- Natural language description of when to use this agent
- Critical for automatic delegation
- Max 1024 characters
- Should be specific and action-oriented
- Include "use proactively" for automatic invocation

```yaml
description: "Expert code review specialist. Proactively reviews code for quality, security, and maintainability. Use immediately after writing or modifying code."
```

### Optional Fields

**tools:**
- Allowlist or denylist of tools (principle of least privilege)
- Use `allowed` for allowlist, `denied` for denylist

```yaml
# Allowlist approach (recommended)
tools:
  allowed:
    - Read
    - Grep
    - Glob

# Denylist approach
tools:
  denied:
    - Write
    - Edit
```

**model:**
- Model override: `sonnet`, `opus`, `haiku`, `inherit`
- Strategic assignment based on task complexity

```yaml
model: opus      # High-stakes security reviews
model: haiku     # Fast exploration
model: inherit   # Use parent's model
```

**permissionMode:**
- `default`: Standard prompting
- `acceptEdits`: Auto-accept file edits in working directory
- `auto`: AI classifier reviews commands
- `dontAsk`: Denies all prompts
- `bypassPermissions`: **DANGEROUS** - skip all checks

```yaml
permissionMode: acceptEdits  # Safe for code modification agents
```

**skills:**
- Preload specific skills into agent context

```yaml
skills:
  - compile-bisect
  - pytorch-inductor
```

**callable_agents:**
- List of agents this agent can delegate to
- Used for validation in multi-agent systems

```yaml
callable_agents:
  - dynamo-expert-agent
  - inductor-expert-agent
  - bisector-agent
```

**parent_agent:**
- Indicates which agent typically calls this one
- Documentation only, not enforced

```yaml
parent_agent: coordinator-agent
```

**maxTurns:**
- Limit conversation length

```yaml
maxTurns: 10
```

**memory:**
- Enable persistent memory: `user`, `project`, `local`

```yaml
memory: project  # Shareable via version control
```

**color:**
- UI display color: `red`, `blue`, `green`, `yellow`, `purple`, `orange`, `pink`, `cyan`

```yaml
color: blue
```

**isolation:**
- `worktree`: Run agent in isolated git worktree (temporary branch)

```yaml
isolation: worktree
```

## Worktree Isolation Pattern

When using `isolation: "worktree"`, the agent runs in a temporary git worktree with its own working directory and branch.

### Critical Considerations

**Files created in worktree stay in worktree:**
- Any files the agent creates exist only in the temporary worktree directory
- They do NOT appear in the parent's main working directory
- Must explicitly transfer needed files before exiting

### Proper Handling Pattern

```markdown
## Workflow

1. **Run Analysis/Generation** (in isolated worktree)
   - Create analysis files
   - Generate reports
   - Run tests or builds

2. **Transfer Results to Main Directory**
   - **Option 1**: Include file contents in deliverable output
     - Parent can recreate files from agent's response
   - **Option 2**: Use Read + explicit file recreation
     - Agent reads created files
     - Agent returns content in structured format
     - Parent recreates files in main working directory
   - **Option 3**: Note file paths for manual retrieval
     - Agent reports where files are located
     - User can access worktree directory if needed

3. **Exit Worktree**
   - Use ExitWorktree with appropriate action
   - `keep`: Preserve worktree for later access
   - `remove`: Clean up temporary worktree

## Deliverables

When running in worktree isolation, include file contents in response:

\`\`\`markdown
## Analysis Results

**Files Created** (in worktree):
- `/path/to/worktree/analysis.md`
- `/path/to/worktree/report.txt`

**File Contents**:

### analysis.md
\`\`\`markdown
[Full content here so parent can recreate]
\`\`\`

### report.txt
\`\`\`
[Full content here so parent can recreate]
\`\`\`
\`\`\`
```

### Anti-Pattern vs Good Pattern

❌ **Bad - Agent creates files but doesn't transfer them:**

```markdown
# Agent output
Created analysis at /tmp/worktree_abc123/analysis.md
Exiting worktree...

# Parent can't access the file - it's gone!
```

✅ **Good - Agent includes file contents:**

```markdown
# Agent output
Created analysis at /tmp/worktree_abc123/analysis.md

## Analysis Contents
\`\`\`markdown
[Full analysis content here]
\`\`\`

Exiting worktree...

# Parent can recreate analysis.md in main directory from this content
```

### When to Use Worktree Isolation

**Use for:**
- Agents that need clean git state
- Experimental changes that shouldn't affect main tree
- Bisection or testing workflows
- Parallel exploration without conflicts

**Don't use for:**
- Simple read-only analysis (no need for isolation)
- Agents that need to modify main working tree directly
- Short-lived queries that don't change files

## Multi-Agent System Patterns

### Coordinator Pattern

**Use Case:** Route tasks to specialized agents and synthesize responses.

**Frontmatter:**
```yaml
---
name: coordinator-agent
description: Orchestrates debugging by routing tasks to specialized agents
tools:
  allowed:
    - Read
    - Bash
skills:
  - compile-overview
callable_agents:
  - dynamo-expert-agent
  - inductor-expert-agent
  - bisector-agent
parent_agent: null
---
```

**System Prompt Pattern:**
```markdown
## Routing Decision

Based on task type:
- Graph breaks → dynamo-expert-agent
- Fusion issues → inductor-expert-agent
- Compilation failures → bisector-agent first

## Synthesis

Combine findings from specialists into unified guidance:
1. Summary (2-3 sentences)
2. Actionable steps
3. Code examples
4. Source attribution
```

### Specialist Pattern

**Use Case:** Deep expertise in specific domain.

**Frontmatter:**
```yaml
---
name: bisector-agent
description: Compiler bisector specialist for automatically isolating compilation failures
tools:
  allowed:
    - Read
    - Bash
  denied:
    - Write
    - Edit
skills:
  - compile-bisect
callable_agents:
  - coordinator-agent
  - dynamo-expert-agent
  - inductor-expert-agent
parent_agent: coordinator-agent
---
```

**System Prompt Pattern:**
```markdown
## Identity

You are a **compiler bisector specialist**. Your role is to:
- Guide users through automated bisection workflows
- Execute bisector to isolate failing backend/subsystem
- Interpret bisection results
- Route to appropriate stage expert based on findings

**Scope**: Bisection orchestration and result interpretation

**Not in scope**: Deep analysis of specific stages (delegate to experts)
```

### Parallel Exploration Pattern

**Use Case:** Multiple independent investigations.

Create specialized agents for different perspectives:
- UX perspective
- Technical architecture
- Security review
- Performance analysis

Spawn multiple agents in parallel and synthesize findings.

## Handoff Protocols

### Parent to Subagent Delegation

**Automatic Delegation:**
- Claude matches task to agent based on `description`
- Works best with clear, specific descriptions

**Explicit Delegation:**
```
Use the code-reviewer agent to check the authentication module
```

**@-mention Pattern:**
```
@"code-reviewer (agent)" look at the auth changes
```

### Context Isolation Details

**Subagents receive:**
- Their own system prompt
- The delegation prompt from parent
- Project CLAUDE.md files
- Git status snapshot
- Preloaded skills (if specified in `skills` field)

**Subagents do NOT receive:**
- Parent conversation history
- Parent tool call results
- Parent system prompt
- Skills not explicitly preloaded

### Structured Handoff with JSON

For complex multi-agent systems, emit structured handoff requests:

```markdown
## Handoff Protocol

When delegating to specialists, emit handoff_request JSON:

\`\`\`json
{
  "type": "handoff_request",
  "from_agent": "coordinator-agent",
  "to_agent": "bisector-agent",
  "task": {
    "type": "bisect_failure",
    "issue": "segfault during compilation",
    "context": {
      "repro_script": "repro.py",
      "error_message": "Segmentation fault"
    }
  },
  "expected_deliverable": "bisection_result",
  "priority": "high"
}
\`\`\`

This enables validation and tracking in orchestration layers.
```

## Security Best Practices

### Permission Modes

**Progressive Security Levels:**

```yaml
# Most restrictive - denies prompts
permissionMode: dontAsk

# Standard - prompts for approval
permissionMode: default

# Auto-accepts edits in working directory
permissionMode: acceptEdits

# AI classifier reviews commands
permissionMode: auto

# DANGEROUS - Skip all checks
permissionMode: bypassPermissions  # Use with extreme caution!
```

### Validation with Hooks

Block dangerous operations before execution:

```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-command.sh"
```

**Example validation script:**
```bash
#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Block dangerous operations
if echo "$COMMAND" | grep -iE '\b(rm -rf|DROP|DELETE)\b' > /dev/null; then
  echo "Blocked: Dangerous operation not allowed" >&2
  exit 2  # Exit code 2 blocks the operation
fi

exit 0
```

### Plugin Restrictions

**Important:** Plugin agents cannot use:
- `hooks`
- `mcpServers`
- `permissionMode`

**Workaround:** Copy plugin agent to `.claude/agents/` or `~/.claude/agents/` to enable these features.

## Common Tool Patterns

### Read-Only Analysis
```yaml
tools:
  allowed:
    - Read
    - Grep
    - Glob
```

### Test Execution
```yaml
tools:
  allowed:
    - Bash
    - Read
    - Grep
```

### Code Modification
```yaml
tools:
  allowed:
    - Read
    - Edit
    - Write
    - Grep
    - Glob
```

### Full Access
```yaml
# Omit tools field to inherit all available tools
```

## System Prompt Best Practices

### Recommended Structure

```markdown
# Agent Name

## Identity

You are a [role] specializing in [domain]. Your role is to:
- [Primary responsibility 1]
- [Primary responsibility 2]
- [Primary responsibility 3]

**Scope**: [What's in scope]

**Not in scope**: [What to delegate or avoid]

## Deliverables

Return [format type] in this format:

\`\`\`markdown
## Summary
<2-3 sentences>

## Implementation
1. <actionable step with file:line>
2. <actionable step with file:line>

## Code
<code example if applicable>

---
*Sources: <attribution>*
\`\`\`

## Workflow

1. **[Phase 1 Name]**
   - [Step 1]
   - [Step 2]
   - [Expected outcome]

2. **[Phase 2 Name]**
   - [Step 1]
   - [Step 2]
   - [Expected outcome]

## Guardrails

**NEVER:**
- [Anti-pattern 1 with reason]
- [Anti-pattern 2 with reason]

**ALWAYS:**
- [Best practice 1 with reason]
- [Best practice 2 with reason]

## Examples

### Example 1: [Scenario]

**User**: "[Query]"

**Workflow**:
1. [Step]
2. [Step]

**Response**:
\`\`\`markdown
[Expected output]
\`\`\`

## Handoff Protocol

When delegating to [other agent]:
- Provide [required context]
- Include [necessary details]
- Expect [deliverable format]
```

## Anti-Patterns to Avoid

### ❌ Vague Descriptions

```yaml
# Bad
description: "Helper agent"
description: "Does debugging stuff"

# Good
description: "Compiler bisector specialist for automatically isolating compilation failures. Use when debugging torch.compile errors, crashes, or incorrect output."
```

### ❌ Overly Broad Scope

```yaml
# Bad - One agent doing everything
name: do-everything
description: "Handles all debugging tasks"

# Good - Focused, single-purpose
name: bisector-agent
description: "Isolates compilation failures through automated bisection"
```

### ❌ Excessive Tool Access

```yaml
# Bad - Read-only agent with write access
name: code-analyzer
description: "Analyzes code without modifying"
tools:
  allowed:
    - Read
    - Write    # Not needed!
    - Edit     # Not needed!

# Good
tools:
  allowed:
    - Read
    - Grep
    - Glob
```

### ❌ Missing Context in Delegation

```python
# Bad - No context provided
"Debug the authentication issue"

# Good - Full context
"Debug authentication failure in src/auth/login.py. Error: 'Invalid token signature'. Recent changes: Added JWT validation. Stack trace: [...]"
```

### ❌ Nested Subagents

```yaml
# Bad - Subagent trying to spawn subagents
name: coordinator
tools:
  allowed:
    - Agent     # Don't include Agent in subagent tools!
    - Read
```

### ❌ Undefined Output Format

```markdown
# Bad - No structure
"Review the code and report issues"

# Good - Clear structure
"Provide feedback organized by priority:
- Critical issues (must fix)
- Warnings (should fix)
- Suggestions (consider improving)

Include file:line references and code examples for each issue."
```

## Testing Best Practices

### Step 1: Create Test Scenarios

Prepare test queries that should trigger your agent:

```
# For bisector-agent
"My model crashes during compilation with a segfault"
"torch.compile fails with an error about invalid lowering"

# For code-reviewer
"Can you review the changes I just made?"
"Check this code for security issues"
```

### Step 2: Test Automatic Delegation

Ask relevant questions and verify:
1. Claude automatically delegates to your agent
2. Agent receives appropriate context
3. Agent produces expected deliverable format

### Step 3: Test Multi-Agent Coordination

If your agent is part of a multi-agent system:
1. Verify handoff protocols work
2. Check context propagation
3. Validate routing decisions
4. Confirm synthesis of results

### Step 4: Debug Issues

**Agent doesn't activate:**
- Make description more specific
- Add trigger words users would say
- Include "use proactively" in description

**Agent lacks context:**
- Check what's included in delegation prompt
- Verify necessary skills are preloaded
- Ensure CLAUDE.md files are accessible

**Agent produces wrong output:**
- Review deliverable format specification
- Check workflow instructions clarity
- Verify examples match expected behavior
