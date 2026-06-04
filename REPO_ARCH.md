# Repository Architecture

## Overview

Claude Code plugin for PyTorch torch.compile debugging. Provides skills, agents, and MCP servers integration for semantic API search across Dynamo, AOT Autograd, and Inductor compilation stages.

## Project Structure

```
ai-marketplace/
├── .claude-plugin/          # Plugin metadata
│   ├── plugin.json          # Plugin configuration
│   └── marketplace.json     # Marketplace discovery metadata
├── agents/                  # Specialized AI agents
│   └── *.md                 # Agent definitions
├── skills/                  # User-invocable skills
│   └── */SKILL.md          # Skill definitions
├── hooks/                   # Plugin lifecycle hooks
│   └── hooks.json          # Hook definitions
├── scripts/                 # Setup and maintenance
│   └── ensure-setup.sh      # Auto-install dependencies
├── settings.json           # MCP server configurations
└── pyproject.toml          # Package metadata
```

## Architecture

### Plugin System

**Auto-Discovery:**
- Skills in `skills/` directory are automatically discovered
- Agents in `agents/` directory are automatically discovered
- No explicit path configuration needed in plugin.json

**Auto-Setup (SessionStart Hook):**
- Installs `acp-steering-mcp` package
- Indexes PyTorch modules (dynamo, inductor, functorch)
- Runs via `scripts/ensure-setup.sh`

**Skill Naming:**
- `/torch-compile:skill-name`

**Agent Definition Format:**
```yaml
---
name: agent-name
skills:
  - skill-one
  - skill-two
callable_agents:
  - other-agent
---
# Agent prompt...
```

### MCP Integration

**Steering Server:**
- Provides semantic search over PyTorch API documentation
- Indexed modules: torch._dynamo, torch._inductor, torch._functorch
- Auto-configured via `.claude-plugin/plugin.json` mcpServers section

**Usage:**
```bash
# Skills can query API docs automatically
# Example: "What's the signature for Pointwise.__init__?"
```

## Design Principles

### 1. Direct Log Interpretation
Claude reads TORCH_LOGS output and debug files directly with skill guidance. No intermediate parsing - full context available for better analysis.

### 2. Auto-Discovery
Skills and agents are discovered automatically from standard directories. No manual configuration in plugin.json needed.

### 3. Portability
No hardcoded paths. PyTorch source location auto-detected or configurable via `PYTORCH_SRC` environment variable.

### 4. Self-Contained
All dependencies installed automatically via SessionStart hook. No manual setup required.

## Development Workflow

### Adding a New Skill

**Skill/agent development:**
```
/skill-writer - Create a new Claude Code skill
/agent-writer - Create a specialized agent definition
```


1. Create directory in `skills/`:
```bash
mkdir skills/my-new-skill
```

2. Add `SKILL.md` with frontmatter:
```markdown
---
name: my-new-skill
description: Brief description
---

# Skill Content
...
```
### Updating Documentation

Documentation files:
- `README.md` - User-facing documentation, installation, and troubleshooting
- `REPO_ARCH.md` - This file (architecture, design principles, and development)
- `CLAUDE.md` - Code guidelines and testing practices

Update these files when making structural changes or adding new skills/agents.

## References

- **Plugin Documentation**: https://code.claude.com/docs/en/plugins
- **Skill Documentation**: https://code.claude.com/docs/en/skills
- **MCP Documentation**: https://code.claude.com/docs/en/mcp
