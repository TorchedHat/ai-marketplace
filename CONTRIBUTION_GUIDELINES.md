# Contributing to AI Marketplace

We welcome contributions of new tools, skills, agents, and plugins to the PyTorch AI Marketplace!

## Development Workflow

### Local Developement

To test a feature locally, clone the repository, checkout your branch and then
run 
```bash
claude plugin marketplace add ./ai-marketplace
claude plugin install your-plugin 
```

### Adding a New AI Tool

**Using plugin authoring tools (recommended):**
```bash
/ai-writer:skill-writer    # Create a new Claude Code skill/
/ai-writer:agent-writer    # Create a specialized agent definition
/ai-writer:plugin-writer   # Create a new plugin
```

**Manual creation:**

1. Choose target plugin (e.g., `torch-compile/` or `ai-writer/`)

2. Create directory in `<plugin>/skills/`:
```bash
mkdir torch-compile/skills/my-new-skill
```

3. Add `SKILL.md` with frontmatter:
```markdown
---
name: my-new-skill
description: Brief description
---

# Skill Content
...
```

### Adding a New Agent

Agents are specialized AI assistants that can use specific skills and call other agents.

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
# Agent prompt and instructions...
```

Place agent definitions in `<plugin>/agents/` directory. They will be auto-discovered by Claude Code.

### Adding a New Plugin

1. Create plugin directory structure:
```bash
mkdir -p new-plugin/{.claude-plugin,skills,agents,hooks,scripts}
```

2. Create `new-plugin/.claude-plugin/plugin.json`:
```json
{
  "name": "new-plugin",
  "version": "1.0.0",
  "description": "Plugin description",
  "skills": "./skills/",
  "agents": []
}
```

3. Add to `.claude-plugin/marketplace.json`:
```json
{
  "name": "new-plugin",
  "description": "Plugin description",
  "category": "development",
  "source": "./new-plugin"
}
```

## Code Guidelines

See [CLAUDE.md](CLAUDE.md) for detailed code style, type safety, and documentation requirements.
The linting requirements are enforced by the pre-commit hook.

## Design Principles

When contributing, please follow these design principles:

### 1. Multi-Plugin Marketplace
Multiple plugins in one repository using subdirectory structure. Each plugin is self-contained with its own configuration, skills, agents, and MCP servers.

### 2. Auto-Discovery
Skills and agents are discovered automatically from standard directories. No manual listing in plugin.json needed.

### 3. Portability
No hardcoded paths. Relative paths for local plugins. PyTorch source location auto-detected or configurable via `PYTORCH_SRC` environment variable.

### 4. Self-Contained
All dependencies installed automatically via SessionStart hooks. No manual setup required.

## Documentation Updates

When making structural changes or adding new plugins/skills/agents, update:
- `README.md` - User-facing documentation, installation, and troubleshooting
- `REPO_ARCH.md` - Architecture and design principles
- `CLAUDE.md` - Code guidelines and testing practices
- This file - Contributing guidelines

## Testing

Include tests for all new functionality:
- One test file per source file mimicing the file structure.
- Test classes named `Test<FunctionName>`
- For test functions start with test_.
- Test docstrings explaining what is being tested

## Pull Request Process

1. Ensure your code follows the guidelines in [CLAUDE.md](CLAUDE.md)
2. Update documentation as needed
3. Add tests for new functionality
4. Pre-commit hook requirements are satisfied.
5. Verify your changes work with the plugin authoring tools if applicable
6. Submit a pull request with a clear description of changes

## Questions?

- Open an issue: [GitHub Issues](https://github.com/TorchedHat/ai-marketplace/issues)
- See architecture details: [REPO_ARCH.md](REPO_ARCH.md)
- Review code guidelines: [CLAUDE.md](CLAUDE.md)
