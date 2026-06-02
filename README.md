# PyTorch AI Marketplace

A curated collection of Claude Code plugins, skills, and agents for PyTorch development. This marketplace provides AI-powered tools for debugging, development, and optimization across the PyTorch ecosystem.

## What is This?

The PyTorch AI Marketplace is a **Claude Code plugin marketplace** that provides:

- **Pre-built skills** - Ready-to-use AI capabilities for common PyTorch workflows
- **Specialized agents** - Expert AI assistants for specific PyTorch subsystems
- **MCP integrations** - Semantic search and tooling for PyTorch codebases
- **Development tools** - Meta-skills for creating your own skills and agents

This marketplace serves the broader PyTorch community by packaging AI tools that understand PyTorch internals, debugging workflows, and development patterns.

## Available Tool Collections

### torch.compile Debugging Tools

Multi-agent system for PyTorch compiler development with 10 skills and 4 specialized agents.

**Features:**
- Stage-specific debugging (Dynamo, AOT Autograd, Inductor)
- Automatic bisection for compilation failures
- Semantic API search over PyTorch compiler modules
- Implementation guidance for compiler development

👉 **[Full documentation](TORCH_COMPILE_TOOLS.md)**

### Coming Soon

- **torchvision tools** - Image processing and computer vision workflows
- **distributed training** - Multi-GPU and distributed debugging
- **performance optimization** - Profiling and optimization workflows
- **model analysis** - Architecture analysis and validation

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/morrison-turnansky/ai-marketplace.git
cd ai-marketplace

# Start Claude Code with all marketplace plugins
claude --plugin-dir .
```

### Marketplace Install (Future)

Once published:

```bash
# Add the marketplace
claude plugin marketplace add https://raw.githubusercontent.com/morrison-turnansky/ai-marketplace/main/.claude-marketplace/marketplace.json

# Install specific tool collections
claude plugin install ai-marketplace
```

### Selective Installation

To use only specific tools, configure in your `.claude/settings.json`:

```json
{
  "plugins": {
    "ai-marketplace": {
      "enabled": true,
      "source": {
        "type": "directory",
        "path": "/path/to/ai-marketplace"
      }
    }
  }
}
```

## Using Marketplace Tools

### Slash Commands

Access skills directly:
```bash
/compile-overview          # torch.compile pipeline reference
/pytorch-dynamo           # Dynamo implementation guidance
/skill-writer             # Create new skills
```

With marketplace namespace:
```bash
/ai-marketplace:compile-overview
/ai-marketplace:pytorch-dynamo
```

### Natural Language

Skills activate automatically based on context:
```
"Debug this torch.compile graph break"
"How do I implement a new Dynamo VariableTracker?"
"Bisect this compilation failure"
```

### Specialized Agents

Delegate complex tasks to expert agents:
- Use `compile-debug` agent for multi-stage compilation debugging
- Use `dynamo-expert` for Dynamo-specific questions
- Use `aot-expert` for AOT Autograd and gradient issues
- Use `inductor-expert` for lowering and codegen

## Repository Structure

```
ai-marketplace/
├── .claude-plugin/          # Plugin metadata
│   └── plugin.json          # Marketplace configuration
├── .claude-marketplace/     # Marketplace listing
│   └── marketplace.json     # Discovery and installation metadata
├── agents/                  # Specialized AI agents
│   └── *.md                 # Agent definitions
├── skills/                  # User-invocable skills
│   └── */SKILL.md          # Skill definitions
├── scripts/                 # Setup and maintenance
│   └── ensure-setup.sh      # Auto-install dependencies
├── hooks.json              # Plugin lifecycle hooks
├── settings.json           # MCP server configurations
└── pyproject.toml          # Package metadata
```

## For Plugin Developers

This repository serves as a **reference implementation** for Claude Code plugin marketplace patterns:

- **Plugin packaging** - See `.claude-plugin/plugin.json` for plugin structure
- **Marketplace listing** - See `.claude-marketplace/marketplace.json` for discovery
- **Skill development** - Use `/skill-writer` to create new skills
- **Agent development** - Use `/agent-writer` to create specialized agents
- **Auto-setup hooks** - See `hooks.json` and `scripts/ensure-setup.sh`

### Creating Your Own Tools

1. **Use the development skills:**
   ```bash
   /skill-writer    # Interactive skill creation
   /agent-writer    # Interactive agent creation
   ```

2. **Study the examples:**
   - `skills/compile-overview/` - Reference documentation skill
   - `skills/compile-bisect/` - Interactive debugging skill
   - `agents/compile-debug.md` - Multi-skill coordinator agent

3. **See the architecture guide:**
   - [REPO_ARCH.md](REPO_ARCH.md) - Plugin structure and patterns
   - [CLAUDE.md](CLAUDE.md) - Code guidelines and testing

## Documentation

- **[TORCH_COMPILE_TOOLS.md](TORCH_COMPILE_TOOLS.md)** - torch.compile debugging tools documentation
- **[REPO_ARCH.md](REPO_ARCH.md)** - Architecture and development guide
- **[CLAUDE.md](CLAUDE.md)** - Code guidelines and testing

## Contributing

We welcome contributions of new tools, skills, and agents for the PyTorch ecosystem!

**Adding new tool collections:**
1. Create a new directory for your tool collection
2. Add skills in `skills/` with descriptive SKILL.md files
3. Add specialized agents in `agents/` if needed
4. Update marketplace.json with your tool metadata
5. Submit a PR with documentation

**Improving existing tools:**
- See individual tool documentation (e.g., TORCH_COMPILE_TOOLS.md)
- Check REPO_ARCH.md for contribution guidelines
- Follow code guidelines in CLAUDE.md

## Troubleshooting

### Plugin Not Loading

```bash
cd /path/to/ai-marketplace
ls -la .claude-plugin/plugin.json  # Should exist
claude --plugin-dir .
```

### Skills Not Found

```bash
# List all available skills
find skills -name "SKILL.md"

# Check plugin configuration
cat .claude-plugin/plugin.json
```

### Tool-Specific Issues

See documentation for specific tool collections:
- torch.compile tools: [TORCH_COMPILE_TOOLS.md](TORCH_COMPILE_TOOLS.md)

## License

Part of the PyTorch devcontainer tooling. See [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/morrison-turnansky/ai-marketplace/issues)
- **Documentation**: See individual tool documentation in this repository
- **Contributing**: See [REPO_ARCH.md](REPO_ARCH.md) for guidelines
