# Installation Guide

## Quick Install

Clone and run with `--plugin-dir`:

```bash
# Clone the repository
git clone https://github.com/morrison-turnansky/torch-compile-ai.git
cd torch-compile-ai

# Start Claude Code with the plugin
claude --plugin-dir .
```

The plugin automatically (via SessionStart hook):
- ✅ Installs required dependencies (`acp-steering-mcp`)
- ✅ Indexes PyTorch modules on first use
- ✅ Configures MCP server
- ✅ Loads all 8 skills and 5 agents

## Prerequisites

1. **Claude Code** installed and configured
2. **Python environment** with `uv` available
3. **PyTorch source** (optional, for API documentation):
   - If available at `/workspaces/pytorch-devcontainers/pytorch` or `~/pytorch`, indexing happens automatically
   - Otherwise, set `PYTORCH_SRC` environment variable to PyTorch source location

## Using the Plugin

Once loaded with `claude --plugin-dir .`, skills are available:

```bash
# Use skills directly (no namespace when using --plugin-dir)
/compile-overview
/pytorch-dynamo
/compile-trace-inductor

# Or ask naturally
"How do I debug a graph break in torch.compile?"
```

## Future: Marketplace Install

Once the repo is public, installation will be even simpler:

```bash
# Add marketplace
claude plugin marketplace add https://raw.githubusercontent.com/morrison-turnansky/torch-compile-ai/main/.claude-marketplace/marketplace.json

# Install plugin
claude plugin install torch-compile-ai
```

## Verify Installation

After starting Claude Code:

```bash
# Skills should be available
# Type "/" to see autocomplete

# Check that setup ran successfully
# Look for log output about:
#   - acp-steering-mcp installation
#   - PyTorch module indexing
```

## Troubleshooting

### Plugin Not Loading

Check that you're in the right directory:
```bash
cd /path/to/torch-compile-ai
ls -la .claude-plugin/plugin.json  # Should exist
claude --plugin-dir .
```

### Skills Not Found

Verify plugin structure:
```bash
# Should show 8 skills
find skills -name "SKILL.md"

# Check plugin.json has skills path
cat .claude-plugin/plugin.json | grep skills
```

### MCP Server Errors

Ensure `acp-steering-mcp` is installed:
```bash
which acp-steering-mcp
# or
uv pip install git+https://github.com/ambient-code/steering.git
```

### Indexing Failed

Check PyTorch source is available:
```bash
ls -la $PYTORCH_SRC/torch/_dynamo
ls -la $PYTORCH_SRC/torch/_inductor
ls -la $PYTORCH_SRC/torch/_functorch
```

If PyTorch source is not available, API documentation will be limited but core functionality still works.

## Uninstall

Simply stop using `--plugin-dir`:

```bash
# Just run Claude Code normally without the flag
claude
```

Or remove from settings if you added it manually:
```bash
# Edit ~/.claude/settings.json and remove the torch-compile-ai plugin entry
```

## Updates

Pull the latest changes:

```bash
cd /path/to/torch-compile-ai
git pull origin main

# Then restart Claude Code
claude --plugin-dir .
```
