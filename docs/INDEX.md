# Documentation Index

User-facing documentation for the multi-agent torch.compile development system.

## Start Here

**[../GETTING_STARTED.md](../GETTING_STARTED.md)** - Interactive examples and workflow
- How to use the coordinator
- Real debugging session examples
- Tips and troubleshooting

## Setup

1. **[CONTAINER_SETUP.md](CONTAINER_SETUP.md)** - Container deployment guide
   - Persistent vs ephemeral storage
   - Automated setup script
   - Verification and troubleshooting

2. **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation guide
   - Manual step-by-step setup
   - Python dependencies
   - MCP server configuration
   - PyTorch indexing

## Reference

3. **[CURRENT_STATUS.md](CURRENT_STATUS.md)** - Implementation status
   - What's complete
   - Configuration details
   - Performance metrics
   - Next steps

4. **[PROMPT_COMPARISON.md](PROMPT_COMPARISON.md)** - Prompt optimization analysis
   - Standard vs concise versions
   - Size reduction metrics
   - Context efficiency

5. **[plan.md](plan.md)** - Original architecture plan
   - Design rationale
   - Multi-agent architecture
   - Tool selection decisions

## Quick Reference

### Run Setup (Container)

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

**First run:** ~10-15 minutes (includes indexing)  
**Subsequent:** ~30 seconds (deps + config only)

### Usage

1. Run `setup.sh` on container startup
2. Start Claude Code
3. Load coordinator: `prompts/coordinator-concise.md`
4. Try test scenarios: `tests/multi-agent/test_scenarios.md`

### Files

- **Setup:** `setup.sh`
- **Dependencies:** `pyproject.toml`
- **MCP Server:** `server.py`
- **Parsers:** `analyzers/` (13 implementations)
- **Prompts:** `prompts/` (coordinator + specialists)
- **Tests:** `tests/analyzers/` (23 unit tests)

### Support

- Issues: File in repository issue tracker
- Development: See `../CLAUDE.md` for code guidelines
- Examples: See `../tests/multi-agent/test_scenarios.md`
