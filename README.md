# Multi-Agent Development System for torch.compile

Combines debug log parsing with API documentation lookup for PyTorch compiler development.

**Components:**
- **9 MCP parsers** for torch.compile debug output
- **Steering MCP** for API documentation (dynamo + inductor)
- **Multi-agent coordinator** routing queries to specialists

## Quick Start

```bash
# 1. Run setup (one-time per container startup)
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh

# 2. Start Claude Code and load coordinator
Load prompts/coordinator-concise.md

# 3. Ask questions naturally
"Why isn't my reduction fusing with the pointwise op?"
```

**First run:** ~10-15 minutes (includes PyTorch indexing)  
**Subsequent:** ~30 seconds

## Installation

```bash
./setup.sh
```

This script:
- Installs pip packages (`acp-steering-mcp`, `torch-compile-ai`)
- Indexes PyTorch (dynamo + inductor) on first run
- Configures MCP servers in `~/.claude/settings.json`

## Usage

### Load Coordinator

```
Load prompts/coordinator-concise.md
```

### Example Queries

**Debug graph break:**
```
Why does this graph break? def fn(x): return x[x.item()]
```

**API lookup:**
```
What are the parameters for Pointwise.__init__?
```

**Performance:**
```
Parse fusion logs and explain why ops aren't fusing
```

See `tests/multi-agent/test_scenarios.md` for complete examples.

## Development

**Run tests:**
```bash
pytest tests/analyzers/ -v
```

**Update prompts:**
Edit `prompts/*.md` to adjust routing or output format

**Re-index after PyTorch updates:**
```bash
./setup.sh
```

## Documentation

- **[REPO_ARCH.md](REPO_ARCH.md)** - Architecture and design
- **[CLAUDE.md](CLAUDE.md)** - Code guidelines and testing
- **[tests/multi-agent/test_scenarios.md](tests/multi-agent/test_scenarios.md)** - Test scenarios

## Troubleshooting

**MCP server not starting:**
```bash
python server.py  # Test manually
cat ~/.claude/settings.json  # Verify config
```

**Steering returns no results:**
```bash
ls /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/  # Check indices
./setup.sh  # Re-run setup
```

**Tests failing:**
```bash
pytest tests/analyzers/ -v -s
```

## License

Part of the PyTorch devcontainer tooling.
