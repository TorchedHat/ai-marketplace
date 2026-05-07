# Multi-Agent Architecture - Status

**Last Updated:** 2026-05-07

## Summary

✅ **COMPLETE** - Portable multi-agent torch.compile development system ready for container deployment.

**Phase 1:** ✅ MCP Servers (debug-tracer + steering configured)
**Phase 2:** ✅ Coordinator & Specialist Prompts (standard + concise)  
**Phase 3:** ✅ Configuration Complete (MCP servers registered in settings.json)
**Phase 4:** ✅ Container Portability (setup.sh + pyproject.toml dependency management)

## Ready to Use

### MCP Servers ✅

**torch-compile-ai:** 13 parsers, 23 tests passing
**acp-steering-mcp:** Indexed and configured
- Dynamo: 1,208 functions, 647 classes, 23,834 lines of docs
- Inductor: 2,457 functions, 1,122 classes, 53,964 lines of docs

**Configuration:** `~/.claude/settings.json` updated

### Prompts ✅

**Use concise versions for production:**
- `prompts/coordinator-concise.md` (126 lines)
- `prompts/dynamo-expert-concise.md` (95 lines)
- `prompts/inductor-expert-concise.md` (131 lines)

**Total:** 352 lines (52% reduction vs standard)

### Testing ✅

**Test scenarios:** `tests/multi-agent/test_scenarios.md`
1. Simple lowering (inductor-expert)
2. Graph break debug (debug-tracer + dynamo-expert)
3. Multi-domain design (both specialists in parallel)
4. API lookup (steering-mcp only)
5. Fusion debug (debug-tracer + inductor-expert)

## Configuration Details

### Container-Portable Setup

**All dependencies in pyproject.toml:**
```toml
dependencies = [
    "mcp>=1.0.0",
    "acp-steering-mcp>=0.1.0",
]
```

**Automated setup script:** `setup.sh`
- Installs dependencies: `uv pip install -e .`
- Indexes PyTorch (first run only)
- Configures MCP servers in `~/.claude/settings.json`

**Persistent storage:** `/workspaces/` (code + indices)
**Ephemeral:** `~/.claude/settings.json` (recreated by setup.sh)

**MCP Servers in ~/.claude/settings.json:**
```json
{
  "mcpServers": {
    "steering": {
      "command": "acp-steering-mcp",
      "env": {"STEERING_REPOS_PATH": "/workspaces/.../ai-tooling/.acp-indices"}
    },
    "debug-tracer": {
      "command": "python",
      "args": ["/workspaces/.../ai-tooling/torch-compile-ai/server.py"],
      "cwd": "/workspaces/.../ai-tooling/torch-compile-ai"
    }
  }
}
```

**Indices:** `/workspaces/.../ai-tooling/.acp-indices/` (persistent)
- dynamo/ (784KB, 1,208 functions, 647 classes)
- inductor/ (1.6MB, 2,457 functions, 1,122 classes)

## Metrics

**Context Efficiency:**
- MCP-only: ~10KB (87% reduction)
- Single specialist: ~60KB (60% reduction)
- Multi-specialist: ~110KB (27% reduction)
- vs. All skills: ~150KB

**Prompt Efficiency:**
- Concise: 352 lines (~7KB)
- Standard: 733 lines (~15KB)
- Savings: 52%

## Container Deployment

### First Startup

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh  # ~10-15 minutes (includes indexing)
```

### Subsequent Startups

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh  # ~30 seconds (pip install + config only)
```

### Files

All setup is portable via `/workspaces/`:
- **Code:** `/workspaces/.../ai-tooling/torch-compile-ai/`
- **Indices:** `/workspaces/.../ai-tooling/.acp-indices/`
- **Dependencies:** `pyproject.toml` (installed on startup)
- **Setup:** `setup.sh` (run on startup)

## Next Steps

1. **Run setup.sh** on container startup
2. **Load coordinator prompt** (`prompts/coordinator-concise.md`)
3. **Test routing** - Try one of the 5 scenarios
4. **Measure accuracy** - Track routing suggestions
5. **Iterate** - Refine prompts based on results

## Success Criteria

- ✅ All 13 parsers implemented
- ✅ Prompts created (standard + concise)
- ✅ MCP servers configured
- ✅ PyTorch indexed (dynamo + inductor)
- ✅ Settings.json automation via setup.sh
- ✅ Dependency management via pyproject.toml
- ✅ Container portability (all data in /workspaces/)
- ✅ Automated setup script (setup.sh)
- 🎯 Routing accuracy 80%+ (ready to measure)
- 🎯 Synthesis quality (ready to validate)
