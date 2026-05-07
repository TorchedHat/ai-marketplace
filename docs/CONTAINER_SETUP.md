# Container Setup Guide

Complete guide for portable multi-agent system deployment in containerized environments.

## Container Architecture

### Persistent Volume: `/workspaces/`

Everything in `/workspaces/` persists across container restarts:

```
/workspaces/pytorch-devcontainers/
├── ai-tooling/
│   ├── .acp-indices/              # PERSISTENT - PyTorch API indices
│   │   ├── dynamo/                # 1,208 functions, 647 classes
│   │   └── inductor/              # 2,457 functions, 1,122 classes
│   └── torch-compile-ai/          # PERSISTENT - All code
│       ├── setup.sh               # Run on every container startup
│       ├── server.py
│       ├── analyzers/
│       ├── prompts/
│       ├── tests/
│       └── docs/
└── pytorch/                       # PERSISTENT - PyTorch source
```

### Ephemeral (Recreated on Startup)

- `~/.claude/settings.json` - MCP server configuration
- Python packages installed via pip (in container environment)
- Any files in home directory outside `/workspaces/`

## Startup Sequence

### 1. Container Starts

Container environment is fresh:
- No pip packages installed
- No `~/.claude/settings.json`
- But `/workspaces/` has all code and indices from previous session

### 2. Run setup.sh

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

**What it does:**
1. Install Python packages: `uv pip install -e .` (torch-compile-ai) and `git+https://github.com/ambient-code/steering.git` (acp-steering-mcp)
2. Check if indices exist in `/workspaces/ai-tooling/.acp-indices/`
3. Index PyTorch if needed (only on first run, ~10-15 minutes)
4. Create `~/.claude/settings.json` with MCP server config
5. Verify setup and test MCP servers

**Timing:**
- First run: ~10-15 minutes (includes indexing)
- Subsequent runs: ~30 seconds (just pip + config)

### 3. Start Development

MCP servers are now configured and ready:
- `torch-compile-ai`: 13 parsers for torch.compile debug logs
- `steering-mcp`: API documentation for dynamo + inductor

## Setup Script Details

### Path Configuration

All paths in `setup.sh` use `/workspaces/` for persistence:

```bash
# Defined in setup.sh
WORKSPACES="/workspaces/pytorch-devcontainers"
AI_TOOLING="$WORKSPACES/ai-tooling/torch-compile-ai"
PYTORCH_SRC="$WORKSPACES/pytorch"
INDICES="$WORKSPACES/ai-tooling/.acp-indices"
SETTINGS="$HOME/.claude/settings.json"
```

### Indexing Logic

Indices are created only if they don't exist:

```bash
if [ ! -f "$INDICES/dynamo/steering.json" ]; then
    # Index torch._dynamo (~2-3 minutes)
    repomap ./torch/_dynamo --repo-name dynamo --verbose
    mv ~/.acp/repos/dynamo "$INDICES/"
fi
```

This means:
- **First run:** Creates indices in persistent storage
- **Subsequent runs:** Skips indexing (indices already exist)

### MCP Configuration

`~/.claude/settings.json` is recreated on every startup:

```json
{
  "skipDangerousModePermissionPrompt": true,
  "mcpServers": {
    "steering": {
      "command": "acp-steering-mcp",
      "env": {
        "STEERING_REPOS_PATH": "/workspaces/pytorch-devcontainers/ai-tooling/.acp-indices"
      }
    },
    "debug-tracer": {
      "command": "python",
      "args": [
        "/workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai/server.py"
      ],
      "cwd": "/workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai",
      "env": {
        "PYTHONPATH": "/workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai"
      }
    }
  }
}
```

**Key points:**
- `STEERING_REPOS_PATH` points to persistent indices in `/workspaces/`
- `args` and `cwd` point to persistent code in `/workspaces/`
- This config is identical on every startup

## Automated Startup

### Option 1: Manual (Recommended for Testing)

```bash
# SSH into container
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

### Option 2: Container Entrypoint

Add to container's entrypoint or startup script:

```dockerfile
# In Dockerfile or docker-compose.yml
CMD ["/bin/bash", "-c", "cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai && ./setup.sh && exec /bin/bash"]
```

### Option 3: VSCode devcontainer.json

```json
{
  "postStartCommand": "cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai && ./setup.sh"
}
```

## Verification

After running `setup.sh`, verify everything is ready:

### 1. Check Indices

```bash
ls /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/
# Should show: dynamo/ inductor/

cat /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/dynamo/steering.json | grep statistics
# Should show: "modules": ..., "functions": 1208, "classes": 647
```

### 2. Check MCP Config

```bash
cat ~/.claude/settings.json
# Should show both debug-tracer and steering MCP servers
```

### 3. Check Python Packages

```bash
which acp-steering-mcp repomap
# Should output paths to installed binaries
```

### 4. Test MCP Servers

In Claude Code:
```
User: What MCP tools are available?

Expected output:
- torch-compile-ai tools (13 parsers)
- steering-mcp tools (query_api_docs, query_class_hierarchy, list_symbols)
```

## Troubleshooting

### setup.sh Fails on Indexing

**Symptom:** `repomap` command not found or fails

**Fix:**
```bash
# Verify acp-steering-mcp installed
which acp-steering-mcp repomap

# If not, install manually
uv pip install acp-steering-mcp

# Verify PyTorch source exists
ls /workspaces/pytorch-devcontainers/pytorch/torch/_dynamo/
```

### Indices Empty or Corrupted

**Symptom:** steering.json missing or has 0 functions

**Fix:**
```bash
# Remove corrupted indices
rm -rf /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/

# Re-run setup
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

### MCP Servers Not Available in Claude Code

**Symptom:** "debug-tracer MCP server not available"

**Fix:**
```bash
# Verify settings.json exists
cat ~/.claude/settings.json

# Test debug-tracer manually
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
python server.py
# Should start and wait for input

# Restart Claude Code to reload MCP servers
```

### pip Install Fails

**Symptom:** "Could not find a version that satisfies the requirement"

**Fix:**
```bash
# Install torch-compile-ai
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
uv pip install -e .

# Install acp-steering-mcp from GitHub
uv pip install "git+https://github.com/ambient-code/steering.git"

# Or with regular pip
pip install -e .
pip install "git+https://github.com/ambient-code/steering.git"

# Check Python version (requires 3.10+)
python --version
```

## Maintenance

### Re-indexing After PyTorch Updates

If PyTorch source is updated:

```bash
# Remove old indices
rm -rf /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/

# Re-run setup (will re-index)
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai
./setup.sh
```

Or manual re-index:

```bash
cd /workspaces/pytorch-devcontainers/pytorch

# Re-index dynamo
repomap ./torch/_dynamo --repo-name dynamo --verbose
mv ~/.acp/repos/dynamo /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/

# Re-index inductor
repomap ./torch/_inductor --repo-name inductor --verbose
mv ~/.acp/repos/inductor /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/
```

### Updating Prompts

Prompts are in persistent storage, so edits survive restarts:

```bash
cd /workspaces/pytorch-devcontainers/ai-tooling/torch-compile-ai/prompts/

# Edit routing logic
vim coordinator-concise.md

# Edit specialist output format
vim dynamo-expert-concise.md
vim inductor-expert-concise.md
```

No need to re-run `setup.sh` after editing prompts.

### Monitoring Index Size

```bash
# Check index size
du -sh /workspaces/pytorch-devcontainers/ai-tooling/.acp-indices/

# Expected:
# dynamo: ~784KB
# inductor: ~1.6MB
# Total: ~2.4MB

# If too large (>10MB), re-index with reduced scope
```

## Performance

### First Run (with Indexing)

```
📦 Installing Python packages... (~30s)
🔍 Indexing torch._dynamo... (~2-3 minutes)
🔍 Indexing torch._inductor... (~5-8 minutes)
⚙️  Configuring MCP servers... (~1s)
✅ Verifying setup... (~5s)

Total: ~10-15 minutes
```

### Subsequent Runs (Indices Exist)

```
📦 Installing Python packages... (~30s)
✓ Dynamo index exists
✓ Inductor index exists
⚙️  Configuring MCP servers... (~1s)
✅ Verifying setup... (~5s)

Total: ~30-40 seconds
```

### Storage Usage

```
.acp-indices/: ~2.4MB (persistent)
Python packages: ~50MB (ephemeral, reinstalled on startup)
```

## Next Steps

After setup:

1. **Start Claude Code**
2. **Load coordinator prompt:** `prompts/coordinator-concise.md`
3. **Test routing:** Try one of 5 scenarios from `tests/multi-agent/test_scenarios.md`
4. **Measure accuracy:** Track routing suggestions and synthesis quality

## References

- **Installation Guide:** `docs/INSTALLATION.md`
- **Test Scenarios:** `tests/multi-agent/test_scenarios.md`
- **Current Status:** `docs/CURRENT_STATUS.md`
- **Setup Script:** `setup.sh`
