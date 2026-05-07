#!/bin/bash
# Multi-Agent System Setup Script
# Run this script on container startup to configure MCP servers and indices

set -e

echo "🚀 Setting up Multi-Agent Development System..."

# Paths (everything in /workspaces/ to persist)
WORKSPACES="/workspaces/pytorch-devcontainers"
AI_TOOLING="$WORKSPACES/ai-tooling/torch-compile-ai"
PYTORCH_SRC="$WORKSPACES/pytorch"
INDICES="$WORKSPACES/ai-tooling/.acp-indices"
SETTINGS="$HOME/.claude/settings.json"

# 1. Install Python packages
echo "📦 Installing Python packages..."
uv pip install -q -e "$AI_TOOLING"

# Install acp-steering-mcp from GitHub
if ! command -v acp-steering-mcp &> /dev/null; then
    echo "   📦 Installing acp-steering-mcp from GitHub..."
    if uv pip install -q "git+https://github.com/ambient-code/steering.git" 2>/dev/null; then
        echo "   ✓ acp-steering-mcp installed"
        STEERING_AVAILABLE=true
    else
        echo "   ⚠️  Failed to install acp-steering-mcp (API lookups disabled)"
        echo "      Try manually: pip install git+https://github.com/ambient-code/steering.git"
        STEERING_AVAILABLE=false
    fi
else
    echo "   ✓ acp-steering-mcp already installed"
    STEERING_AVAILABLE=true
fi

# 2. Create indices directory
echo "📁 Creating indices directory..."
mkdir -p "$INDICES"

# 3. Index PyTorch modules (if steering is available and not already indexed)
if [ "$STEERING_AVAILABLE" = true ]; then
    if [ ! -f "$INDICES/dynamo/steering.json" ]; then
        echo "🔍 Indexing torch._dynamo (this takes ~2-3 minutes)..."
        cd "$PYTORCH_SRC"
        repomap ./torch/_dynamo --repo-name dynamo --verbose > /dev/null 2>&1
        mv ~/.acp/repos/dynamo "$INDICES/"
        echo "   ✓ Dynamo indexed: $(cat "$INDICES/dynamo/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*') functions"
    else
        echo "   ✓ Dynamo index exists"
    fi

    if [ ! -f "$INDICES/inductor/steering.json" ]; then
        echo "🔍 Indexing torch._inductor (this takes ~5-8 minutes)..."
        cd "$PYTORCH_SRC"
        repomap ./torch/_inductor --repo-name inductor --verbose > /dev/null 2>&1
        mv ~/.acp/repos/inductor "$INDICES/"
        echo "   ✓ Inductor indexed: $(cat "$INDICES/inductor/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*') functions"
    else
        echo "   ✓ Inductor index exists"
    fi
else
    echo "   ⏭️  Skipping PyTorch indexing (steering not available)"
fi

# 4. Configure MCP servers
echo "⚙️  Configuring MCP servers..."
mkdir -p "$HOME/.claude"

# Build settings.json with conditional steering
if [ "$STEERING_AVAILABLE" = true ]; then
    cat > "$SETTINGS" << EOF
{
  "skipDangerousModePermissionPrompt": true,
  "mcpServers": {
    "steering": {
      "command": "acp-steering-mcp",
      "env": {
        "STEERING_REPOS_PATH": "$INDICES"
      }
    },
    "debug-tracer": {
      "command": "python",
      "args": [
        "$AI_TOOLING/server.py"
      ],
      "cwd": "$AI_TOOLING",
      "env": {
        "PYTHONPATH": "$AI_TOOLING"
      }
    }
  }
}
EOF
else
    cat > "$SETTINGS" << EOF
{
  "skipDangerousModePermissionPrompt": true,
  "mcpServers": {
    "debug-tracer": {
      "command": "python",
      "args": [
        "$AI_TOOLING/server.py"
      ],
      "cwd": "$AI_TOOLING",
      "env": {
        "PYTHONPATH": "$AI_TOOLING"
      }
    }
  }
}
EOF
fi

echo "   ✓ Settings written to $SETTINGS"

# 5. Verify setup
echo "✅ Verifying setup..."

# Check Python packages
if ! command -v acp-steering-mcp &> /dev/null; then
    echo "   ❌ acp-steering-mcp not installed"
    exit 1
fi

if ! command -v repomap &> /dev/null; then
    echo "   ❌ repomap not installed"
    exit 1
fi

# Check indices
DYNAMO_FUNCS=$(cat "$INDICES/dynamo/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*')
INDUCTOR_FUNCS=$(cat "$INDICES/inductor/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*')

echo "   ✓ acp-steering-mcp: $(which acp-steering-mcp)"
echo "   ✓ repomap: $(which repomap)"
echo "   ✓ Dynamo index: $DYNAMO_FUNCS functions"
echo "   ✓ Inductor index: $INDUCTOR_FUNCS functions"
echo "   ✓ MCP config: $SETTINGS"

# 6. Test MCP servers
echo "🧪 Testing MCP servers..."

# Test debug-tracer server
if python "$AI_TOOLING/server.py" <<< '{"jsonrpc":"2.0","method":"tools/list","id":1}' 2>/dev/null | grep -q "parse_dynamo_guards"; then
    echo "   ✓ torch-compile-ai responding"
else
    echo "   ⚠️  torch-compile-ai not responding (might be expected in non-interactive mode)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Multi-Agent System Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Summary:"
echo "   • MCP Servers: debug-tracer, steering"
echo "   • Indices: dynamo ($DYNAMO_FUNCS funcs), inductor ($INDUCTOR_FUNCS funcs)"
echo "   • Config: $SETTINGS"
echo "   • Prompts: $AI_TOOLING/prompts/"
echo ""
echo "🚀 Next Steps:"
echo "   1. Start Claude Code"
echo "   2. Load coordinator: $AI_TOOLING/prompts/coordinator-concise.md"
echo "   3. Test scenarios: $AI_TOOLING/tests/multi-agent/test_scenarios.md"
echo ""
echo "📚 Documentation: $AI_TOOLING/docs/INSTALLATION.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
