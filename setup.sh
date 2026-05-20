#!/bin/bash
# Multi-Agent System Setup Script
# Run this script on container startup to configure MCP servers and indices

set -e

echo "🚀 Setting up Multi-Agent Development System..."

# Paths
WORKSPACES="/workspaces/pytorch-devcontainers"
AI_TOOLING="$WORKSPACES/torch-compile-ai"
PYTORCH_SRC="$WORKSPACES/pytorch"
INDICES="$HOME/.acp/repos"  # Standard acp-steering location
SETTINGS="$HOME/.claude/settings.json"

# 1. Install Python packages
echo "📦 Installing Python packages..."
cd "$AI_TOOLING"
echo "   📦 Installing torch-compile-ai with dev dependencies..."
uv pip install -q -e ".[dev]"

# Install and configure pre-commit
if command -v pre-commit &> /dev/null; then
    echo "   🔧 Installing git pre-commit hooks..."
    pre-commit install --install-hooks
    echo "   ✓ Pre-commit hooks installed"

    # Run pre-commit on all files to ensure everything is clean
    echo "   🧪 Running pre-commit checks..."
    if pre-commit run --all-files 2>&1 | tail -1 | grep -q "Passed"; then
        echo "   ✓ All pre-commit checks passed"
    else
        echo "   ⚠️  Some pre-commit checks need attention (non-blocking)"
    fi
else
    echo "   ⚠️  pre-commit not available (dev dependencies may not be installed)"
fi

cd "$WORKSPACES"

# Install acp-steering-mcp from GitHub
if ! command -v acp-steering-mcp &> /dev/null; then
    echo "   📦 Installing acp-steering-mcp from GitHub..."
    if uv pip install -q "git+https://github.com/ambient-code/steering.git" 2>/dev/null; then
        echo "   ✓ acp-steering-mcp installed"
        STEERING_AVAILABLE=true
    else
        echo "   ⚠️  Failed to install acp-steering-mcp (API lookups disabled)"
        echo "      Try manually: uv pip install git+https://github.com/ambient-code/steering.git"
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
        echo "   ✓ Dynamo indexed: $(cat "$INDICES/dynamo/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*') functions"
    else
        echo "   ✓ Dynamo index exists"
    fi

    if [ ! -f "$INDICES/inductor/steering.json" ]; then
        echo "🔍 Indexing torch._inductor (this takes ~5-8 minutes)..."
        cd "$PYTORCH_SRC"
        repomap ./torch/_inductor --repo-name inductor --verbose > /dev/null 2>&1
        echo "   ✓ Inductor indexed: $(cat "$INDICES/inductor/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*') functions"
    else
        echo "   ✓ Inductor index exists"
    fi
else
    echo "   ⏭️  Skipping PyTorch indexing (steering not available)"
fi

# 4. Configure Claude settings (preferences only, MCP in .mcp.json)
echo "⚙️  Configuring Claude settings..."
mkdir -p "$HOME/.claude"

cat > "$SETTINGS" << EOF
{
  "skipDangerousModePermissionPrompt": true
}
EOF

echo "   ✓ Settings written to $SETTINGS"
echo "   ℹ️  MCP servers configured in .mcp.json (project-local)"

# 5. Create skill symlinks
echo "🔗 Creating skill symlinks..."
CLAUDE_SKILLS="$WORKSPACES/.claude/skills"
mkdir -p "$CLAUDE_SKILLS"

# Remove old symlinks if they exist
rm -f "$CLAUDE_SKILLS/compile-bisect"
rm -f "$CLAUDE_SKILLS/compile-overview"
rm -f "$CLAUDE_SKILLS/compile-trace-aot"
rm -f "$CLAUDE_SKILLS/compile-trace-dynamo"
rm -f "$CLAUDE_SKILLS/compile-trace-inductor"
rm -f "$CLAUDE_SKILLS/pytorch-dynamo"
rm -f "$CLAUDE_SKILLS/pytorch-inductor"

# Create new symlinks
ln -s "$AI_TOOLING/vertical-plugins/bisector/skills/compile-bisect" "$CLAUDE_SKILLS/compile-bisect"
ln -s "$AI_TOOLING/coordinator/skills/compile-overview" "$CLAUDE_SKILLS/compile-overview"
ln -s "$AI_TOOLING/vertical-plugins/aot-debugger/skills/compile-trace-aot" "$CLAUDE_SKILLS/compile-trace-aot"
ln -s "$AI_TOOLING/vertical-plugins/dynamo-debugger/skills/compile-trace-dynamo" "$CLAUDE_SKILLS/compile-trace-dynamo"
ln -s "$AI_TOOLING/vertical-plugins/inductor-debugger/skills/compile-trace-inductor" "$CLAUDE_SKILLS/compile-trace-inductor"
ln -s "$AI_TOOLING/vertical-plugins/dynamo-debugger/skills/pytorch-dynamo" "$CLAUDE_SKILLS/pytorch-dynamo"
ln -s "$AI_TOOLING/vertical-plugins/inductor-debugger/skills/pytorch-inductor" "$CLAUDE_SKILLS/pytorch-inductor"

echo "   ✓ Created 7 skill symlinks in $CLAUDE_SKILLS"

# 7. Verify setup
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
echo "   ✓ pre-commit: $(which pre-commit 2>/dev/null || echo 'not installed')"
echo "   ✓ Dynamo index: $DYNAMO_FUNCS functions"
echo "   ✓ Inductor index: $INDUCTOR_FUNCS functions"
echo "   ✓ Claude settings: $SETTINGS"
echo "   ✓ MCP servers: $WORKSPACES/.mcp.json"
echo "   ✓ Skills: $(ls -1 $CLAUDE_SKILLS | wc -l) symlinks created"

# 8. Test MCP servers
echo "🧪 Testing MCP servers..."

# Test torch-compile-ai server
if python "$AI_TOOLING/server.py" <<< '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":1}
{"jsonrpc":"2.0","method":"tools/list","id":2}' 2>/dev/null | grep -q "parse_graph_breaks"; then
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
echo "   • MCP Servers: debug-tracer, steering (in .mcp.json)"
echo "   • Indices: dynamo ($DYNAMO_FUNCS funcs), inductor ($INDUCTOR_FUNCS funcs)"
echo "   • Settings: $SETTINGS"
echo "   • Skills: 7 torch.compile debugging skills linked"
echo "   • Pre-commit: Ruff linter/formatter + pytest hooks enabled"
echo "   • Prompts: $AI_TOOLING/prompts/"
echo ""
echo "🚀 Next Steps:"
echo "   1. Start Claude Code"
echo "   2. Load coordinator: $AI_TOOLING/prompts/coordinator-concise.md"
echo "   3. Test scenarios: $AI_TOOLING/tests/multi-agent/test_scenarios.md"
echo ""
echo "📚 Documentation: $AI_TOOLING/docs/INSTALLATION.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
