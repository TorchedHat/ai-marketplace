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
        echo "🔍 Indexing torch._dynamo..."
        cd "$PYTORCH_SRC"
        repomap ./torch/_dynamo --repo-name dynamo --verbose > /dev/null 2>&1
        echo "   ✓ Dynamo indexed: $(cat "$INDICES/dynamo/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*') functions"
    else
        echo "   ✓ Dynamo index exists"
    fi

    if [ ! -f "$INDICES/inductor/steering.json" ]; then
        echo "🔍 Indexing torch._inductor..."
        cd "$PYTORCH_SRC"
        repomap ./torch/_inductor --repo-name inductor --verbose > /dev/null 2>&1
        echo "   ✓ Inductor indexed: $(cat "$INDICES/inductor/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*') functions"
    else
        echo "   ✓ Inductor index exists"
    fi

    if [ ! -f "$INDICES/functorch/steering.json" ]; then
        echo "🔍 Indexing torch._functorch..."
        cd "$PYTORCH_SRC"
        repomap ./torch/_functorch --repo-name functorch --verbose > /dev/null 2>&1
        echo "   ✓ Functorch indexed: $(cat "$INDICES/functorch/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*') functions"
    else
        echo "   ✓ Functorch index exists"
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

# 5. Create symlinks
echo "🔗 Creating symlinks..."

# 5a. Symlink .mcp.json to project root
if [ -f "$AI_TOOLING/.mcp.json" ]; then
    rm -f "$WORKSPACES/.mcp.json"
    ln -s "$AI_TOOLING/.mcp.json" "$WORKSPACES/.mcp.json"
    echo "   ✓ Linked .mcp.json to project root"
else
    echo "   ⚠️  $AI_TOOLING/.mcp.json not found"
fi

# 5b. Symlink skills to .claude/skills
CLAUDE_SKILLS="$WORKSPACES/.claude/skills"
mkdir -p "$CLAUDE_SKILLS"

# Remove old AI tooling symlinks (keep skill-developer and other non-AI skills)
find "$CLAUDE_SKILLS" -type l | while read link; do
    target=$(readlink "$link")
    if [[ "$target" == *"torch-compile-ai"* ]] || [[ "$target" == *"ai-tooling"* ]]; then
        rm -f "$link"
    fi
done

# Discover and symlink all skills from vertical-plugins and coordinator to .claude/skills
SKILL_COUNT=0
for skill_dir in "$AI_TOOLING"/vertical-plugins/*/skills/* "$AI_TOOLING"/coordinator/skills/*; do
    if [ -d "$skill_dir" ] && ([ -f "$skill_dir/SKILL.md" ] || [ -f "$skill_dir/SKILL.yaml" ]); then
        skill_name=$(basename "$skill_dir")
        ln -s "$skill_dir" "$CLAUDE_SKILLS/$skill_name"
        echo "   → Linked: $skill_name"
        SKILL_COUNT=$((SKILL_COUNT + 1))
    fi
done

echo "   ✓ Created $SKILL_COUNT skill symlinks to .claude/skills"

# 5c. Symlink skills to agent-plugins (single source of truth from vertical-plugins)
echo "   🔗 Symlinking skills to agent-plugins..."
AGENT_SKILL_COUNT=0

# Skill mappings: agent-name -> skill-names
declare -A SKILL_MAPPINGS=(
    ["coordinator-agent"]="compile-overview"
    ["dynamo-debugger-agent"]="pytorch-dynamo compile-trace-dynamo"
    ["inductor-debugger-agent"]="pytorch-inductor compile-trace-inductor"
    ["aot-debugger-agent"]="pytorch-aot compile-trace-aot"
    ["bisector-agent"]="compile-bisect"
)

for agent_name in "${!SKILL_MAPPINGS[@]}"; do
    agent_skills_dir="$AI_TOOLING/agent-plugins/$agent_name/skills"
    mkdir -p "$agent_skills_dir"

    # Remove old copies/symlinks
    if [ -d "$agent_skills_dir" ]; then
        rm -rf "$agent_skills_dir"/*
    fi

    for skill_name in ${SKILL_MAPPINGS[$agent_name]}; do
        # Find skill source in vertical-plugins or coordinator
        skill_source=""
        if [ "$skill_name" = "compile-overview" ]; then
            skill_source="$AI_TOOLING/coordinator/skills/$skill_name"
        else
            # Search in vertical-plugins
            for vertical_dir in "$AI_TOOLING"/vertical-plugins/*; do
                candidate="$vertical_dir/skills/$skill_name"
                if [ -d "$candidate" ]; then
                    skill_source="$candidate"
                    break
                fi
            done
        fi

        if [ -n "$skill_source" ] && [ -d "$skill_source" ]; then
            ln -s "$skill_source" "$agent_skills_dir/$skill_name"
            AGENT_SKILL_COUNT=$((AGENT_SKILL_COUNT + 1))
        else
            echo "   ⚠️  Skill not found: $skill_name for $agent_name"
        fi
    done
done

echo "   ✓ Created $AGENT_SKILL_COUNT skill symlinks to agent-plugins"

# 6. Verify setup
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
FUNCTORCH_FUNCS=$(cat "$INDICES/functorch/steering.json" | grep -o '"functions": [0-9]*' | grep -o '[0-9]*')

echo "   ✓ Python: $(which python)"
echo "   ✓ acp-steering-mcp: $(which acp-steering-mcp)"
echo "   ✓ repomap: $(which repomap)"
echo "   ✓ pre-commit: $(which pre-commit 2>/dev/null || echo 'not installed')"
echo "   ✓ Dynamo index: $DYNAMO_FUNCS functions"
echo "   ✓ Inductor index: $INDUCTOR_FUNCS functions"
echo "   ✓ Functorch index: $FUNCTORCH_FUNCS functions"
echo "   ✓ Claude settings: $SETTINGS"
echo "   ✓ MCP servers: $WORKSPACES/.mcp.json"
echo "   ✓ Skills: $(ls -1 $CLAUDE_SKILLS | wc -l) symlinks created"

# 7. Test MCP servers
echo "🧪 Testing MCP servers..."

# Test torch-compile-ai server (import test - MCP servers are designed for long-running sessions)
if python -c "from analyzers import dynamo_parsers, aot_parsers, inductor_parsers" 2>/dev/null; then
    echo "   ✓ torch-compile-ai server validated"
else
    echo "   ❌ torch-compile-ai server imports failed"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Multi-Agent System Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Summary:"
echo "   • MCP Servers: debug-tracer, steering (in .mcp.json)"
echo "   • Indices: dynamo ($DYNAMO_FUNCS funcs), inductor ($INDUCTOR_FUNCS funcs), functorch ($FUNCTORCH_FUNCS funcs)"
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
