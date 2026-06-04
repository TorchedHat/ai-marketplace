#!/bin/bash
# Auto-setup script for ai-marketplace plugin
# Runs on SessionStart to ensure dependencies are installed

set -e

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INDICES="$HOME/.acp/repos"

# Silent mode if not in terminal
if [ -t 1 ]; then
    QUIET=""
else
    QUIET="-q"
fi

echo "🔍 Checking ai-marketplace dependencies..."

# Debug: Show PATH and which python/pip we're using
echo "  Python: $(command -v python3 || echo 'not found')"
echo "  Pip: $(command -v pip || echo 'not found')"
echo "  UV: $(command -v uv || echo 'not found')"

# Verify PyTorch is installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo ""
    echo "❌ PyTorch is not installed in the Python environment"
    echo "   This plugin requires PyTorch to be installed."
    echo "   Install with: pip install torch"
    exit 1
fi

# 1. Check if acp-steering-mcp is installed
if ! command -v acp-steering-mcp &> /dev/null; then
    echo "📦 Installing acp-steering-mcp..."

    # Try uv first, fall back to pip
    if command -v uv &> /dev/null; then
        # Use virtual env if active, otherwise install system-wide
        if [ -n "$VIRTUAL_ENV" ]; then
            uv pip install $QUIET "git+https://github.com/ambient-code/steering.git" 2>/dev/null || {
                echo "⚠️  Failed to install acp-steering-mcp (API lookups will be limited)"
                exit 0  # Non-blocking
            }
        else
            uv pip install --system $QUIET "git+https://github.com/ambient-code/steering.git" 2>/dev/null || {
                echo "⚠️  Failed to install acp-steering-mcp (API lookups will be limited)"
                exit 0  # Non-blocking
            }
        fi
    elif command -v pip &> /dev/null; then
        pip install $QUIET "git+https://github.com/ambient-code/steering.git" 2>/dev/null || {
            echo "⚠️  Failed to install acp-steering-mcp (API lookups will be limited)"
            exit 0  # Non-blocking
        }
    else
        echo "⚠️  Neither uv nor pip found - cannot install acp-steering-mcp"
        exit 0  # Non-blocking
    fi

    # Verify installation
    if command -v acp-steering-mcp &> /dev/null; then
        echo "  ✓ acp-steering-mcp installed at: $(command -v acp-steering-mcp)"
    else
        echo "  ✗ acp-steering-mcp installation failed or not in PATH"
    fi
else
    echo "  ✓ acp-steering-mcp found at: $(command -v acp-steering-mcp)"
fi

# 2. Check if PyTorch indices exist
NEEDS_INDEXING=false

if [ ! -f "$INDICES/dynamo/steering.json" ]; then
    NEEDS_INDEXING=true
fi

if [ "$NEEDS_INDEXING" = true ]; then
    # Resolve PyTorch source:
    # 1. PYTORCH_PATH env var
    # 2. Detect from Python environment
    PYTORCH_SRC=""

    if [[ -n "${PYTORCH_PATH:-}" && -d "${PYTORCH_PATH}" ]]; then
        PYTORCH_SRC="$PYTORCH_PATH"
    else
        # Try to detect PyTorch from Python environment
        TORCH_LOCATION=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null || echo "")
        if [[ -n "$TORCH_LOCATION" && -d "$TORCH_LOCATION/_dynamo" ]]; then
            # Found development PyTorch source
            PYTORCH_SRC="$(dirname "$TORCH_LOCATION")"
        fi
    fi

    if [ -n "$PYTORCH_SRC" ] && [ -d "$PYTORCH_SRC/torch/_dynamo" ]; then
        echo "🔍 Indexing PyTorch modules (one-time setup)..."
        echo "   Using PyTorch at: $PYTORCH_SRC"
        mkdir -p "$INDICES"

        cd "$PYTORCH_SRC"

        # Index Dynamo
        if [ ! -f "$INDICES/dynamo/steering.json" ]; then
            repomap ./torch/_dynamo --repo-name dynamo --verbose > /dev/null 2>&1 && \
            echo "   ✓ Dynamo indexed" || echo "   ⚠️  Dynamo indexing failed"
        fi

        # Index Inductor
        if [ ! -f "$INDICES/inductor/steering.json" ]; then
            repomap ./torch/_inductor --repo-name inductor --verbose > /dev/null 2>&1 && \
            echo "   ✓ Inductor indexed" || echo "   ⚠️  Inductor indexing failed"
        fi

        # Index Functorch
        if [ ! -f "$INDICES/functorch/steering.json" ]; then
            repomap ./torch/_functorch --repo-name functorch --verbose > /dev/null 2>&1 && \
            echo "   ✓ Functorch indexed" || echo "   ⚠️  Functorch indexing failed"
        fi
    else
        echo "⚠️  PyTorch source not found - API lookups will be limited"
        echo "   To enable API documentation, set PYTORCH_PATH:"
        echo "     export PYTORCH_PATH=/path/to/pytorch"
    fi
fi

echo "✅ ai-marketplace ready!"

# 3. Verify steering MCP server can see the indices
# Note: MCP servers cache the index at startup, so if repos were just created,
# the server needs to restart to pick them up
if [ "$NEEDS_INDEXING" = true ]; then
    echo ""
    echo "📝 Note: Steering repositories were just created."
    echo "   The MCP server may need to restart to pick them up."
    echo "   If steering queries return 'No repositories found', restart Claude Code."
fi
