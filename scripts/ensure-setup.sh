#!/bin/bash
# Auto-setup script for torch-compile-ai plugin
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

echo "🔍 Checking torch-compile-ai dependencies..."

# 1. Check if acp-steering-mcp is installed
if ! command -v acp-steering-mcp &> /dev/null; then
    echo "📦 Installing acp-steering-mcp..."
    uv pip install $QUIET "git+https://github.com/ambient-code/steering.git" 2>/dev/null || {
        echo "⚠️  Failed to install acp-steering-mcp (API lookups will be limited)"
        exit 0  # Non-blocking
    }
fi

# 2. Check if PyTorch indices exist
NEEDS_INDEXING=false

if [ ! -f "$INDICES/dynamo/steering.json" ]; then
    NEEDS_INDEXING=true
fi

if [ "$NEEDS_INDEXING" = true ]; then
    # Check if PyTorch source is available
    # 1. Check if PYTORCH_SRC env var is set
    # 2. Try to find PyTorch in common locations
    if [ -z "$PYTORCH_SRC" ]; then
        # Try common locations
        for loc in \
            "$HOME/pytorch" \
            "$HOME/projects/pytorch" \
            "/workspaces/pytorch-devcontainers/pytorch" \
            "/workspace/pytorch" \
            "/opt/pytorch" \
            "$(pwd)/../pytorch" \
            ; do
            if [ -d "$loc/torch/_dynamo" ]; then
                PYTORCH_SRC="$loc"
                break
            fi
        done
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
        echo "   To enable API documentation, set PYTORCH_SRC environment variable"
        echo "   Example: export PYTORCH_SRC=/path/to/pytorch"
    fi
fi

echo "✅ torch-compile-ai ready!"
