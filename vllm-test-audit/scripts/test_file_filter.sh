#!/usr/bin/env bash
set -euo pipefail

# Candidate Finder for vLLM Test Oracle Auditor
#
# Finds test files from given paths. Pattern analysis is left to the agent.
#
# Usage:
#   candidate_finder.sh <directory-or-file> [directory-or-file...]
#   candidate_finder.sh tests/compile/correctness_e2e/

if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory-or-file> [directory-or-file...]" >&2
    exit 1
fi

for arg in "$@"; do
    if [ -f "$arg" ]; then
        case "$arg" in
            */test_*.py) echo "$arg" ;;
        esac
    elif [ -d "$arg" ]; then
        find "$arg" -name 'test_*.py' 2>/dev/null
    fi
done | sort -u
