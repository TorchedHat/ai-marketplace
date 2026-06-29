#!/usr/bin/env bash
set -euo pipefail

# Candidate Finder for vLLM Test Oracle Auditor
#
# Greps test files for assertion patterns that indicate potentially
# coincidentally-correct tests. Outputs FILE:LINE:PATTERN_NAME triples.
#
# Usage:
#   candidate_finder.sh <directory-or-file> [directory-or-file...]
#   candidate_finder.sh tests/compile/correctness_e2e/
#   candidate_finder.sh tests/v1/e2e/general/test_cascade_attention.py

if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory-or-file> [directory-or-file...]" >&2
    exit 1
fi

# Collect all test_*.py files from arguments, excluding tests/v1/determinism/
# (has VLLM_BATCH_INVARIANT=1 via autouse conftest)
files=()
for arg in "$@"; do
    if [ -f "$arg" ]; then
        case "$arg" in
            */v1/determinism/*) ;;
            */test_*.py) files+=("$arg") ;;
        esac
    elif [ -d "$arg" ]; then
        while IFS= read -r f; do
            files+=("$f")
        done < <(find "$arg" -name 'test_*.py' -not -path '*/v1/determinism/*' 2>/dev/null)
    fi
done

if [ ${#files[@]} -eq 0 ]; then
    echo "# No test files found" >&2
    exit 0
fi

# Pattern definitions: GREP_PATTERN|PATTERN_NAME
patterns=(
    'compare_two_settings\(|COMPARE_TWO_SETTINGS'
    'compare_all_settings\(|COMPARE_ALL_SETTINGS'
    'check_outputs_equal\(|CHECK_OUTPUTS_EQUAL'
    'validate_generated_texts\(|VALIDATE_GENERATED_TEXTS'
    '\.outputs\[0\]\.text ==|TEXT_EQUALITY'
    '\.text == ref|TEXT_EQUALITY'
    '\] \* [0-9]|BATCH_MULTIPLY'
)

# Run grep for each pattern and emit structured output
for entry in "${patterns[@]}"; do
    pattern="${entry%%|*}"
    name="${entry##*|}"
    { grep -rHn -E "$pattern" "${files[@]}" 2>/dev/null || true; } | while IFS=: read -r file line rest; do
        echo "${file}:${line}:${name}"
    done
done | sort -t: -k1,1 -k2,2n | uniq
