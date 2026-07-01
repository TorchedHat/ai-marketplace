#!/usr/bin/env bash
set -euo pipefail

# List all test functions from directories, files, or file::function targets.
# Output is sorted by directory, then file, then function order within file.
# Outputs DIR,FILE,FUNCTION as CSV, one per line.
#
# Usage:
#   list_tests.sh <directory-or-file-or-target> [...]
#   list_tests.sh tests/compile/correctness_e2e/
#   list_tests.sh tests/v1/e2e/spec_decode/test_spec_decode.py
#   list_tests.sh tests/v1/e2e/spec_decode/test_spec_decode.py::test_mtp_correctness

if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory|file|file::function> [...]" >&2
    exit 1
fi

list_functions() {
    local file="$1"
    grep -oP '(?<=^def |^    def |^async def )test_\w+' "$file" 2>/dev/null | \
        while IFS= read -r func; do
            echo "$(dirname "$file"),$(basename "$file"),${func}"
        done
}

for arg in "$@"; do
    if [[ "$arg" == *"::"* ]]; then
        # file::function — convert to csv
        f="${arg%%::*}"
        echo "$(dirname "$f"),$(basename "$f"),${arg##*::}"
    elif [ -f "$arg" ]; then
        case "$arg" in
            */test_*.py) list_functions "$arg" ;;
        esac
    elif [ -d "$arg" ]; then
        find "$arg" -name 'test_*.py' 2>/dev/null | sort | while IFS= read -r file; do
            list_functions "$file"
        done
    fi
done
