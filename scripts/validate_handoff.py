#!/usr/bin/env python3
"""Simple helper to validate handoff requests from coordinator.

Usage from coordinator agent or interactive session:
    python scripts/validate_handoff.py '{"type": "handoff_request", ...}'
"""

import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from orchestrate import TorchCompileOrchestrator


def main():
    """Validate handoff request passed as JSON string argument."""
    if len(sys.argv) < 2:
        print("Usage: python validate_handoff.py '<handoff_json>'")
        print("   or: echo '<handoff_json>' | python validate_handoff.py")
        sys.exit(1)

    # Read from stdin or argument
    if sys.argv[1] == "-":
        handoff_json = sys.stdin.read()
    else:
        handoff_json = sys.argv[1]

    try:
        handoff = json.loads(handoff_json)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        sys.exit(1)

    # Initialize orchestrator
    repo_root = Path(__file__).parent.parent
    orchestrator = TorchCompileOrchestrator(
        agent_manifest_dir=repo_root / "managed-agent-cookbooks",
        schema_dir=repo_root / "schemas",
        validation_mode="permissive",
    )

    # Validate handoff
    result = orchestrator.validate_handoff(handoff)

    # Print result
    print()
    if result["valid"]:
        print("✅ Handoff valid")
        if result["warnings"]:
            print(f"   Warnings: {len(result['warnings'])}")
    else:
        print("❌ Handoff invalid")

    print(f"   Schema: {'✓' if result['schema_valid'] else '✗'}")
    print(f"   Allowlist: {'✓' if result['allowlist_valid'] else '✗'}")

    if result["warnings"]:
        print()
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")

    # Exit with appropriate code
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
