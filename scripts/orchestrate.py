#!/usr/bin/env python3
"""Orchestration layer for torch-compile multi-agent system.

Validates handoff requests and agent responses against schemas and allowlists.
Operates in permissive mode: logs warnings but doesn't block invalid handoffs.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_allowlist import build_allowlist, validate_handoff_allowlist
from orchestrate_schemas import (
    load_schemas,
    validate_handoff_request,
    validate_response,
)


@dataclass
class HandoffMetrics:
    """Telemetry for a single agent handoff.

    Attributes:
        agent_from: Name of agent initiating handoff
        agent_to: Name of target agent
        latency_ms: Time taken for handoff in milliseconds
        success: Whether handoff completed successfully
        validation_warnings: List of validation warning messages
        timestamp: When handoff occurred
    """

    agent_from: str
    agent_to: str
    latency_ms: float
    success: bool
    validation_warnings: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HandoffState:
    """Tracks state for a multi-agent conversation.

    Attributes:
        session_id: Unique identifier for this conversation
        request_path: List of agents in handoff chain (e.g., ["coordinator", "bisector", "inductor"])
        context: Shared state dictionary passed between agents
        metrics: List of HandoffMetrics for all handoffs in this session
    """

    session_id: str
    request_path: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    metrics: list[HandoffMetrics] = field(default_factory=list)

    def add_handoff(self, from_agent: str, to_agent: str) -> None:
        """Record handoff in request path.

        Args:
            from_agent: Name of agent initiating handoff
            to_agent: Name of target agent
        """
        if not self.request_path:
            self.request_path.append(from_agent)
        self.request_path.append(to_agent)


class TorchCompileOrchestrator:
    """Orchestrates handoffs between torch-compile debugging agents.

    Validates handoff requests and responses against JSON schemas and
    callable_agents allowlists. Operates in permissive mode: logs warnings
    but allows all handoffs to proceed.

    Attributes:
        allowlist: Dictionary mapping agent names to callable agents
        schemas: Dictionary of loaded JSON schemas
        validation_mode: "permissive" (warn only) or "strict" (block invalid)
    """

    def __init__(
        self,
        agent_manifest_dir: Path,
        schema_dir: Path,
        validation_mode: str = "permissive",
    ):
        """Initialize orchestrator.

        Args:
            agent_manifest_dir: Path to managed-agent-cookbooks directory
            schema_dir: Path to schemas directory
            validation_mode: "permissive" (default) or "strict"
        """
        self.allowlist = build_allowlist(agent_manifest_dir)
        self.schemas = load_schemas(schema_dir)
        self.validation_mode = validation_mode

        # Log initialization
        print(
            f"[ORCHESTRATOR] Initialized ({len(self.allowlist)} agents, {len(self.schemas)} schemas)"
        )
        for agent, callables in sorted(self.allowlist.items()):
            short_name = agent.replace("-agent", "")
            print(f"[ORCHESTRATOR] Allowlist: {short_name} → {len(callables)} agents")
        print(
            f"[ORCHESTRATOR] Validation mode: {validation_mode.upper()} ({'warnings only' if validation_mode == 'permissive' else 'blocks invalid'})"
        )
        print("[ORCHESTRATOR] Ready for handoff validation")

    def validate_handoff(self, handoff: dict) -> dict[str, Any]:
        """Validate handoff request against schema and allowlist.

        Args:
            handoff: Handoff request dictionary

        Returns:
            Dictionary with validation results:
            {
                "valid": bool,
                "warnings": [str],
                "schema_valid": bool,
                "allowlist_valid": bool
            }
        """
        warnings = []

        # Validate against schema
        schema_valid, schema_errors = validate_handoff_request(handoff, self.schemas)
        if not schema_valid:
            warnings.extend([f"Schema: {e}" for e in schema_errors])

        # Validate against allowlist
        allowlist_valid, allowlist_error = validate_handoff_allowlist(
            handoff, self.allowlist
        )
        if not allowlist_valid:
            warnings.append(f"Allowlist: {allowlist_error}")

        # In permissive mode, always return valid=True but log warnings
        if self.validation_mode == "permissive":
            valid = True
            if warnings:
                print(f"[ORCHESTRATOR] ⚠️  Warnings (allowing anyway):")
                for warning in warnings:
                    print(f"[ORCHESTRATOR]     {warning}")
        else:
            valid = schema_valid and allowlist_valid
            if not valid:
                print(f"[ORCHESTRATOR] ❌ Validation failed:")
                for warning in warnings:
                    print(f"[ORCHESTRATOR]     {warning}")

        return {
            "valid": valid,
            "warnings": warnings,
            "schema_valid": schema_valid,
            "allowlist_valid": allowlist_valid,
        }

    def validate_agent_response(
        self, response: dict, agent_name: str
    ) -> dict[str, Any]:
        """Validate agent response against stage-specific schema.

        Args:
            response: Agent response dictionary
            agent_name: Name of agent that produced response

        Returns:
            Dictionary with validation results:
            {
                "valid": bool,
                "warnings": [str]
            }
        """
        valid, errors = validate_response(response, agent_name, self.schemas)

        warnings = []
        if not valid:
            warnings = errors

        # In permissive mode, always return valid=True but log warnings
        if self.validation_mode == "permissive":
            if warnings:
                print(f"[ORCHESTRATOR] ⚠️  Response warnings from {agent_name}:")
                for warning in warnings:
                    print(f"[ORCHESTRATOR]     {warning}")
            return {"valid": True, "warnings": warnings}
        else:
            if not valid:
                print(f"[ORCHESTRATOR] ❌ Response validation failed for {agent_name}:")
                for warning in warnings:
                    print(f"[ORCHESTRATOR]     {warning}")
            return {"valid": valid, "warnings": warnings}

    def log_handoff(
        self,
        from_agent: str,
        to_agent: str,
        latency_ms: float,
        success: bool,
        warnings: list[str],
    ) -> None:
        """Log handoff metrics to console.

        Args:
            from_agent: Name of agent initiating handoff
            to_agent: Name of target agent
            latency_ms: Time taken in milliseconds
            success: Whether handoff succeeded
            warnings: List of validation warnings
        """
        status = "✓" if success else "✗"
        from_short = from_agent.replace("-agent", "")
        to_short = to_agent.replace("-agent", "")

        log_msg = f"[ORCHESTRATOR] {from_short} → {to_short} ({latency_ms:.0f}ms) {status}"
        print(log_msg)


def main():
    """Run orchestrator in console mode for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Torch-compile multi-agent orchestrator"
    )
    parser.add_argument(
        "--mode",
        choices=["console", "validate"],
        default="console",
        help="Run mode: console (show status) or validate (validate handoff from stdin)",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(__file__).parent.parent / "managed-agent-cookbooks",
        help="Path to agent manifest directory",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path(__file__).parent.parent / "schemas",
        help="Path to schemas directory",
    )
    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = TorchCompileOrchestrator(args.manifest_dir, args.schema_dir)

    if args.mode == "console":
        print()
        print("[ORCHESTRATOR] Console mode - orchestrator initialized and ready")
        print(
            "[ORCHESTRATOR] Waiting for handoff requests (pass JSON via stdin in 'validate' mode)"
        )

    elif args.mode == "validate":
        import json

        print()
        print("[ORCHESTRATOR] Reading handoff request from stdin...")
        try:
            handoff = json.load(sys.stdin)
            result = orchestrator.validate_handoff(handoff)
            print()
            print(f"[ORCHESTRATOR] Validation result: {result}")
        except json.JSONDecodeError as e:
            print(f"[ORCHESTRATOR] ❌ Invalid JSON: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
