"""Unit tests for torch-compile orchestration layer.

Tests allowlist building, schema validation, handoff validation,
and permissive mode operation.
"""

import json
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from agent_allowlist import build_allowlist, validate_handoff_allowlist
from orchestrate_schemas import (
    load_schemas,
    validate_handoff_request,
    validate_response,
)
from orchestrate import TorchCompileOrchestrator


# Test fixtures
TEST_ROOT = Path(__file__).parent.parent
MANIFEST_DIR = TEST_ROOT / "managed-agent-cookbooks"
SCHEMA_DIR = TEST_ROOT / "schemas"


class TestAllowlistBuilder:
    """Test agent allowlist building."""

    def test_build_allowlist(self):
        """Test that allowlist loads all agents."""
        allowlist = build_allowlist(MANIFEST_DIR)

        # Should have 5 agents
        assert len(allowlist) >= 5
        assert "coordinator-agent" in allowlist
        assert "dynamo-debugger-agent" in allowlist
        assert "inductor-debugger-agent" in allowlist
        assert "aot-debugger-agent" in allowlist
        assert "bisector-agent" in allowlist

    def test_coordinator_callable_agents(self):
        """Test that coordinator can call all specialists."""
        allowlist = build_allowlist(MANIFEST_DIR)

        coordinator_callables = allowlist["coordinator-agent"]
        assert "dynamo-debugger-agent" in coordinator_callables
        assert "inductor-debugger-agent" in coordinator_callables
        assert "aot-debugger-agent" in coordinator_callables
        assert "bisector-agent" in coordinator_callables

    def test_bidirectional_handoffs(self):
        """Test that specialists can call coordinator."""
        allowlist = build_allowlist(MANIFEST_DIR)

        # Specialists should be able to call coordinator
        assert "coordinator-agent" in allowlist["dynamo-debugger-agent"]
        assert "coordinator-agent" in allowlist["inductor-debugger-agent"]
        assert "coordinator-agent" in allowlist["aot-debugger-agent"]


class TestAllowlistValidation:
    """Test handoff allowlist validation."""

    def test_valid_handoff(self):
        """Test validation of allowed handoff."""
        allowlist = build_allowlist(MANIFEST_DIR)
        handoff = {
            "from_agent": "coordinator-agent",
            "to_agent": "dynamo-debugger-agent",
        }

        valid, error = validate_handoff_allowlist(handoff, allowlist)
        assert valid is True
        assert error == ""

    def test_invalid_handoff_not_in_allowlist(self):
        """Test validation of handoff not in allowlist."""
        allowlist = build_allowlist(MANIFEST_DIR)
        # Dynamo debugger cannot call bisector (not in its callable_agents)
        handoff = {
            "from_agent": "dynamo-debugger-agent",
            "to_agent": "bisector-agent",
        }

        valid, error = validate_handoff_allowlist(handoff, allowlist)
        assert valid is False
        assert "cannot call" in error

    def test_invalid_handoff_unknown_agent(self):
        """Test validation with unknown agent."""
        allowlist = build_allowlist(MANIFEST_DIR)
        handoff = {"from_agent": "unknown-agent", "to_agent": "dynamo-debugger-agent"}

        valid, error = validate_handoff_allowlist(handoff, allowlist)
        assert valid is False
        assert "Unknown agent" in error


class TestSchemaLoading:
    """Test JSON schema loading."""

    def test_load_schemas(self):
        """Test that all schemas load successfully."""
        schemas = load_schemas(SCHEMA_DIR)

        # Should have at least 5 schemas
        assert len(schemas) >= 5
        assert "handoff_request" in schemas
        assert "dynamo_response" in schemas
        assert "inductor_response" in schemas
        assert "aot_response" in schemas
        assert "coordinator_routing" in schemas

    def test_schema_structure(self):
        """Test that schemas have expected structure."""
        schemas = load_schemas(SCHEMA_DIR)

        # handoff_request should have required fields
        handoff_schema = schemas["handoff_request"]
        assert "$schema" in handoff_schema
        assert "type" in handoff_schema
        assert handoff_schema["type"] == "object"


class TestHandoffValidation:
    """Test handoff request schema validation."""

    def test_valid_handoff_request(self):
        """Test validation of valid handoff request."""
        schemas = load_schemas(SCHEMA_DIR)
        handoff = {
            "type": "handoff_request",
            "from_agent": "coordinator-agent",
            "to_agent": "dynamo-debugger-agent",
            "task": {"type": "debug_graph_break", "issue": "test issue"},
            "expected_deliverable": "structured_json",
            "priority": "high",
        }

        valid, errors = validate_handoff_request(handoff, schemas)
        assert valid is True
        assert errors == []

    def test_invalid_handoff_request_missing_field(self):
        """Test validation of handoff request missing required field."""
        schemas = load_schemas(SCHEMA_DIR)
        handoff = {
            "type": "handoff_request",
            "from_agent": "coordinator-agent",
            # Missing to_agent, task, expected_deliverable, priority
        }

        valid, errors = validate_handoff_request(handoff, schemas)
        assert valid is False
        assert len(errors) > 0


class TestResponseValidation:
    """Test agent response schema validation."""

    def test_valid_dynamo_response(self):
        """Test validation of valid dynamo expert response."""
        schemas = load_schemas(SCHEMA_DIR)
        response = {
            "specialist": "dynamo-debugger-agent",
            "version": "1.0.0",
            "task": "test task",
            "confidence": "high",
            "insight": "test insight",
            "files": [],
            "concepts": [],
            "guidance": "test guidance",
            "code": "pass",
            "steps": [],
            "dependencies": [],
            "pitfalls": [],
            "skill_references": [],
        }

        valid, errors = validate_response(response, "dynamo-debugger-agent", schemas)
        assert valid is True
        assert errors == []

    def test_invalid_response_missing_field(self):
        """Test validation of response missing required field."""
        schemas = load_schemas(SCHEMA_DIR)
        response = {
            "specialist": "dynamo-debugger-agent",
            # Missing many required fields
        }

        valid, errors = validate_response(response, "dynamo-debugger-agent", schemas)
        assert valid is False
        assert len(errors) > 0


class TestOrchestrator:
    """Test TorchCompileOrchestrator class."""

    def test_initialization(self, capsys):
        """Test orchestrator initialization."""
        orchestrator = TorchCompileOrchestrator(MANIFEST_DIR, SCHEMA_DIR)

        # Check that allowlist and schemas are loaded
        assert len(orchestrator.allowlist) >= 5
        assert len(orchestrator.schemas) >= 5
        assert orchestrator.validation_mode == "permissive"

        # Check console output
        captured = capsys.readouterr()
        assert "[ORCHESTRATOR] Initialized" in captured.out
        assert "5 agents" in captured.out
        assert "PERMISSIVE" in captured.out

    def test_validate_valid_handoff(self, capsys):
        """Test validation of valid handoff in permissive mode."""
        orchestrator = TorchCompileOrchestrator(MANIFEST_DIR, SCHEMA_DIR)
        capsys.readouterr()  # Clear initialization output

        handoff = {
            "type": "handoff_request",
            "from_agent": "coordinator-agent",
            "to_agent": "dynamo-debugger-agent",
            "task": {"type": "debug_graph_break", "issue": "test"},
            "expected_deliverable": "structured_json",
            "priority": "high",
        }

        result = orchestrator.validate_handoff(handoff)

        assert result["valid"] is True
        assert result["schema_valid"] is True
        assert result["allowlist_valid"] is True
        assert len(result["warnings"]) == 0

        # No warnings should be printed
        captured = capsys.readouterr()
        assert "⚠️" not in captured.out

    def test_validate_invalid_handoff_permissive(self, capsys):
        """Test that invalid handoff is allowed in permissive mode."""
        orchestrator = TorchCompileOrchestrator(MANIFEST_DIR, SCHEMA_DIR)
        capsys.readouterr()  # Clear initialization output

        # Invalid: dynamo expert cannot call bisector
        handoff = {
            "type": "handoff_request",
            "from_agent": "dynamo-debugger-agent",
            "to_agent": "bisector-agent",
            "task": {"type": "bisect_failure", "issue": "test"},
            "expected_deliverable": "bisection_result",
            "priority": "high",
        }

        result = orchestrator.validate_handoff(handoff)

        # In permissive mode, valid=True even with warnings
        assert result["valid"] is True
        assert result["allowlist_valid"] is False
        assert len(result["warnings"]) > 0

        # Warnings should be printed
        captured = capsys.readouterr()
        assert "⚠️" in captured.out
        assert "Warnings (allowing anyway)" in captured.out

    def test_log_handoff(self, capsys):
        """Test handoff logging to console."""
        orchestrator = TorchCompileOrchestrator(MANIFEST_DIR, SCHEMA_DIR)
        capsys.readouterr()  # Clear initialization output

        orchestrator.log_handoff(
            from_agent="coordinator-agent",
            to_agent="dynamo-debugger-agent",
            latency_ms=125.0,
            success=True,
            warnings=[],
        )

        captured = capsys.readouterr()
        assert "coordinator → dynamo-debugger (125ms) ✓" in captured.out


class TestPermissiveMode:
    """Test permissive validation mode."""

    def test_permissive_allows_schema_violations(self, capsys):
        """Test that schema violations are allowed in permissive mode."""
        orchestrator = TorchCompileOrchestrator(
            MANIFEST_DIR, SCHEMA_DIR, validation_mode="permissive"
        )
        capsys.readouterr()

        # Missing required fields
        handoff = {"type": "handoff_request", "from_agent": "coordinator-agent"}

        result = orchestrator.validate_handoff(handoff)

        # Should be valid=True with warnings
        assert result["valid"] is True
        assert result["schema_valid"] is False
        assert len(result["warnings"]) > 0

        captured = capsys.readouterr()
        assert "⚠️" in captured.out

    def test_permissive_allows_allowlist_violations(self, capsys):
        """Test that allowlist violations are allowed in permissive mode."""
        orchestrator = TorchCompileOrchestrator(
            MANIFEST_DIR, SCHEMA_DIR, validation_mode="permissive"
        )
        capsys.readouterr()

        # Not in allowlist
        handoff = {
            "type": "handoff_request",
            "from_agent": "dynamo-debugger-agent",
            "to_agent": "bisector-agent",
            "task": {"type": "test", "issue": "test"},
            "expected_deliverable": "structured_json",
            "priority": "high",
        }

        result = orchestrator.validate_handoff(handoff)

        # Should be valid=True with warnings
        assert result["valid"] is True
        assert result["allowlist_valid"] is False
        assert len(result["warnings"]) > 0

        captured = capsys.readouterr()
        assert "⚠️" in captured.out
