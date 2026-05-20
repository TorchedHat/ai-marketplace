"""Schema loader and validator for torch-compile multi-agent system.

Loads JSON schemas and validates handoff requests and agent responses against them.
"""

import json
from pathlib import Path

import jsonschema

# Mapping from agent name to response schema filename
AGENT_TO_SCHEMA = {
    "dynamo-debugger-agent": "dynamo_response.json",
    "inductor-debugger-agent": "inductor_response.json",
    "aot-debugger-agent": "aot_response.json",
    "coordinator-agent": "coordinator_routing.json",
    "bisector-agent": "coordinator_routing.json",  # Uses same schema as coordinator
}


def load_schemas(schema_dir: Path) -> dict[str, dict]:
    """Load all JSON schemas from schema directory.

    Args:
        schema_dir: Path to directory containing JSON schema files

    Returns:
        Dictionary mapping schema name (without .json) to parsed schema:
        {
            "handoff_request": {...},
            "dynamo_response": {...},
            "inductor_response": {...},
            ...
        }

    Example:
        >>> schemas = load_schemas(Path("schemas"))
        >>> print(list(schemas.keys()))
        ["handoff_request", "dynamo_response", "inductor_response", ...]
    """
    schemas = {}

    for schema_file in schema_dir.glob("*.json"):
        try:
            with open(schema_file) as f:
                schema = json.load(f)
            schemas[schema_file.stem] = schema
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to load {schema_file}: {e}")
            continue

    return schemas


def validate_handoff_request(handoff: dict, schemas: dict[str, dict]) -> tuple[bool, list[str]]:
    """Validate handoff request against handoff_request.json schema.

    Args:
        handoff: Handoff request dictionary to validate
        schemas: Schema dictionary from load_schemas()

    Returns:
        Tuple of (is_valid, validation_errors):
        - (True, []) if valid
        - (False, [error_messages]) if invalid

    Example:
        >>> handoff = {"type": "handoff_request", "from_agent": "coordinator-agent", ...}
        >>> valid, errors = validate_handoff_request(handoff, schemas)
        >>> print(valid, errors)
        True []
    """
    schema = schemas.get("handoff_request")
    if not schema:
        return False, ["handoff_request.json schema not found"]

    try:
        jsonschema.validate(instance=handoff, schema=schema)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e)]


def validate_response(
    response: dict, agent_name: str, schemas: dict[str, dict]
) -> tuple[bool, list[str]]:
    """Validate agent response against its stage-specific schema.

    Args:
        response: Agent response dictionary to validate
        agent_name: Name of the agent that produced the response
        schemas: Schema dictionary from load_schemas()

    Returns:
        Tuple of (is_valid, validation_errors):
        - (True, []) if valid
        - (False, [error_messages]) if invalid

    Example:
        >>> response = {"specialist": "dynamo-expert-agent", "version": "1.0.0", ...}
        >>> valid, errors = validate_response(response, "dynamo-expert-agent", schemas)
        >>> print(valid, errors)
        True []
    """
    # Get schema filename for this agent
    schema_filename = AGENT_TO_SCHEMA.get(agent_name)
    if not schema_filename:
        return False, [f"No schema defined for agent: {agent_name}"]

    # Get schema from loaded schemas
    schema_key = schema_filename.replace(".json", "")
    schema = schemas.get(schema_key)
    if not schema:
        return False, [f"Schema not found: {schema_filename}"]

    try:
        jsonschema.validate(instance=response, schema=schema)
        return True, []
    except jsonschema.ValidationError as e:
        # Extract the most relevant error message
        error_msg = str(e).split("\n")[0]  # First line is most relevant
        return False, [error_msg]
