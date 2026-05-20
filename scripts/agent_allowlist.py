"""Agent allowlist builder and validator for torch-compile multi-agent system.

Parses agent.yaml manifests to build callable_agents graph and validates
handoff requests against the allowlist.
"""

from pathlib import Path

import yaml


def build_allowlist(agent_manifest_dir: Path) -> dict[str, list[str]]:
    """Build callable_agents graph from agent.yaml manifests.

    Parses all agent.yaml files in the manifest directory and extracts
    the callable_agents list for each agent.

    Args:
        agent_manifest_dir: Path to directory containing agent subdirectories,
                          each with an agent.yaml file

    Returns:
        Dictionary mapping agent name to list of agents it can call:
        {
            "coordinator-agent": ["dynamo-expert-agent", "inductor-expert-agent", ...],
            "dynamo-expert-agent": ["coordinator-agent", "inductor-expert-agent"],
            ...
        }

    Example:
        >>> allowlist = build_allowlist(Path("managed-agent-cookbooks"))
        >>> print(allowlist["coordinator-agent"])
        ["dynamo-expert-agent", "inductor-expert-agent", "aot-debugger-agent", "bisector-agent"]
    """
    allowlist = {}

    for agent_yaml_path in agent_manifest_dir.glob("*/agent.yaml"):
        try:
            with open(agent_yaml_path) as f:
                manifest = yaml.safe_load(f)

            agent_name = manifest.get("name")
            if not agent_name:
                continue

            callable = manifest.get("callable_agents", [])
            allowlist[agent_name] = callable

        except (OSError, yaml.YAMLError) as e:
            print(f"Warning: Failed to load {agent_yaml_path}: {e}")
            continue

    return allowlist


def validate_handoff_allowlist(handoff: dict, allowlist: dict[str, list[str]]) -> tuple[bool, str]:
    """Validate that handoff is allowed according to callable_agents.

    Checks if the from_agent is permitted to call the to_agent based on
    the allowlist graph.

    Args:
        handoff: Handoff request dictionary with from_agent and to_agent fields
        allowlist: Allowlist graph from build_allowlist()

    Returns:
        Tuple of (is_valid, error_message):
        - (True, "") if handoff is allowed
        - (False, error_message) if handoff is not allowed

    Example:
        >>> handoff = {"from_agent": "coordinator-agent", "to_agent": "dynamo-expert-agent"}
        >>> valid, msg = validate_handoff_allowlist(handoff, allowlist)
        >>> print(valid, msg)
        True ""
    """
    from_agent = handoff.get("from_agent")
    to_agent = handoff.get("to_agent")

    # Check if from_agent exists in allowlist
    if from_agent not in allowlist:
        return False, f"Unknown agent: {from_agent}"

    # Check if to_agent is in from_agent's callable_agents list
    if to_agent not in allowlist[from_agent]:
        return False, f"{from_agent} cannot call {to_agent} (not in callable_agents)"

    return True, ""
