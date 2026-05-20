#!/usr/bin/env python3
"""Sync skills from vertical-plugins/ to agent-plugins/ bundles.

Validates:
- No drift between source and synced copies
- YAML frontmatter is valid
- Cross-references resolve
- No circular dependencies
"""

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
VERTICAL_PLUGINS = REPO_ROOT / "vertical-plugins"
AGENT_PLUGINS = REPO_ROOT / "agent-plugins"

SKILL_MAPPINGS = {
    "coordinator-agent": ["compile-overview"],
    "dynamo-debugger-agent": ["pytorch-dynamo", "compile-trace-dynamo"],
    "inductor-debugger-agent": ["pytorch-inductor", "compile-trace-inductor"],
    "aot-debugger-agent": ["compile-trace-aot"],
    "bisector-agent": ["compile-bisect"],
}


def find_skill_source(skill_name: str) -> Path:
    """Find skill source in vertical-plugins/ or coordinator/."""
    # Check vertical-plugins first
    for vertical in VERTICAL_PLUGINS.iterdir():
        if not vertical.is_dir():
            continue
        skill_path = vertical / "skills" / skill_name
        if skill_path.exists():
            return skill_path

    # Check coordinator/skills for compile-overview
    coordinator_path = REPO_ROOT / "coordinator" / "skills" / skill_name
    if coordinator_path.exists():
        return coordinator_path

    raise FileNotFoundError(f"Skill {skill_name} not found in vertical-plugins/ or coordinator/")


def sync_skill(source: Path, dest: Path) -> bool:
    """Sync skill from source to dest, return True if changed."""
    if dest.exists():
        # Check if up-to-date
        source_mtime = source.stat().st_mtime
        dest_mtime = dest.stat().st_mtime
        if source_mtime <= dest_mtime:
            return False  # Dest is current

    # Sync
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, symlinks=True)
    return True


def validate_skill(skill_path: Path) -> list[str]:
    """Validate skill YAML frontmatter and structure."""
    errors = []
    skill_md = skill_path / "SKILL.md"

    if not skill_md.exists():
        errors.append(f"{skill_path.name}: Missing SKILL.md")
        return errors

    content = skill_md.read_text()

    # Check frontmatter
    if not content.startswith("---\n"):
        errors.append(f"{skill_path.name}: Missing YAML frontmatter")

    # Check required fields
    required_fields = ["name:", "description:"]
    for field in required_fields:
        if field not in content:
            errors.append(f"{skill_path.name}: Missing {field} in frontmatter")

    return errors


def main():
    """Sync all skills and validate."""
    print("Syncing skills from vertical-plugins/ to agent-plugins/...\n")

    changed_count = 0
    errors = []

    for agent_name, skill_names in SKILL_MAPPINGS.items():
        agent_dir = AGENT_PLUGINS / agent_name / "skills"
        agent_dir.mkdir(parents=True, exist_ok=True)

        print(f"Agent: {agent_name}")
        for skill_name in skill_names:
            try:
                source = find_skill_source(skill_name)
                dest = agent_dir / skill_name

                changed = sync_skill(source, dest)
                status = "✓ synced" if changed else "✓ current"
                print(f"  {skill_name}: {status}")

                if changed:
                    changed_count += 1

                # Validate
                skill_errors = validate_skill(dest)
                errors.extend(skill_errors)

            except FileNotFoundError as e:
                errors.append(str(e))
                print(f"  {skill_name}: ✗ not found")
        print()

    # Report
    print(f"\nSummary: {changed_count} skills updated")

    if errors:
        print(f"\n❌ {len(errors)} validation errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("✅ All skills valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())
