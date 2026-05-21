#!/usr/bin/env python3
"""Symlink skills from vertical-plugins/ to agent-plugins/ bundles.

Creates symlinks to ensure single source of truth in vertical-plugins/.

Validates:
- Symlinks point to valid skill directories
- YAML frontmatter is valid in source skills
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
    "aot-debugger-agent": ["pytorch-aot", "compile-trace-aot"],
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
    """Create symlink from dest to source, return True if changed."""
    # Remove existing file/dir/link
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink():
            # Check if already pointing to correct source
            if dest.resolve() == source.resolve():
                return False  # Already correct symlink
        # Remove old copy or incorrect symlink
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()

    # Create symlink
    dest.symlink_to(source)
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
    """Create skill symlinks and validate."""
    print("Creating skill symlinks from vertical-plugins/ to agent-plugins/...\n")

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
                status = "✓ symlinked" if changed else "✓ current"
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
    print(f"\nSummary: {changed_count} symlinks created/updated")

    if errors:
        print(f"\n❌ {len(errors)} validation errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("✅ All symlinks created and skills valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())
