#!/usr/bin/env python3
"""Validate skills for YAML frontmatter, cross-references, and circular dependencies."""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).parent.parent
VERTICAL_PLUGINS = REPO_ROOT / "vertical-plugins"

def extract_skill_references(content: str) -> List[str]:
    """Extract [[skill-name]] references from content."""
    pattern = r'\[\[([^\]]+)\]\]'
    return re.findall(pattern, content)

def validate_frontmatter(skill_path: Path) -> List[str]:
    """Validate YAML frontmatter."""
    errors = []
    skill_md = skill_path / "SKILL.md"

    if not skill_md.exists():
        return [f"{skill_path.name}: Missing SKILL.md"]

    content = skill_md.read_text()

    if not content.startswith("---\n"):
        errors.append(f"{skill_path.name}: Missing YAML frontmatter")
        return errors

    # Extract frontmatter
    parts = content.split("---\n")
    if len(parts) < 3:
        errors.append(f"{skill_path.name}: Incomplete YAML frontmatter")
        return errors

    frontmatter = parts[1]

    # Check required fields
    required_fields = ["name:", "description:"]
    for field in required_fields:
        if field not in frontmatter:
            errors.append(f"{skill_path.name}: Missing {field} in frontmatter")

    return errors

def find_all_skills() -> Dict[str, Path]:
    """Find all skills in vertical-plugins/."""
    skills = {}
    for vertical in VERTICAL_PLUGINS.iterdir():
        if not vertical.is_dir():
            continue
        skills_dir = vertical / "skills"
        if not skills_dir.exists():
            continue
        for skill in skills_dir.iterdir():
            if skill.is_dir() and (skill / "SKILL.md").exists():
                skills[skill.name] = skill
    return skills

def validate_cross_references(skills: Dict[str, Path]) -> List[str]:
    """Validate cross-references between skills."""
    errors = []

    for skill_name, skill_path in skills.items():
        skill_md = skill_path / "SKILL.md"
        content = skill_md.read_text()

        references = extract_skill_references(content)
        for ref in references:
            if ref not in skills:
                errors.append(f"{skill_name}: References non-existent skill [[{ref}]]")

    return errors

def detect_circular_dependencies(skills: Dict[str, Path]) -> List[str]:
    """Detect circular dependencies in skill references."""
    errors = []

    # Build dependency graph
    graph: Dict[str, Set[str]] = {name: set() for name in skills}

    for skill_name, skill_path in skills.items():
        skill_md = skill_path / "SKILL.md"
        content = skill_md.read_text()
        references = extract_skill_references(content)
        graph[skill_name] = set(ref for ref in references if ref in skills)

    # DFS to detect cycles
    def has_cycle(node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if has_cycle(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    visited: Set[str] = set()
    for skill_name in skills:
        if skill_name not in visited:
            if has_cycle(skill_name, visited, set()):
                errors.append(f"Circular dependency detected involving: {skill_name}")

    return errors

def main():
    """Run all validations."""
    print("Validating skills in vertical-plugins/...\n")

    all_errors = []

    # Find all skills
    skills = find_all_skills()
    print(f"Found {len(skills)} skills\n")

    # Validate frontmatter
    print("Validating YAML frontmatter...")
    for skill_name, skill_path in skills.items():
        errors = validate_frontmatter(skill_path)
        all_errors.extend(errors)
    if not all_errors:
        print("  ✓ All frontmatter valid\n")
    else:
        print(f"  ✗ {len(all_errors)} errors\n")

    # Validate cross-references
    print("Validating cross-references...")
    ref_errors = validate_cross_references(skills)
    all_errors.extend(ref_errors)
    if not ref_errors:
        print("  ✓ All references valid\n")
    else:
        print(f"  ✗ {len(ref_errors)} errors\n")

    # Detect circular dependencies
    print("Checking for circular dependencies...")
    circ_errors = detect_circular_dependencies(skills)
    all_errors.extend(circ_errors)
    if not circ_errors:
        print("  ✓ No circular dependencies\n")
    else:
        print(f"  ✗ {len(circ_errors)} errors\n")

    # Report
    if all_errors:
        print(f"\n❌ {len(all_errors)} validation errors:")
        for error in all_errors:
            print(f"  - {error}")
        return 1
    else:
        print("✅ All validations passed")
        return 0

if __name__ == "__main__":
    sys.exit(main())
