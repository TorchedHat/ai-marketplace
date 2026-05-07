"""
Post-Grad Pass Analyzer

Analyzes post-grad optimization passes applied to backward graphs:
- CSE (Common Subexpression Elimination)
- DCE (Dead Code Elimination)
- Pattern matching
- No-op removal
"""

import re
from pathlib import Path


async def analyze_post_grad_passes(
    backward_graph_before: str,
    backward_graph_after: str | None = None
) -> str:
    """
    Analyze post-grad pass effects on backward graphs.

    Parses backward graph before and after post-grad passes to identify:
    - CSE (Common Subexpression Elimination) - duplicate operations merged
    - DCE (Dead Code Elimination) - unused operations removed
    - Pattern matching - optimization patterns applied
    - No-op removal - identity operations eliminated

    Args:
        backward_graph_before: Path to backward graph before post-grad passes
        backward_graph_after: Optional path to backward graph after post-grad passes

    Returns:
        Formatted analysis of post-grad optimizations applied
    """
    # Check if files exist
    before_path = Path(backward_graph_before)
    if not before_path.exists():
        return f"Error: Before file not found: {backward_graph_before}"

    # Read before graph
    before_content = before_path.read_text()

    # Single file mode: analyze potential optimizations
    if backward_graph_after is None:
        return _analyze_single_graph(before_content, str(before_path))

    # Two file mode: compare before and after
    after_path = Path(backward_graph_after)
    if not after_path.exists():
        return f"Error: After file not found: {backward_graph_after}"

    after_content = after_path.read_text()

    return _compare_graphs(before_content, after_content, str(before_path), str(after_path))


def _extract_operations(graph_content: str) -> list[dict[str, str]]:
    """
    Extract operations from FX graph.

    Returns list of dicts with:
    - name: Variable name
    - module: Module (e.g., 'aten')
    - op: Operation type
    - args: Arguments
    - full_line: Complete line
    """
    ops = []
    # Match pattern: var: "type" = torch.ops.aten.op(...) or var = torch.ops...
    pattern = r'^\s*(\w+):\s*"[^"]*"\s*=\s*torch\.ops\.(\w+)\.(\w+)\.(.*)|^\s*(\w+)\s*=\s*torch\.ops\.(\w+)\.(\w+)\.(.*)'

    for line in graph_content.split('\n'):
        match = re.match(pattern, line)
        if match:
            if match.group(1):  # Typed version
                ops.append({
                    'name': match.group(1),
                    'module': match.group(2),
                    'op': match.group(3),
                    'args': match.group(4),
                    'full_line': line.strip()
                })
            elif match.group(5):  # Untyped version
                ops.append({
                    'name': match.group(5),
                    'module': match.group(6),
                    'op': match.group(7),
                    'args': match.group(8),
                    'full_line': line.strip()
                })

    return ops


def _find_duplicates(ops: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    """
    Find duplicate operations (CSE candidates).

    Returns list of (name1, name2, operation) tuples
    """
    duplicates = []
    seen = {}

    for op in ops:
        # Create signature from module, op, and args
        signature = f"{op['module']}.{op['op']}{op['args']}"

        if signature in seen:
            duplicates.append((seen[signature], op['name'], f"{op['module']}.{op['op']}"))
        else:
            seen[signature] = op['name']

    return duplicates


def _find_unused(ops: list[dict[str, str]], graph_content: str) -> list[str]:
    """
    Find unused variables (DCE candidates).

    Returns list of variable names that are defined but never used
    """
    unused = []

    for op in ops:
        name = op['name']
        # Count how many times this variable appears
        # Should appear at least twice: once in definition, once in use
        count = len(re.findall(rf'\b{re.escape(name)}\b', graph_content))

        # If only appears once (in its definition), it's unused
        if count == 1:
            unused.append(name)

    return unused


def _find_noops(ops: list[dict[str, str]]) -> list[str]:
    """
    Find no-op operations (identity operations).

    Returns list of operation names that are no-ops
    """
    noops = []

    for op in ops:
        # Check for mul/div by 1.0
        if op['op'] == 'mul' and ('1.0)' in op['args'] or '1)' in op['args']):
            noops.append(f"{op['name']}: {op['op']} by 1.0")
        elif op['op'] == 'div' and ('1.0)' in op['args'] or '1)' in op['args']):
            noops.append(f"{op['name']}: {op['op']} by 1.0")
        # Check for add/sub 0
        elif op['op'] == 'add' and ('0.0)' in op['args'] or ', 0)' in op['args']):
            noops.append(f"{op['name']}: {op['op']} 0")
        elif op['op'] == 'sub' and ('0.0)' in op['args'] or ', 0)' in op['args']):
            noops.append(f"{op['name']}: {op['op']} 0")

    return noops


def _analyze_single_graph(content: str, path: str) -> str:
    """
    Analyze a single graph for optimization opportunities.
    """
    ops = _extract_operations(content)
    duplicates = _find_duplicates(ops)
    unused = _find_unused(ops, content)
    noops = _find_noops(ops)

    result = [
        "Post-Grad Pass Analysis (Single Graph)",
        "=" * 60,
        f"File: {path}",
        "",
        f"Total Operations: {len(ops)}",
        ""
    ]

    # CSE opportunities
    result.append("CSE Opportunities (Duplicate Operations):")
    if duplicates:
        for name1, name2, op in duplicates:
            result.append(f"  - {name1} and {name2}: duplicate {op}")
    else:
        result.append("  None detected")
    result.append("")

    # DCE opportunities
    result.append("DCE Opportunities (Unused Operations):")
    if unused:
        for name in unused:
            result.append(f"  - {name}: never used")
    else:
        result.append("  None detected")
    result.append("")

    # No-ops
    result.append("No-op Operations:")
    if noops:
        for noop in noops:
            result.append(f"  - {noop}")
    else:
        result.append("  None detected")
    result.append("")

    # Performance impact
    total_removable = len(duplicates) + len(unused) + len(noops)
    if total_removable > 0:
        reduction_pct = (total_removable / len(ops)) * 100 if ops else 0
        result.extend([
            "Performance Impact Estimate:",
            f"  - Removable operations: {total_removable}",
            f"  - Potential reduction: {reduction_pct:.1f}%",
            f"  - Estimated speedup: {min(reduction_pct * 0.5, 20):.1f}%"
        ])

    return '\n'.join(result)


def _compare_graphs(before: str, after: str, before_path: str, after_path: str) -> str:
    """
    Compare before and after graphs to identify applied optimizations.
    """
    before_ops = _extract_operations(before)
    after_ops = _extract_operations(after)

    before_duplicates = _find_duplicates(before_ops)
    before_unused = _find_unused(before_ops, before)
    before_noops = _find_noops(before_ops)

    after_duplicates = _find_duplicates(after_ops)
    after_unused = _find_unused(after_ops, after)
    after_noops = _find_noops(after_ops)

    result = [
        "Post-Grad Pass Analysis",
        "=" * 60,
        f"Before: {before_path}",
        f"After:  {after_path}",
        ""
    ]

    # Node count changes
    nodes_before = len(before_ops)
    nodes_after = len(after_ops)
    nodes_removed = nodes_before - nodes_after
    reduction_pct = (nodes_removed / nodes_before * 100) if nodes_before > 0 else 0

    result.extend([
        "Graph Changes:",
        f"  - Nodes before: {nodes_before}",
        f"  - Nodes after:  {nodes_after}",
        f"  - Removed:      {nodes_removed}",
        f"  - Reduction:    {reduction_pct:.1f}%",
        ""
    ])

    # CSE applied
    cse_applied = len(before_duplicates) - len(after_duplicates)
    result.append("CSE (Common Subexpression Elimination):")
    if cse_applied > 0:
        result.append(f"  ✓ Eliminated {cse_applied} duplicate operation(s)")
        for name1, name2, op in before_duplicates[:3]:  # Show first 3
            result.append(f"    - Merged {name1}/{name2}: {op}")
    else:
        result.append("  - No duplicates eliminated")
    result.append("")

    # DCE applied
    dce_applied = len(before_unused) - len(after_unused)
    result.append("DCE (Dead Code Elimination):")
    if dce_applied > 0:
        result.append(f"  ✓ Eliminated {dce_applied} unused operation(s)")
        for name in before_unused[:3]:  # Show first 3
            result.append(f"    - Removed {name}")
    else:
        result.append("  - No dead code eliminated")
    result.append("")

    # No-op removal
    noops_removed = len(before_noops) - len(after_noops)
    result.append("No-op Removal:")
    if noops_removed > 0:
        result.append(f"  ✓ Removed {noops_removed} identity operation(s)")
        for noop in before_noops[:3]:  # Show first 3
            result.append(f"    - Removed {noop}")
    else:
        result.append("  - No identity operations removed")
    result.append("")

    # Performance impact
    speedup_estimate = min(reduction_pct * 0.8, 25)  # Conservative estimate
    memory_saved = nodes_removed * 0.5  # Rough estimate in KB

    result.extend([
        "Performance Impact:",
        f"  - Estimated speedup: {speedup_estimate:.1f}%",
        f"  - Memory saved: ~{memory_saved:.1f} KB",
        f"  - Overall: {'Significant' if reduction_pct > 15 else 'Moderate' if reduction_pct > 5 else 'Minor'} optimization"
    ])

    return '\n'.join(result)
