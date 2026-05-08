"""
Dynamo Stage Parsers

Parse TORCH_LOGS output and debug files from Dynamo compilation stage.
"""

import re
from collections import Counter


async def parse_graph_breaks(log_content: str) -> str:
    """
    Parse TORCH_LOGS="graph_breaks" stdout output.

    Args:
        log_content: Stdout from running with TORCH_LOGS="graph_breaks"

    Returns:
        Formatted analysis of graph breaks
    """
    breaks = []
    lines = log_content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('Graph break:'):
            # Extract operation
            op_match = re.search(r'Graph break:\s+(.+)$', line)
            operation = op_match.group(1).strip() if op_match else "unknown"

            # Look ahead for Reason and location
            reason = "unknown"
            location = "unknown"

            for j in range(i+1, min(i+5, len(lines))):
                next_line = lines[j].strip()

                if next_line.startswith('Reason:'):
                    reason_match = re.search(r'Reason:\s+(.+)$', next_line)
                    if reason_match:
                        reason = reason_match.group(1).strip()

                elif next_line.startswith('User code:') or next_line.startswith('File'):
                    loc_match = re.search(r'(?:User code:|File)\s+(.+)$', next_line)
                    if loc_match:
                        location = loc_match.group(1).strip()

            breaks.append({
                "operation": operation,
                "reason": reason,
                "location": location
            })

        i += 1

    if not breaks:
        return "No graph breaks found."

    # Categorize breaks
    categories = Counter(_categorize_break(b["reason"]) for b in breaks)

    result = f"## Graph Breaks Analysis\n\n"
    result += f"**Total breaks:** {len(breaks)}\n\n"
    result += f"**By category:**\n"
    for category, count in categories.most_common():
        result += f"- {category}: {count}\n"
    result += f"\n**Details:**\n"
    for i, b in enumerate(breaks, 1):
        result += f"\n{i}. **{b['operation']}**\n"
        result += f"   - Reason: {b['reason']}\n"
        result += f"   - Location: {b['location']}\n"

    return result


def _categorize_break(reason: str) -> str:
    """Categorize break reason."""
    reason_lower = reason.lower()

    if 'data-dependent' in reason_lower or 'item()' in reason_lower:
        return "Data-dependent operation"
    elif 'dynamic control' in reason_lower or 'if' in reason_lower:
        return "Dynamic control flow"
    elif 'skip list' in reason_lower or 'print' in reason_lower:
        return "Unsupported operation"
    elif 'mutation' in reason_lower:
        return "In-place mutation"
    else:
        return "Other"


async def parse_fx_graph(graph_content: str) -> str:
    """
    Parse FX graph file content (fx_graph_readable.py from TORCH_LOGS="dynamo").

    Args:
        graph_content: Content of fx_graph_readable.py file

    Returns:
        Formatted analysis of FX graph structure
    """
    # Count operations
    ops = re.findall(r'torch\.ops\.(\w+)\.(\w+)', graph_content)
    op_counts = Counter(f"{namespace}.{op}" for namespace, op in ops)

    # Find placeholders (inputs)
    placeholders = re.findall(r'%(\w+):\s*\[.*?\]\s*=\s*placeholder', graph_content)

    # Find outputs
    outputs = re.findall(r'return\s+\[(.*?)\]', graph_content, re.DOTALL)

    result = f"## FX Graph Analysis\n\n"
    result += f"**Inputs:** {len(placeholders)}\n"
    if placeholders:
        result += f"  - {', '.join(placeholders)}\n"
    result += f"\n**Operations:** {sum(op_counts.values())}\n"
    for op, count in op_counts.most_common(10):
        result += f"  - {op}: {count}\n"
    if len(op_counts) > 10:
        result += f"  - ... and {len(op_counts) - 10} more\n"
    result += f"\n**Outputs:** {len(outputs)}\n"

    return result


async def parse_pre_grad_passes(before_content: str, after_content: str) -> str:
    """
    Parse pre-grad pass effects by comparing before/after FX graphs.

    Args:
        before_content: Content of fx_graph_readable.py (before passes)
        after_content: Content of fx_graph_transformed.py (after passes)

    Returns:
        Formatted analysis of what changed
    """
    # Count operations in each
    before_ops = re.findall(r'torch\.ops\.(\w+)\.(\w+)', before_content)
    after_ops = re.findall(r'torch\.ops\.(\w+)\.(\w+)', after_content)

    before_count = len(before_ops)
    after_count = len(after_ops)

    result = f"## Pre-Grad Pass Analysis\n\n"
    result += f"**Before:** {before_count} operations\n"
    result += f"**After:** {after_count} operations\n"
    result += f"**Change:** {after_count - before_count:+d} operations\n\n"

    if before_count == after_count:
        result += "No operation count change (passes may have modified operation types)\n"
    elif after_count < before_count:
        result += f"**Optimizations applied:** {before_count - after_count} operations eliminated\n"
    else:
        result += f"**Operations expanded:** {after_count - before_count} operations added\n"

    return result
