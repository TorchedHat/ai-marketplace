"""
AOT Autograd Stage Parsers

Parse TORCH_LOGS output and debug files from AOT compilation stage.
"""

import re
from collections import Counter


async def parse_aot_joint_graph(graph_content: str) -> str:
    """
    Parse AOT joint graph file (model__*__joint_*.py from TORCH_LOGS="aot_joint_graph").

    Args:
        graph_content: Content of joint graph file

    Returns:
        Formatted analysis of joint graph structure
    """
    # Count forward vs backward operations
    forward_ops = len(re.findall(r'#\s*forward', graph_content, re.IGNORECASE))
    backward_ops = len(re.findall(r'#\s*backward', graph_content, re.IGNORECASE))

    # Total operations
    all_ops = re.findall(r'torch\.ops\.(\w+)\.(\w+)', graph_content)
    op_counts = Counter(f"{ns}.{op}" for ns, op in all_ops)

    result = f"## AOT Joint Graph Analysis\n\n"
    result += f"**Total operations:** {len(all_ops)}\n"
    result += f"**Forward operations:** ~{forward_ops}\n"
    result += f"**Backward operations:** ~{backward_ops}\n\n"
    result += f"**Top operations:**\n"
    for op, count in op_counts.most_common(10):
        result += f"  - {op}: {count}\n"

    return result


async def parse_aot_graphs(forward_content: str, backward_content: str | None = None) -> str:
    """
    Parse partitioned AOT graphs (model__*__forward/backward_*.py from TORCH_LOGS="aot_graphs").

    Args:
        forward_content: Content of forward graph file
        backward_content: Content of backward graph file (optional)

    Returns:
        Formatted analysis of partitioned graphs
    """
    # Parse forward graph
    fwd_ops = re.findall(r'torch\.ops\.(\w+)\.(\w+)', forward_content)
    fwd_placeholders = len(re.findall(r'placeholder', forward_content))

    result = f"## AOT Partitioned Graphs Analysis\n\n"
    result += f"**Forward Graph:**\n"
    result += f"  - Operations: {len(fwd_ops)}\n"
    result += f"  - Inputs: {fwd_placeholders}\n"

    if backward_content:
        bwd_ops = re.findall(r'torch\.ops\.(\w+)\.(\w+)', backward_content)
        bwd_placeholders = len(re.findall(r'placeholder', backward_content))

        result += f"\n**Backward Graph:**\n"
        result += f"  - Operations: {len(bwd_ops)}\n"
        result += f"  - Inputs (saved activations): {bwd_placeholders}\n"
        result += f"\n**Memory:** {bwd_placeholders} activations saved for backward\n"

    return result


async def parse_post_grad_passes(log_content: str) -> str:
    """
    Parse post-grad pass output (TORCH_LOGS="post_grad_graphs" stdout or file).

    Args:
        log_content: Stdout or file content from post_grad_graphs logging

    Returns:
        Formatted analysis of post-grad optimizations
    """
    # Look for pass names
    passes = re.findall(r'Running pass:\s+(\w+)', log_content)

    # Look for optimization messages
    optimizations = re.findall(r'(Fused|Eliminated|Replaced|Optimized)\s+(.+)', log_content)

    result = f"## Post-Grad Pass Analysis\n\n"

    if passes:
        result += f"**Passes run:** {len(passes)}\n"
        for p in passes[:10]:
            result += f"  - {p}\n"
        if len(passes) > 10:
            result += f"  - ... and {len(passes) - 10} more\n"

    if optimizations:
        result += f"\n**Optimizations applied:** {len(optimizations)}\n"
        for action, desc in optimizations[:5]:
            result += f"  - {action}: {desc}\n"
        if len(optimizations) > 5:
            result += f"  - ... and {len(optimizations) - 5} more\n"

    if not passes and not optimizations:
        result += "No post-grad pass information found in logs.\n"

    return result
