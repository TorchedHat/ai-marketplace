"""
Dynamo Stage Analyzers

Parsers for:
- Graph breaks (TORCH_LOGS="graph_breaks")
- FX graph structure (fx_graph_*.py)
- Pre-grad pass effects (before/after comparison)
"""

from collections import Counter
from pathlib import Path

try:
    from pydantic import BaseModel

    class GraphBreak(BaseModel):
        """Graph break information."""
        location: str  # file:line
        reason: str
        operation: str
        graph_count: int

    class FXNode(BaseModel):
        """FX graph node information."""
        name: str
        op: str  # placeholder, call_function, call_module, get_attr, output
        target: str | None = None
        args: list[str] = []
except ImportError:
    # Pydantic not available or incompatible
    GraphBreak = None
    FXNode = None


async def parse_graph_breaks(log_content: str) -> str:
    """
    Parse TORCH_LOGS="graph_breaks" output.

    Example log format:
    ```
    Graph break: print(y)
      Reason: call_function print in skip list
      User code: /path/to/file.py:5 in fn
      Graph Count: 2
    ```
    """
    import re

    breaks: list[GraphBreak] = []

    lines = log_content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for "Graph break:" line
        if line.startswith('Graph break:'):
            # Extract operation
            op_match = re.search(r'Graph break:\s+(.+)$', line)
            operation = op_match.group(1).strip() if op_match else "unknown"

            # Look ahead for Reason and User code
            reason = "unknown"
            location = "unknown"
            graph_count = 1

            for j in range(i+1, min(i+5, len(lines))):
                next_line = lines[j].strip()

                if next_line.startswith('Reason:'):
                    reason_match = re.search(r'Reason:\s+(.+)$', next_line)
                    if reason_match:
                        reason = reason_match.group(1).strip()

                elif next_line.startswith('User code:'):
                    loc_match = re.search(r'User code:\s+(.+)$', next_line)
                    if loc_match:
                        location = loc_match.group(1).strip()

                elif next_line.startswith('Graph Count:'):
                    count_match = re.search(r'Graph Count:\s+(\d+)', next_line)
                    if count_match:
                        graph_count = int(count_match.group(1))

            breaks.append(GraphBreak(
                location=location,
                reason=reason,
                operation=operation,
                graph_count=graph_count
            ))

            i += 1
        else:
            i += 1

    # Categorize by reason
    reasons = {}
    for break_item in breaks:
        reason = _categorize_break_reason(break_item.reason)
        reasons[reason] = reasons.get(reason, 0) + 1

    return f"""Graph Break Analysis:

Total Breaks: {len(breaks)}
Final Graph Count: {breaks[-1].graph_count if breaks else 1}
Impact: Code split into {breaks[-1].graph_count if breaks else 1} separate compiled graphs

Breaks by Category:
{_format_reasons(reasons)}

Individual Breaks:
{_format_breaks(breaks[:10])}  # Show first 10

Recommendations:
{_suggest_fixes(breaks)}
"""


def _categorize_break_reason(reason: str) -> str:
    """Categorize break reason into common types."""
    reason_lower = reason.lower()

    if 'skip list' in reason_lower or 'print' in reason_lower:
        return "Unsupported operation (skip list)"
    elif 'dynamic control' in reason_lower or 'if' in reason_lower or 'while' in reason_lower:
        return "Dynamic control flow"
    elif 'data-dependent' in reason_lower or 'tensor.item()' in reason_lower:
        return "Data-dependent operation"
    elif 'mutation' in reason_lower:
        return "In-place mutation"
    elif 'generator' in reason_lower or 'yield' in reason_lower:
        return "Generator/iterator"
    elif 'external' in reason_lower or 'c extension' in reason_lower:
        return "External C extension"
    elif 'inline' in reason_lower:
        return "Cannot inline"
    else:
        return "Other"


async def analyze_fx_graph(fx_graph_path: str) -> str:
    """
    Analyze FX graph structure from fx_graph_readable.py.

    Extracts:
    - Node count and types
    - Operation types used
    - Graph complexity metrics
    - Potential optimization opportunities
    """
    import re
    from collections import Counter
    from pathlib import Path

    # Check file exists
    if not Path(fx_graph_path).exists():
        return f"Error: FX graph file not found: {fx_graph_path}"

    try:
        with open(fx_graph_path) as f:
            content = f.read()
    except Exception as e:
        return f"Error reading FX graph file: {e}"

    # Parse operations from the graph
    operations = []
    tensor_shapes = {}
    input_shapes = []
    output_shape = None

    # Extract function signature to get input shapes
    # Format: def forward(self, arg0_1: "f32[2, 4, 16, 16]"):
    sig_match = re.search(r'def forward\(self,\s*(.+?)\):', content)
    if sig_match:
        args_str = sig_match.group(1)
        # Extract argument shapes
        arg_matches = re.findall(r'(\w+):\s*"([^"]+)"', args_str)
        for arg_name, shape_dtype in arg_matches:
            input_shapes.append(f"{arg_name}: {shape_dtype}")

    # Parse each operation line
    # Format: var_name: "type[shape]" = torch.ops.aten.op_name.variant(args); ...
    op_pattern = re.compile(
        r'(\w+):\s*"([^"]+)"\s*=\s*torch\.ops\.(\w+)\.(\w+)\.(\w+)\(([^)]*)\)',
        re.MULTILINE
    )

    for match in op_pattern.finditer(content):
        var_name = match.group(1)
        shape_dtype = match.group(2)
        namespace = match.group(3)  # aten, prims, etc.
        op_name = match.group(4)
        # variant = match.group(5)  # Not currently used
        # args = match.group(6)  # Not currently used

        full_op = f"{namespace}.{op_name}"
        operations.append(full_op)
        tensor_shapes[var_name] = shape_dtype

    # Count operation types
    op_counts = Counter(operations)

    # Categorize operations
    pointwise_ops = []
    reduction_ops = []
    view_ops = []
    memory_ops = []
    other_ops = []

    for op in op_counts:
        op_lower = op.lower()
        if any(x in op_lower for x in ['add', 'mul', 'sub', 'div', 'relu', 'sigmoid', 'tanh', 'clamp', 'convert']):
            pointwise_ops.append(op)
        elif any(x in op_lower for x in ['sum', 'mean', 'max', 'min', 'argmax', 'argmin']):
            reduction_ops.append(op)
        elif any(x in op_lower for x in ['view', 'reshape', 'permute', 'transpose', 'squeeze', 'unsqueeze']):
            view_ops.append(op)
        elif any(x in op_lower for x in ['index', 'gather', 'scatter', 'iota']):
            memory_ops.append(op)
        else:
            other_ops.append(op)

    # Find return statement to get output
    # Format can be: return (var,) or return var
    # Look for actual return statement (not in comments)
    lines = content.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('return ') and '#' not in line[:line.find('return')] if 'return' in line else True:
            # Extract variable name from return (var,) or return var
            return_match = re.search(r'return\s+\(?([a-zA-Z_]\w*)', line)
            if return_match:
                output_var = return_match.group(1)
                if output_var in tensor_shapes:
                    output_shape = tensor_shapes[output_var]
                break

    # Calculate graph metrics
    total_ops = len(operations)
    unique_ops = len(op_counts)

    # Identify potential optimization opportunities
    optimizations = []

    # Check for fusable patterns
    if 'aten.add' in op_counts and 'aten.mul' in op_counts:
        optimizations.append("Potential for add-mul fusion (fused multiply-add)")

    if 'aten.relu' in op_counts:
        relu_count = op_counts['aten.relu']
        optimizations.append(f"ReLU operations ({relu_count}) can be fused with preceding ops")

    if 'aten.view' in op_counts or 'aten.reshape' in op_counts:
        optimizations.append("View/reshape ops are typically zero-cost (no data movement)")

    # Check for repeated operations
    repeated = [(op, count) for op, count in op_counts.items() if count > 5]
    if repeated:
        optimizations.append(f"Highly repeated ops: {', '.join(f'{op} ({count}x)' for op, count in repeated[:3])}")

    # Format output
    result = f"""FX Graph Analysis:

File: {fx_graph_path}

Inputs:
{_format_list(input_shapes) if input_shapes else '  [No inputs detected]'}

Output:
  {output_shape if output_shape else '[Not detected]'}

Operation Summary:
  Total operations: {total_ops}
  Unique operation types: {unique_ops}

Operation Categories:
  Pointwise/Elementwise: {len(pointwise_ops)} ops ({sum(op_counts[op] for op in pointwise_ops)} total)
    {_format_top_ops(op_counts, pointwise_ops)}

  View/Reshape: {len(view_ops)} ops ({sum(op_counts[op] for op in view_ops)} total)
    {_format_top_ops(op_counts, view_ops)}

  Memory Access: {len(memory_ops)} ops ({sum(op_counts[op] for op in memory_ops)} total)
    {_format_top_ops(op_counts, memory_ops)}

  Reduction: {len(reduction_ops)} ops ({sum(op_counts[op] for op in reduction_ops)} total)
    {_format_top_ops(op_counts, reduction_ops)}

  Other: {len(other_ops)} ops ({sum(op_counts[op] for op in other_ops)} total)
    {_format_top_ops(op_counts, other_ops)}

Most Common Operations:
{_format_op_counts(op_counts, top_n=10)}

Graph Characteristics:
  Complexity: {'High' if total_ops > 50 else 'Medium' if total_ops > 20 else 'Low'} ({total_ops} operations)
  Structure: {'Complex/branching' if unique_ops / max(total_ops, 1) > 0.5 else 'Sequential/linear'}
  Op diversity: {unique_ops / max(total_ops, 1):.1%} ({unique_ops} unique / {total_ops} total)

Optimization Opportunities:
{_format_list(optimizations) if optimizations else '  [None detected]'}
"""

    return result


def _format_list(items: list, indent: str = "  ") -> str:
    """Format a list of items with indentation."""
    if not items:
        return f"{indent}[None]"
    return "\n".join(f"{indent}- {item}" for item in items)


def _format_top_ops(op_counts: Counter, ops: list, top_n: int = 3) -> str:
    """Format top N operations from a category."""
    if not ops:
        return "[None]"

    # Sort by count and take top N
    sorted_ops = sorted([(op, op_counts[op]) for op in ops], key=lambda x: x[1], reverse=True)
    top_ops = sorted_ops[:top_n]

    if not top_ops:
        return "[None]"

    return ", ".join(f"{op} ({count})" for op, count in top_ops)


def _format_op_counts(op_counts: Counter, top_n: int = 10) -> str:
    """Format operation counts in descending order."""
    if not op_counts:
        return "  [None]"

    sorted_ops = op_counts.most_common(top_n)
    return "\n".join(f"  {i+1}. {op}: {count}" for i, (op, count) in enumerate(sorted_ops))


async def analyze_pre_grad_passes(fx_graph_path: str) -> str:
    """
    Analyze pre-grad pass effects by comparing before/after FX graphs.

    Automatically detects fx_graph_readable.py and fx_graph_transformed.py
    in the same directory and compares them.

    Args:
        fx_graph_path: Path to either fx_graph_readable.py or fx_graph_transformed.py

    Returns:
        Formatted analysis of pre-grad optimizations including:
        - Operation transformations (view -> reshape, etc.)
        - Node count changes
        - Detected optimization patterns
        - Performance impact estimation
    """
    from collections import Counter
    from pathlib import Path

    # Resolve paths to before/after files
    input_path = Path(fx_graph_path)

    if not input_path.exists():
        return f"Error: FX graph file not found: {fx_graph_path}"

    # Determine before and after paths
    parent_dir = input_path.parent
    before_path = parent_dir / "fx_graph_readable.py"
    after_path = parent_dir / "fx_graph_transformed.py"

    # Check which files exist
    before_exists = before_path.exists()
    after_exists = after_path.exists()

    if not before_exists and not after_exists:
        return f"Error: Neither fx_graph_readable.py nor fx_graph_transformed.py found in {parent_dir}"

    if not before_exists:
        return f"Error: No fx_graph_readable.py found for comparison in {parent_dir}"

    if not after_exists:
        # Only have readable, provide basic analysis
        return await _analyze_single_fx_graph(str(before_path))

    # Parse both graphs
    before_ops, before_nodes = _parse_fx_graph_operations(before_path)
    after_ops, after_nodes = _parse_fx_graph_operations(after_path)

    # Calculate differences
    before_count = len(before_ops)
    after_count = len(after_ops)
    reduction = before_count - after_count
    reduction_pct = (reduction / before_count * 100) if before_count > 0 else 0

    # Detect transformations
    transformations = _detect_transformations(before_nodes, after_nodes)

    # Detect optimization patterns
    optimizations = _detect_optimization_patterns(before_ops, after_ops, before_nodes, after_nodes)

    # Count operation type changes
    before_op_counts = Counter(before_ops)
    after_op_counts = Counter(after_ops)

    # Find added/removed/changed operations
    before_set = set(before_op_counts.keys())
    after_set = set(after_op_counts.keys())

    removed_ops = before_set - after_set
    added_ops = after_set - before_set
    changed_ops = []

    for op in before_set & after_set:
        if before_op_counts[op] != after_op_counts[op]:
            changed_ops.append((op, before_op_counts[op], after_op_counts[op]))

    # Estimate performance impact
    impact = _estimate_performance_impact(reduction, optimizations, transformations)

    # Format output
    result = f"""Pre-Grad Pass Analysis:

Files Compared:
  Before: {before_path.name}
  After:  {after_path.name}
  Directory: {parent_dir}

Node Count Changes:
  Before: {before_count} operations
  After:  {after_count} operations
  Reduction: {reduction} operations ({reduction_pct:.1f}%)

Detected Transformations:
{_format_transformations(transformations)}

Detected Optimizations:
{_format_optimizations(optimizations)}

Operation Changes:
  Removed operations ({len(removed_ops)} types):
{_format_op_list(removed_ops, before_op_counts)}

  Added operations ({len(added_ops)} types):
{_format_op_list(added_ops, after_op_counts)}

  Changed operation counts ({len(changed_ops)} types):
{_format_changed_ops(changed_ops)}

Performance Impact Estimate:
{impact}

Summary:
  {_generate_summary(reduction_pct, optimizations, transformations)}
"""

    return result


def _parse_fx_graph_operations(file_path: "Path") -> tuple[list[str], list[dict]]:
    """
    Parse FX graph file and extract operations and node details.

    Returns:
        Tuple of (operations list, nodes list with details)
    """
    import re

    try:
        with open(file_path) as f:
            content = f.read()
    except Exception:
        return [], []

    operations = []
    nodes = []

    # Parse each operation line
    # Format: var_name: "type[shape]" = torch.ops.namespace.op_name.variant(args); ...
    op_pattern = re.compile(
        r'(\w+):\s*"([^"]+)"\s*=\s*torch\.ops\.(\w+)\.(\w+)\.(\w+)\(([^)]*)\)',
        re.MULTILINE
    )

    for match in op_pattern.finditer(content):
        var_name = match.group(1)
        shape_dtype = match.group(2)
        namespace = match.group(3)
        op_name = match.group(4)
        variant = match.group(5)
        args = match.group(6)

        full_op = f"{namespace}.{op_name}"
        operations.append(full_op)

        nodes.append({
            'name': var_name,
            'shape': shape_dtype,
            'op': full_op,
            'namespace': namespace,
            'op_name': op_name,
            'variant': variant,
            'args': args
        })

    return operations, nodes


def _detect_transformations(before_nodes: list[dict], after_nodes: list[dict]) -> list[str]:
    """
    Detect specific transformations between before and after graphs.

    Common transformations:
    - view -> reshape (view canonicalization)
    - Operation reordering for better fusion
    - Constant folding
    """
    transformations = []

    # Create operation type counters
    before_op_types = Counter(node['op'] for node in before_nodes)
    after_op_types = Counter(node['op'] for node in after_nodes)

    # Check for view -> reshape transformation
    if before_op_types.get('aten.view', 0) > after_op_types.get('aten.view', 0):
        view_reduction = before_op_types.get('aten.view', 0) - after_op_types.get('aten.view', 0)
        reshape_increase = after_op_types.get('aten.reshape', 0) - before_op_types.get('aten.reshape', 0)
        if reshape_increase > 0:
            transformations.append(
                f"View canonicalization: {view_reduction} aten.view -> {reshape_increase} aten.reshape"
            )

    # Check for operation reordering (same ops, different order)
    if len(before_nodes) == len(after_nodes) and before_op_types == after_op_types:
        # Check if node order changed
        if [n['op'] for n in before_nodes] != [n['op'] for n in after_nodes]:
            transformations.append("Operation reordering for fusion optimization")

    # Check for constant operations
    before_constants = sum(1 for n in before_nodes if 'constant' in n['op'].lower())
    after_constants = sum(1 for n in after_nodes if 'constant' in n['op'].lower())
    if before_constants > after_constants:
        transformations.append(
            f"Constant folding: {before_constants - after_constants} constant ops eliminated"
        )

    return transformations


def _detect_optimization_patterns(
    before_ops: list[str],
    after_ops: list[str],
    before_nodes: list[dict],
    after_nodes: list[dict]
) -> list[str]:
    """
    Detect known optimization patterns.

    Patterns:
    - Conv-BN fusion
    - Split-Cat elimination
    - Add-Mul fusion
    - Redundant transpose elimination
    """
    optimizations = []

    before_counts = Counter(before_ops)
    after_counts = Counter(after_ops)

    # Conv-BN fusion detection
    before_bn = before_counts.get('aten.batch_norm', 0) + before_counts.get('aten._native_batch_norm_legit', 0)
    after_bn = after_counts.get('aten.batch_norm', 0) + after_counts.get('aten._native_batch_norm_legit', 0)
    if before_bn > after_bn:
        optimizations.append(f"Conv-BN fusion: {before_bn - after_bn} batch_norm ops eliminated")

    # Split-Cat elimination
    before_split = before_counts.get('aten.split', 0)
    after_split = after_counts.get('aten.split', 0)
    before_cat = before_counts.get('aten.cat', 0)
    after_cat = after_counts.get('aten.cat', 0)

    split_eliminated = before_split - after_split
    cat_eliminated = before_cat - after_cat
    if split_eliminated > 0 and cat_eliminated > 0:
        optimizations.append(
            f"Split-Cat elimination: {split_eliminated} split + {cat_eliminated} cat ops removed"
        )

    # Transpose elimination
    before_transpose = before_counts.get('aten.transpose', 0) + before_counts.get('aten.permute', 0)
    after_transpose = after_counts.get('aten.transpose', 0) + after_counts.get('aten.permute', 0)
    if before_transpose > after_transpose:
        optimizations.append(
            f"Transpose optimization: {before_transpose - after_transpose} transpose/permute ops reduced"
        )

    # View/reshape optimization
    before_views = before_counts.get('aten.view', 0) + before_counts.get('aten.reshape', 0)
    after_views = after_counts.get('aten.view', 0) + after_counts.get('aten.reshape', 0)
    if before_views > after_views:
        optimizations.append(f"View elimination: {before_views - after_views} view/reshape ops removed")

    return optimizations


def _estimate_performance_impact(
    reduction: int,
    optimizations: list[str],
    transformations: list[str]
) -> str:
    """
    Estimate performance impact of pre-grad optimizations.
    """
    impact_lines = []

    if reduction > 0:
        impact_lines.append(f"  - Operation count reduced by {reduction} nodes")
        if reduction < 5:
            impact_lines.append("    Minimal impact: <1% speedup expected")
        elif reduction < 15:
            impact_lines.append("    Moderate impact: 1-5% speedup expected")
        else:
            impact_lines.append("    Significant impact: 5-15% speedup expected")
    elif reduction < 0:
        impact_lines.append(f"  - Operation count increased by {abs(reduction)} nodes")
        impact_lines.append("    Likely transformation for better fusion (not a slowdown)")
    else:
        impact_lines.append("  - No change in operation count")

    if optimizations:
        impact_lines.append(f"  - {len(optimizations)} optimization patterns detected")
        if 'Conv-BN fusion' in str(optimizations):
            impact_lines.append("    Conv-BN fusion: 10-20% speedup on affected layers")
        if 'Split-Cat' in str(optimizations):
            impact_lines.append("    Split-Cat elimination: Memory bandwidth savings")

    if transformations:
        impact_lines.append(f"  - {len(transformations)} transformations applied")
        if 'canonicalization' in str(transformations).lower():
            impact_lines.append("    Canonicalization: Improves pattern matching for Inductor")

    if not impact_lines:
        impact_lines.append("  - No significant optimizations detected")

    return "\n".join(impact_lines)


def _format_transformations(transformations: list[str]) -> str:
    """Format transformation list."""
    if not transformations:
        return "  [No transformations detected]"
    return "\n".join(f"  - {t}" for t in transformations)


def _format_optimizations(optimizations: list[str]) -> str:
    """Format optimization list."""
    if not optimizations:
        return "  [No optimization patterns detected]"
    return "\n".join(f"  - {o}" for o in optimizations)


def _format_op_list(ops: set[str], counts: Counter, max_items: int = 5) -> str:
    """Format operation list with counts."""
    if not ops:
        return "    [None]"

    sorted_ops = sorted([(op, counts[op]) for op in ops], key=lambda x: x[1], reverse=True)
    lines = [f"    - {op}: {count}x" for op, count in sorted_ops[:max_items]]

    if len(sorted_ops) > max_items:
        lines.append(f"    ... and {len(sorted_ops) - max_items} more")

    return "\n".join(lines)


def _format_changed_ops(changed_ops: list[tuple[str, int, int]], max_items: int = 5) -> str:
    """Format changed operation counts."""
    if not changed_ops:
        return "    [None]"

    # Sort by absolute change
    sorted_ops = sorted(changed_ops, key=lambda x: abs(x[2] - x[1]), reverse=True)
    lines = []

    for op, before, after in sorted_ops[:max_items]:
        change = after - before
        sign = "+" if change > 0 else ""
        lines.append(f"    - {op}: {before} -> {after} ({sign}{change})")

    if len(sorted_ops) > max_items:
        lines.append(f"    ... and {len(sorted_ops) - max_items} more")

    return "\n".join(lines)


def _generate_summary(reduction_pct: float, optimizations: list[str], transformations: list[str]) -> str:
    """Generate summary text."""
    summary_parts = []

    if reduction_pct > 10:
        summary_parts.append(f"Significant optimization: {reduction_pct:.1f}% operation reduction")
    elif reduction_pct > 5:
        summary_parts.append(f"Moderate optimization: {reduction_pct:.1f}% operation reduction")
    elif reduction_pct > 0:
        summary_parts.append(f"Minor optimization: {reduction_pct:.1f}% operation reduction")
    elif reduction_pct < 0:
        summary_parts.append("Graph expansion for optimization (not a regression)")
    else:
        summary_parts.append("No operation count change")

    if optimizations:
        summary_parts.append(f"{len(optimizations)} optimization patterns applied")

    if transformations:
        summary_parts.append(f"{len(transformations)} graph transformations")

    if not optimizations and not transformations and abs(reduction_pct) < 1:
        summary_parts.append("Pre-grad passes made minimal changes")

    return " | ".join(summary_parts)


async def _analyze_single_fx_graph(fx_graph_path: str) -> str:
    """
    Analyze a single FX graph when no transformed version is available.
    """
    from pathlib import Path

    path = Path(fx_graph_path)
    ops, nodes = _parse_fx_graph_operations(path)

    return f"""Pre-Grad Pass Analysis (Single File):

File: {path.name}
Directory: {path.parent}

Note: No fx_graph_transformed.py found for comparison.
      Showing basic analysis of current graph.

Operation Summary:
  Total operations: {len(ops)}
  Unique operation types: {len(set(ops))}

Operation Breakdown:
{_format_op_counts(Counter(ops), top_n=10)}

Impact: Cannot estimate without before/after comparison.
        Run with TORCH_LOGS="pre_grad_graphs" to see transformations.
"""


def _format_reasons(reasons: dict[str, int]) -> str:
    """Format break reasons for display."""
    return "\n".join(f"  - {reason}: {count}" for reason, count in reasons.items())


def _format_breaks(breaks: list[GraphBreak]) -> str:
    """Format individual breaks."""
    lines = []
    for i, b in enumerate(breaks, 1):
        lines.append(f"""
Break #{i}:
  Location: {b.location}
  Operation: {b.operation}
  Reason: {b.reason}
  Graph count after: {b.graph_count}
""")
    return "\n".join(lines)


def _suggest_fixes(breaks: list[GraphBreak]) -> str:
    """Suggest fixes based on break reasons."""
    suggestions = []
    seen_categories = set()

    for break_item in breaks:
        category = _categorize_break_reason(break_item.reason)

        # Only suggest once per category
        if category in seen_categories:
            continue
        seen_categories.add(category)

        if 'skip list' in category or 'print' in break_item.operation.lower():
            suggestions.append(
                "• Unsupported operations (print, etc.):\n"
                "  - Remove from compiled region\n"
                "  - Or use torch._dynamo.allow_in_graph() to whitelist"
            )

        elif 'Dynamic control' in category:
            suggestions.append(
                "• Dynamic control flow:\n"
                "  - Replace if/while with torch.cond() for conditional execution\n"
                "  - Or use torch._dynamo.graph_break() explicitly where needed"
            )

        elif 'Data-dependent' in category:
            suggestions.append(
                "• Data-dependent operations (.item(), indexing):\n"
                "  - Avoid tensor.item() - keep tensors symbolic\n"
                "  - Use torch operations instead of Python control flow"
            )

        elif 'mutation' in category:
            suggestions.append(
                "• In-place mutations:\n"
                "  - Use out-of-place operations when possible\n"
                "  - Or ensure mutation pattern is supported"
            )

        elif 'Generator' in category:
            suggestions.append(
                "• Generators/iterators:\n"
                "  - Convert to list before passing to compiled region\n"
                "  - Or materialize iterator outside compilation"
            )

        elif 'External' in category:
            suggestions.append(
                "• External C extensions:\n"
                "  - Register custom op if possible\n"
                "  - Or move to eager execution outside compiled region"
            )

    if not suggestions:
        suggestions.append("• No specific issues detected - breaks may be intentional")

    return "\n\n".join(suggestions)
