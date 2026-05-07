"""
Inductor Stage Analyzers

Parsers for:
- Lowering (ATen → IR nodes)
- Fusion decisions (scheduler)
- LoopBody (ops.* operations)
- Triton codegen
"""

import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel

    class FusionDecision(BaseModel):
        """Fusion decision information."""
        node1: str
        node2: str
        can_fuse: bool
        reason: str
        fusion_type: str | None = None  # vertical, horizontal, reduction
        iteration_space: str | None = None
except ImportError:
    # Pydantic not available or incompatible
    FusionDecision = None


async def analyze_lowering(ir_path: str, fx_graph_path: str) -> str:
    """
    Analyze ATen → IR node lowering.

    Parses FX graph for aten operations and IR file for Inductor IR nodes,
    then maps which aten ops became which IR operations.

    Args:
        ir_path: Path to ir_pre_fusion.txt file
        fx_graph_path: Path to fx_graph_readable.py file

    Returns:
        Formatted analysis string showing lowering transformation

    Raises:
        FileNotFoundError: If either file doesn't exist
    """

    # Validate files exist
    ir_file = Path(ir_path)
    fx_file = Path(fx_graph_path)

    if not ir_file.exists():
        raise FileNotFoundError(f"IR file not found: {ir_path}")
    if not fx_file.exists():
        raise FileNotFoundError(f"FX graph file not found: {fx_graph_path}")

    # Parse both files
    aten_ops = _parse_fx_graph_for_aten_ops(fx_file.read_text())
    ir_data = _parse_ir_file(ir_file.read_text())

    # Build the analysis report
    report_sections = []

    # Header
    report_sections.append("Lowering Analysis: ATen → Inductor IR")
    report_sections.append("=" * 60)
    report_sections.append(f"IR File: {ir_path}")
    report_sections.append(f"FX Graph: {fx_graph_path}")
    report_sections.append("")

    # ATen operations found
    report_sections.append("ATen Operations (from FX Graph):")
    report_sections.append("-" * 60)
    if aten_ops:
        for op in aten_ops:
            report_sections.append(f"  - {op['name']}")
            if op.get('args'):
                report_sections.append(f"    Args: {op['args']}")
    else:
        report_sections.append("  None found")
    report_sections.append("")

    # IR operations found
    report_sections.append("IR Operations (from ir_pre_fusion.txt):")
    report_sections.append("-" * 60)
    if ir_data['operations']:
        for op_type, count in ir_data['op_counts'].items():
            report_sections.append(f"  - {op_type}: {count}x")
        report_sections.append("")
        report_sections.append("  Operation details:")
        for op in ir_data['operations'][:10]:  # Show first 10
            report_sections.append(f"    {op}")
    else:
        report_sections.append("  None found")
    report_sections.append("")

    # Node type analysis
    report_sections.append("Node Type Analysis:")
    report_sections.append("-" * 60)
    for node_type, count in ir_data['node_types'].items():
        report_sections.append(f"  - {node_type}: {count}")
    report_sections.append("")

    # Buffers
    if ir_data['buffers']:
        report_sections.append("Buffers Created:")
        report_sections.append("-" * 60)
        for buf in ir_data['buffers']:
            report_sections.append(f"  - {buf}")
        report_sections.append("")

    # Loop variables/iteration space
    if ir_data['var_ranges']:
        report_sections.append("Iteration Space:")
        report_sections.append("-" * 60)
        for var_range in ir_data['var_ranges']:
            report_sections.append(f"  - {var_range}")
        report_sections.append("")

    # Lowering map (aten → IR)
    report_sections.append("Lowering Map: ATen → IR Operations")
    report_sections.append("-" * 60)
    lowering_map = _map_aten_to_ir(aten_ops, ir_data)
    if lowering_map:
        for aten_op, ir_ops in lowering_map.items():
            report_sections.append(f"  {aten_op} →")
            for ir_op in ir_ops:
                report_sections.append(f"    • {ir_op}")
    else:
        report_sections.append("  No direct mapping found (may be fused or decomposed)")
    report_sections.append("")

    # Missing/external lowerings
    report_sections.append("External/Missing Lowerings:")
    report_sections.append("-" * 60)
    external_ops = [nt for nt in ir_data['node_types'] if 'Extern' in nt]
    if external_ops:
        for ext_op in external_ops:
            report_sections.append(f"  - {ext_op}")
            report_sections.append("    Reason: No Inductor lowering registered, using external kernel")
    else:
        report_sections.append("  None - all operations successfully lowered to Inductor IR")

    return "\n".join(report_sections)


def _parse_fx_graph_for_aten_ops(content: str) -> list[dict[str, str]]:
    """
    Parse FX graph file to extract aten operations.

    Args:
        content: Contents of fx_graph_readable.py

    Returns:
        List of aten operation dictionaries with name and args
    """
    operations = []

    # Pattern: torch.ops.aten.{op_name}.{variant}(args)
    # Example: torch.ops.aten.add.Tensor(arg0_1, 1)
    pattern = r'torch\.ops\.aten\.(\w+)\.(\w+)\(([^)]+)\)'

    for match in re.finditer(pattern, content):
        op_name = match.group(1)
        variant = match.group(2)
        args = match.group(3)

        operations.append({
            'name': f'aten.{op_name}.{variant}',
            'base_op': op_name,
            'variant': variant,
            'args': args.strip()
        })

    return operations


def _parse_ir_file(content: str) -> dict[str, Any]:
    """
    Parse ir_pre_fusion.txt to extract IR nodes and operations.

    Args:
        content: Contents of ir_pre_fusion.txt

    Returns:
        Dictionary containing:
        - node_types: Dict mapping node type names to counts
        - operations: List of operation strings (ops.*)
        - op_counts: Dict mapping operation types to counts
        - buffers: List of buffer names
        - var_ranges: List of iteration variable ranges
    """
    node_types: dict[str, int] = {}
    operations: list[str] = []
    op_counts: dict[str, int] = {}
    buffers: list[str] = []
    var_ranges: list[str] = []

    lines = content.split('\n')

    for line in lines:
        line = line.strip()

        # Extract node types (e.g., "op0: SchedulerNode(ComputedBuffer)")
        node_type_match = re.search(r':\s*SchedulerNode\((\w+)\)', line)
        if node_type_match:
            node_type = node_type_match.group(1)
            node_types[node_type] = node_types.get(node_type, 0) + 1

        # Extract buffer names (e.g., "buf0: ComputedBuffer")
        buffer_match = re.search(r'(buf\d+):\s*\w+', line)
        if buffer_match and buffer_match.group(1) not in buffers:
            buffers.append(buffer_match.group(1))

        # Extract operations (ops.load, ops.add, ops.store, etc.)
        # Pattern: ops.{operation}(...)
        ops_pattern = r'(ops\.\w+)\('
        for ops_match in re.finditer(ops_pattern, line):
            op = ops_match.group(1)
            operations.append(op)

            # Count operation types
            op_type = op.replace('ops.', '')
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

        # Extract iteration variable ranges (e.g., "var_ranges = {p0: 10, p1: 100}")
        var_range_match = re.search(r'var_ranges\s*=\s*\{([^}]+)\}', line)
        if var_range_match:
            ranges = var_range_match.group(1)
            # Parse individual ranges (p0: 10, p1: 100)
            for range_item in ranges.split(','):
                range_item = range_item.strip()
                if ':' in range_item:
                    var_ranges.append(range_item)

    return {
        'node_types': node_types,
        'operations': operations,
        'op_counts': op_counts,
        'buffers': buffers,
        'var_ranges': var_ranges
    }


def _map_aten_to_ir(
    aten_ops: list[dict[str, str]],
    ir_data: dict[str, Any]
) -> dict[str, list[str]]:
    """
    Map aten operations to IR operations.

    Args:
        aten_ops: List of aten operation dictionaries
        ir_data: Parsed IR data

    Returns:
        Dictionary mapping aten op names to IR operation lists
    """
    mapping: dict[str, list[str]] = {}

    # Simple heuristic mapping based on operation names
    # In reality, the lowering is more complex and may involve decomposition
    for aten_op in aten_ops:
        base_op = aten_op['base_op']
        aten_name = aten_op['name']

        # Find IR operations that likely correspond to this aten op
        ir_ops = []

        # Common lowering patterns:
        # aten.add → ops.add
        # aten.mul → ops.mul
        # aten.relu → ops.maximum
        # etc.

        if base_op in ['add', 'sub', 'mul', 'div']:
            if base_op in ir_data['op_counts']:
                ir_ops.append(f"ops.{base_op}")
        elif base_op == 'relu':
            if 'maximum' in ir_data['op_counts']:
                ir_ops.append("ops.maximum")

        # All operations involve load/store
        if 'load' in ir_data['op_counts']:
            ir_ops.append("ops.load (memory read)")
        if 'store' in ir_data['op_counts']:
            ir_ops.append("ops.store (memory write)")

        if ir_ops:
            mapping[aten_name] = ir_ops

    return mapping


async def parse_fusion_decisions(log_content: str) -> str:
    """
    Parse TORCH_LOGS="fusion,schedule" output.

    Example formats:
    1. Successful fusion:
       ```
       FusionDecision: buf0 (Pointwise) <- producer
       FusionDecision: buf1 (Pointwise) <- consumer
         ✓ Ranges match: [10, 100]
         ✓ Vertical fusion (producer-consumer)
         → Fused into 1 kernel
       ```

    2. Failed fusion:
       ```
       FusionDecision: buf2 (Pointwise) <- producer
       FusionDecision: buf3 (Reduction) <- consumer
         ✗ Cannot fuse: Different iteration structure
       ```
    """
    decisions: list[FusionDecision] = []

    # Parse the log content line by line
    lines = log_content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for fusion decision pairs
        if line.startswith('FusionDecision:'):
            # Extract first node (producer)
            node1_match = re.search(r'FusionDecision:\s+(\w+)\s+\((\w+)\)\s+<-\s+producer', line)
            if node1_match and i + 1 < len(lines):
                node1 = node1_match.group(1)
                node1_type = node1_match.group(2)

                # Extract second node (consumer)
                next_line = lines[i + 1].strip()
                node2_match = re.search(r'FusionDecision:\s+(\w+)\s+\((\w+)\)\s+<-\s+consumer', next_line)
                if node2_match:
                    node2 = node2_match.group(1)
                    node2_type = node2_match.group(2)

                    # Parse decision details (next few lines)
                    decision = _parse_fusion_details(
                        lines[i+2:min(i+10, len(lines))],
                        node1, node2, node1_type, node2_type
                    )
                    decisions.append(decision)
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    # Analyze results
    successful = [d for d in decisions if d.can_fuse]
    failed = [d for d in decisions if not d.can_fuse]

    # Count fusion types
    fusion_types = {}
    for d in successful:
        ftype = d.fusion_type or "unknown"
        fusion_types[ftype] = fusion_types.get(ftype, 0) + 1

    # Count failure reasons
    failure_reasons = {}
    for d in failed:
        reason = d.reason
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return f"""Fusion Analysis:

Total Decisions Analyzed: {len(decisions)}
Successful Fusions: {len(successful)}
Failed Fusions: {len(failed)}
Success Rate: {len(successful) / len(decisions) * 100 if decisions else 0:.1f}%

Successful Fusion Types:
{_format_dict(fusion_types)}

Failed Fusion Reasons:
{_format_dict(failure_reasons)}

Detailed Successful Fusions:
{_format_successful_fusions(successful[:5])}  # Show first 5

Detailed Failed Fusions:
{_format_failed_fusions(failed[:5])}  # Show first 5

Impact Estimate:
- Intermediate buffers eliminated: {len(successful)}
- Kernel launches saved: ~{len(successful)}
- Memory bandwidth saved: ~{len(successful) * 4}KB (estimated)
"""


def _parse_fusion_details(
    detail_lines: list[str],
    node1: str,
    node2: str,
    node1_type: str,
    node2_type: str
) -> FusionDecision:
    """Parse the details section of a fusion decision."""

    can_fuse = False
    reason = "Unknown"
    fusion_type = None
    iteration_space = None

    for line in detail_lines:
        line = line.strip()

        # Check for success indicators
        if '✓' in line or 'Fused into' in line:
            can_fuse = True

            # Extract fusion type
            if 'Vertical fusion' in line or 'producer-consumer' in line:
                fusion_type = "vertical"
            elif 'Horizontal fusion' in line or 'multi-consumer' in line:
                fusion_type = "horizontal"
            elif 'Reduction fusion' in line:
                fusion_type = "reduction"

            # Extract iteration space
            ranges_match = re.search(r'Ranges match:\s*\[([^\]]+)\]', line)
            if ranges_match:
                iteration_space = ranges_match.group(1)

        # Check for failure indicators
        elif '✗' in line or 'Cannot fuse' in line:
            can_fuse = False

            # Extract failure reason
            if 'device mismatch' in line.lower():
                reason = "Device mismatch (CPU vs CUDA)"
            elif 'different iteration' in line.lower():
                reason = "Incompatible iteration spaces"
            elif 'dependency cycle' in line.lower():
                reason = "Circular dependency detected"
            elif 'extern kernel' in line.lower():
                reason = "External kernel boundary (matmul, conv)"
            elif 'memory constraint' in line.lower():
                reason = "Memory limit exceeded"
            else:
                # Try to extract reason from line
                reason_match = re.search(r'Cannot fuse:\s*(.+)$', line)
                if reason_match:
                    reason = reason_match.group(1).strip()

        # Stop parsing after we hit the next decision or empty line
        if line.startswith('FusionDecision:') or (not line and can_fuse is not None):
            break

    # If no explicit reason found for success, infer from types
    if can_fuse and not fusion_type:
        if node1_type == node2_type == "Pointwise":
            fusion_type = "vertical"
        elif node1_type == "Pointwise" and node2_type == "Reduction":
            fusion_type = "reduction"

    return FusionDecision(
        node1=node1,
        node2=node2,
        can_fuse=can_fuse,
        reason=reason,
        fusion_type=fusion_type,
        iteration_space=iteration_space
    )


def _format_dict(d: dict[str, int]) -> str:
    """Format dictionary for display."""
    if not d:
        return "  None"
    return "\n".join(f"  - {k}: {v}" for k, v in d.items())


def _format_successful_fusions(decisions: list[FusionDecision]) -> str:
    """Format successful fusion details."""
    if not decisions:
        return "  None"

    lines = []
    for i, d in enumerate(decisions, 1):
        lines.append(f"\n  {i}. {d.node1} + {d.node2}")
        lines.append(f"     Type: {d.fusion_type or 'unknown'}")
        if d.iteration_space:
            lines.append(f"     Iteration space: [{d.iteration_space}]")
        lines.append(f"     Benefit: Eliminates intermediate buffer {d.node1}")

    if len(decisions) > 5:
        lines.append(f"\n  ... and {len(decisions) - 5} more")

    return "\n".join(lines)


async def analyze_loopbody(ir_post_fusion_path: str) -> str:
    """
    Analyze LoopBody IR (ops.* operations).

    Analyzes:
    - Operation types (ops.load, ops.store, ops.relu, ops.reduction, etc.)
    - Memory patterns (sequential vs random access)
    - Loop structure (iteration variables, ranges)
    - Fusion results (FusedSchedulerNode showing what got fused)
    - Buffer usage (input/output buffers, intermediate buffers)

    Args:
        ir_post_fusion_path: Path to ir_post_fusion.txt file

    Returns:
        Formatted analysis string with operations, memory patterns, and fusion info
    """
    # Read the IR file
    file_path = Path(ir_post_fusion_path)
    if not file_path.exists():
        return f"ERROR: File not found: {ir_post_fusion_path}"

    content = file_path.read_text()
    lines = content.split('\n')

    # Initialize data structures
    fused_nodes: list[str] = []
    loop_bodies: list[dict[str, any]] = []
    operation_counts = Counter()
    buffer_usage: dict[str, dict[str, any]] = {}  # {loop_body: {reads: set(), writes: set()}}
    scheduler_nodes: list[str] = []

    # Parse state
    current_loop_body = None
    in_loop_body = False
    var_ranges = {}
    index_exprs: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parse FusedSchedulerNode
        if ': FusedSchedulerNode' in line:
            match = re.match(r'(\w+):\s*FusedSchedulerNode\((.+)\)', line)
            if match:
                node_name = match.group(1)
                constituents = match.group(2)
                fused_nodes.append(f"{node_name}: {constituents}")

        # Parse SchedulerNode (non-fused)
        elif ': SchedulerNode' in line and 'FusedSchedulerNode' not in line:
            match = re.match(r'(\w+):\s*SchedulerNode', line)
            if match:
                scheduler_nodes.append(match.group(1))

        # Parse loop body class definition
        elif line.startswith('class ') and '_loop_body:' in line:
            match = re.match(r'class (\w+):', line)
            if match:
                current_loop_body = match.group(1)
                in_loop_body = True
                var_ranges = {}
                index_exprs = []

        # Parse var_ranges
        elif in_loop_body and 'var_ranges = {' in line:
            # Extract iteration variables and ranges
            # Example: var_ranges = {p0: 8, p1: 32, p2: 32}
            match = re.search(r'var_ranges\s*=\s*\{([^}]+)\}', line)
            if match:
                ranges_str = match.group(1)
                for pair in ranges_str.split(','):
                    if ':' in pair:
                        var, val = pair.split(':')
                        var_ranges[var.strip()] = val.strip()

        # Parse index expressions
        elif in_loop_body and line.startswith('index') and '=' in line:
            # Example: index0 = p1
            # Example: index2 = 16*indirect0 + indirect1 + 256*p0
            match = re.match(r'(index\d+)\s*=\s*(.+)', line)
            if match:
                idx_name = match.group(1)
                idx_expr = match.group(2)
                index_exprs.append(f"{idx_name} = {idx_expr}")

        # Parse operations in loop body
        elif in_loop_body and ' = ops.' in line:
            # Extract operation type
            match = re.search(r'ops\.(\w+)\(', line)
            if match:
                op_type = match.group(1)
                operation_counts[f"ops.{op_type}"] += 1

                # Track buffer usage
                if current_loop_body not in buffer_usage:
                    buffer_usage[current_loop_body] = {'reads': set(), 'writes': set()}

                # Extract buffer names for load/store
                if op_type == 'load':
                    buf_match = re.search(r"ops\.load\('(\w+)'", line)
                    if buf_match:
                        buffer_usage[current_loop_body]['reads'].add(buf_match.group(1))
                elif op_type == 'store':
                    buf_match = re.search(r"ops\.store\('(\w+)'", line)
                    if buf_match:
                        buffer_usage[current_loop_body]['writes'].add(buf_match.group(1))

        # End of loop body class
        elif in_loop_body and (line.startswith('class ') or (not line and i > 0)):
            if current_loop_body and (var_ranges or index_exprs):
                loop_bodies.append({
                    'name': current_loop_body,
                    'var_ranges': var_ranges.copy(),
                    'index_exprs': index_exprs.copy(),
                    'num_indices': len(index_exprs)
                })

            # Check if starting new loop body
            if line.startswith('class ') and '_loop_body:' in line:
                match = re.match(r'class (\w+):', line)
                if match:
                    current_loop_body = match.group(1)
                    var_ranges = {}
                    index_exprs = []
            else:
                in_loop_body = False
                current_loop_body = None

        i += 1

    # Handle last loop body if file ends while parsing
    if current_loop_body and (var_ranges or index_exprs):
        loop_bodies.append({
            'name': current_loop_body,
            'var_ranges': var_ranges.copy(),
            'index_exprs': index_exprs.copy(),
            'num_indices': len(index_exprs)
        })

    # Analyze memory patterns
    all_index_exprs = [idx for lb in loop_bodies for idx in lb['index_exprs']]
    memory_patterns = _analyze_memory_patterns(all_index_exprs)

    # Build result
    result_parts = [
        "LoopBody Analysis:",
        f"\nFile: {ir_post_fusion_path}",
        f"\n{'=' * 70}",
    ]

    # Fusion Results Section
    if fused_nodes:
        result_parts.append("\n## Fusion Results:\n")
        result_parts.append(f"FusedSchedulerNodes found: {len(fused_nodes)}")
        for fused in fused_nodes:
            result_parts.append(f"  - {fused}")

    if scheduler_nodes:
        result_parts.append(f"\nUnfused SchedulerNodes: {len(scheduler_nodes)}")
        result_parts.append(f"  ({', '.join(scheduler_nodes)})")

    # Operations Section
    result_parts.append("\n## Operations:\n")
    if operation_counts:
        total_ops = sum(operation_counts.values())
        result_parts.append(f"Total operations: {total_ops}\n")

        # Group by category
        load_store_ops = {k: v for k, v in operation_counts.items() if 'load' in k or 'store' in k}
        arithmetic_ops = {k: v for k, v in operation_counts.items()
                         if k in ['ops.add', 'ops.sub', 'ops.mul', 'ops.div', 'ops.mod']}
        comparison_ops = {k: v for k, v in operation_counts.items()
                         if k in ['ops.maximum', 'ops.minimum', 'ops.eq', 'ops.ne', 'ops.lt', 'ops.gt']}
        dtype_ops = {k: v for k, v in operation_counts.items() if 'dtype' in k or 'index_expr' in k}
        other_ops = {k: v for k, v in operation_counts.items()
                    if k not in load_store_ops and k not in arithmetic_ops
                    and k not in comparison_ops and k not in dtype_ops}

        if load_store_ops:
            result_parts.append("Memory Operations:")
            for op, count in sorted(load_store_ops.items(), key=lambda x: -x[1]):
                result_parts.append(f"  - {op}: {count}")

        if arithmetic_ops:
            result_parts.append("\nArithmetic Operations:")
            for op, count in sorted(arithmetic_ops.items(), key=lambda x: -x[1]):
                result_parts.append(f"  - {op}: {count}")

        if comparison_ops:
            result_parts.append("\nComparison/Min-Max Operations:")
            for op, count in sorted(comparison_ops.items(), key=lambda x: -x[1]):
                result_parts.append(f"  - {op}: {count}")

        if dtype_ops:
            result_parts.append("\nType Conversion/Index Operations:")
            for op, count in sorted(dtype_ops.items(), key=lambda x: -x[1]):
                result_parts.append(f"  - {op}: {count}")

        if other_ops:
            result_parts.append("\nOther Operations:")
            for op, count in sorted(other_ops.items(), key=lambda x: -x[1]):
                result_parts.append(f"  - {op}: {count}")
    else:
        result_parts.append("No operations found")

    # Loop Structure Section
    result_parts.append("\n## Loop Structure:\n")
    if loop_bodies:
        result_parts.append(f"Loop bodies found: {len(loop_bodies)}\n")
        for lb in loop_bodies:
            result_parts.append(f"{lb['name']}:")
            if lb['var_ranges']:
                result_parts.append(f"  Iteration variables: {lb['var_ranges']}")
                # Calculate total iterations
                try:
                    total_iters = 1
                    for _var, rng in lb['var_ranges'].items():
                        total_iters *= int(rng)
                    result_parts.append(f"  Total iterations: {total_iters:,}")
                except (ValueError, TypeError, KeyError):
                    pass
            result_parts.append(f"  Index expressions: {lb['num_indices']}")
    else:
        result_parts.append("No loop bodies found")

    # Buffer Usage Section
    result_parts.append("\n## Buffer Usage:\n")
    if buffer_usage:
        for loop_name, buffers in buffer_usage.items():
            result_parts.append(f"\n{loop_name}:")
            if buffers['reads']:
                result_parts.append(f"  Reads from: {', '.join(sorted(buffers['reads']))}")
            if buffers['writes']:
                result_parts.append(f"  Writes to: {', '.join(sorted(buffers['writes']))}")
    else:
        result_parts.append("No buffer usage found")

    # Memory Patterns Section
    result_parts.append("\n## Memory Patterns:\n")
    result_parts.append(memory_patterns)

    # Summary Section
    result_parts.append("\n## Summary:\n")
    result_parts.append(f"- Fused kernels: {len(fused_nodes)}")
    result_parts.append(f"- Loop bodies analyzed: {len(loop_bodies)}")
    result_parts.append(f"- Total operations: {sum(operation_counts.values())}")
    result_parts.append(f"- Unique buffer reads: {len(set().union(*[b['reads'] for b in buffer_usage.values()])) if buffer_usage else 0}")
    result_parts.append(f"- Unique buffer writes: {len(set().union(*[b['writes'] for b in buffer_usage.values()])) if buffer_usage else 0}")

    return '\n'.join(result_parts)


def _analyze_memory_patterns(index_exprs: list[str]) -> str:
    """
    Analyze memory access patterns from index expressions.

    Args:
        index_exprs: List of index expression strings

    Returns:
        Analysis of memory access patterns
    """
    if not index_exprs:
        return "No index expressions to analyze"

    # Categorize patterns
    sequential = []
    strided = []
    indirect = []
    complex_pattern = []

    for expr in index_exprs:
        expr_lower = expr.lower()

        # Check for indirect indexing
        if 'indirect' in expr_lower:
            indirect.append(expr)
        # Check for simple sequential (just p0, p1, etc)
        elif expr.count('p') == 1 and '+' not in expr and '*' not in expr:
            sequential.append(expr)
        # Check for strided access (multiplications)
        elif '*' in expr and 'indirect' not in expr_lower:
            strided.append(expr)
        else:
            complex_pattern.append(expr)

    result_lines = []

    if sequential:
        result_lines.append(f"Sequential access: {len(sequential)} expressions")
        result_lines.append(f"  Examples: {', '.join(sequential[:3])}")

    if strided:
        result_lines.append(f"\nStrided access: {len(strided)} expressions")
        result_lines.append(f"  Examples: {', '.join(strided[:3])}")
        result_lines.append("  (Suggests row-major or column-major access)")

    if indirect:
        result_lines.append(f"\nIndirect access: {len(indirect)} expressions")
        result_lines.append(f"  Examples: {', '.join(indirect[:3])}")
        result_lines.append("  (Gather/scatter pattern - may impact performance)")

    if complex_pattern:
        result_lines.append(f"\nComplex patterns: {len(complex_pattern)} expressions")
        result_lines.append(f"  Examples: {', '.join(complex_pattern[:3])}")

    # Overall assessment
    result_lines.append("\nAccess pattern assessment:")
    if len(sequential) > len(strided) + len(indirect):
        result_lines.append("  Primarily sequential - good cache locality expected")
    elif len(indirect) > len(sequential) + len(strided):
        result_lines.append("  Primarily indirect - potential memory bottleneck")
    elif strided:
        result_lines.append("  Mixed access patterns - cache performance depends on stride")
    else:
        result_lines.append("  Complex access patterns - detailed analysis recommended")

    return '\n'.join(result_lines)




async def analyze_triton_codegen(output_code_content: str) -> str:
    """
    Analyze generated Triton kernel code.

    Extracts kernel metadata, tiling configuration, memory patterns, and
    optimization hints from generated Triton source code.

    Args:
        output_code_content: Content of generated output_code.py file

    Returns:
        Formatted analysis report with kernel info, tiling, memory patterns
    """
    # Extract kernels and their metadata
    kernels = _extract_triton_kernels(output_code_content)

    # Analyze source node mappings
    source_mappings = _extract_source_mappings(output_code_content)

    # Summary statistics
    total_kernels = len(kernels)
    kernel_names = [k["name"] for k in kernels]

    # Categorize kernels by type
    pointwise_kernels = [k for k in kernels if "poi" in k["name"]]
    reduction_kernels = [k for k in kernels if "per" in k["name"] or "red" in k["name"]]

    # Build detailed kernel analysis
    kernel_details = []
    for kernel in kernels[:5]:  # Show first 5 in detail
        details = _format_kernel_details(kernel)
        kernel_details.append(details)

    # Build report
    report_lines = [
        "Triton Codegen Analysis:",
        "",
        f"Total Kernels Generated: {total_kernels}",
        f"- Pointwise (poi): {len(pointwise_kernels)}",
        f"- Reduction (per/red): {len(reduction_kernels)}",
        "",
        "Kernel Names:",
    ]

    for i, name in enumerate(kernel_names, 1):
        report_lines.append(f"  {i}. {name}")

    if source_mappings:
        report_lines.append("")
        report_lines.append("Source Node Mappings:")
        for mapping in source_mappings[:5]:  # Show first 5
            report_lines.append(f"  - {mapping}")

    if kernel_details:
        report_lines.append("")
        report_lines.append("Detailed Kernel Analysis:")
        for detail in kernel_details:
            report_lines.append(detail)

    # Performance insights
    insights = _generate_performance_insights(kernels)
    if insights:
        report_lines.append("")
        report_lines.append("Performance Insights:")
        for insight in insights:
            report_lines.append(f"  - {insight}")

    return "\n".join(report_lines)


def _extract_triton_kernels(code: str) -> list[dict[str, any]]:
    """
    Extract Triton kernel definitions from code.

    Returns list of kernel metadata dicts containing name, tiling params,
    decorators, and optimization hints.
    """
    kernels = []
    lines = code.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for kernel definitions
        if line.startswith('@triton.jit') or '@triton_heuristics' in line:
            kernel = _parse_kernel_definition(lines, i)
            if kernel:
                kernels.append(kernel)
                i = kernel.get("end_line", i + 1)
        else:
            i += 1

    return kernels


def _parse_kernel_definition(lines: list[str], start_idx: int) -> dict[str, any] | None:
    """
    Parse a single kernel definition starting at decorator line.

    Extracts name, tiling parameters, hints, and signature from kernel code.
    """
    kernel_info = {
        "decorators": [],
        "tiling_params": {},
        "hints": {},
        "loads": 0,
        "stores": 0,
        "start_line": start_idx,
    }

    # Collect decorators
    i = start_idx
    while i < len(lines) and (lines[i].strip().startswith('@') or
                               'triton_heuristics' in lines[i]):
        decorator_line = lines[i].strip()
        kernel_info["decorators"].append(decorator_line)

        # Extract hints from decorator
        if 'triton_heuristics' in decorator_line:
            # Look for size_hints, reduction_hint, etc.
            if 'size_hints' in decorator_line:
                hints = re.search(r"size_hints=\{([^}]+)\}", decorator_line)
                if hints:
                    kernel_info["hints"]["size_hints"] = hints.group(1)
            if 'reduction_hint' in decorator_line:
                hint = re.search(r"reduction_hint=(\w+\.\w+)", decorator_line)
                if hint:
                    kernel_info["hints"]["reduction_hint"] = hint.group(1)

        i += 1

    # Find function definition
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('def ') and '(' in line:
            # Extract kernel name
            name_match = re.search(r'def (\w+)\(', line)
            if name_match:
                kernel_info["name"] = name_match.group(1)

            # Extract parameters (signature)
            sig_lines = [line]
            paren_count = line.count('(') - line.count(')')
            j = i + 1
            while j < len(lines) and paren_count > 0:
                sig_lines.append(lines[j].strip())
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1
            kernel_info["signature"] = ' '.join(sig_lines)

            # Parse kernel body for tiling params and operations
            body_end = _parse_kernel_body(lines, j, kernel_info)
            kernel_info["end_line"] = body_end
            break
        i += 1

    return kernel_info if "name" in kernel_info else None


def _parse_kernel_body(lines: list[str], start_idx: int, kernel_info: dict[str, any]) -> int:
    """
    Parse kernel body to extract tiling params and operation counts.

    Updates kernel_info dict in place with XBLOCK, RBLOCK, load/store counts.
    """
    i = start_idx
    indent_level = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Determine initial indent level
        if indent_level is None and stripped and not stripped.startswith('#'):
            indent_level = len(line) - len(line.lstrip())

        # Stop at next function definition or dedent
        if stripped.startswith('def ') and i > start_idx:
            break
        if indent_level is not None and stripped and not stripped.startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent < indent_level:
                break

        # Extract tiling parameters
        if 'XBLOCK' in stripped and ':' in stripped and 'constexpr' in stripped:
            match = re.search(r'XBLOCK:\s*tl\.constexpr\s*=\s*(\d+)', stripped)
            if match:
                kernel_info["tiling_params"]["XBLOCK"] = int(match.group(1))

        if 'RBLOCK' in stripped and ':' in stripped and 'constexpr' in stripped:
            match = re.search(r'RBLOCK:\s*tl\.constexpr\s*=\s*(\d+)', stripped)
            if match:
                kernel_info["tiling_params"]["RBLOCK"] = int(match.group(1))

        if 'R0_BLOCK' in stripped and ':' in stripped and 'constexpr' in stripped:
            match = re.search(r'R0_BLOCK:\s*tl\.constexpr\s*=\s*(\d+)', stripped)
            if match:
                kernel_info["tiling_params"]["R0_BLOCK"] = int(match.group(1))

        # Count memory operations
        if 'tl.load(' in stripped:
            kernel_info["loads"] += stripped.count('tl.load(')
        if 'tl.store(' in stripped:
            kernel_info["stores"] += stripped.count('tl.store(')

        i += 1

    return i


def _extract_source_mappings(code: str) -> list[str]:
    """
    Extract source node to ATen operation mappings from comments.

    Returns list of mapping descriptions from generated code comments.
    """
    mappings = []
    lines = code.split('\n')

    for line in lines:
        # Look for "Original ATen:" comments
        if 'Original ATen:' in line:
            mapping = line.strip().lstrip('#').strip()
            mappings.append(mapping)
        # Look for "Source node to ATen node mapping:" sections
        elif 'Source node to ATen node mapping:' in line:
            # This indicates a mapping section follows
            pass

    return mappings


def _format_kernel_details(kernel: dict[str, any]) -> str:
    """
    Format detailed kernel information for display.

    Returns multi-line string with kernel name, tiling, and operation counts.
    """
    lines = [f"\n  Kernel: {kernel['name']}"]

    # Tiling configuration
    if kernel["tiling_params"]:
        lines.append("    Tiling Configuration:")
        for param, value in kernel["tiling_params"].items():
            lines.append(f"      {param}: {value}")

    # Operation counts
    if kernel["loads"] > 0 or kernel["stores"] > 0:
        lines.append("    Memory Operations:")
        if kernel["loads"] > 0:
            lines.append(f"      tl.load: {kernel['loads']}")
        if kernel["stores"] > 0:
            lines.append(f"      tl.store: {kernel['stores']}")

    # Hints
    if kernel["hints"]:
        lines.append("    Optimization Hints:")
        for hint_name, hint_value in kernel["hints"].items():
            lines.append(f"      {hint_name}: {hint_value}")

    return "\n".join(lines)


def _generate_performance_insights(kernels: list[dict[str, any]]) -> list[str]:
    """
    Generate performance insights based on kernel analysis.

    Returns list of actionable performance observations.
    """
    insights = []

    total_kernels = len(kernels)
    total_loads = sum(k.get("loads", 0) for k in kernels)
    total_stores = sum(k.get("stores", 0) for k in kernels)

    # Insight: Kernel count
    if total_kernels == 1:
        insights.append("Single kernel - excellent fusion achieved")
    elif total_kernels <= 3:
        insights.append(f"{total_kernels} kernels - good fusion")
    elif total_kernels > 5:
        insights.append(f"{total_kernels} kernels - consider improving fusion")

    # Insight: Memory operations
    if total_loads > 0 and total_stores > 0:
        load_store_ratio = total_loads / total_stores
        if load_store_ratio > 2:
            insights.append(f"Load/store ratio {load_store_ratio:.1f}:1 - compute-bound favorable")
        elif load_store_ratio < 1:
            insights.append(f"Load/store ratio {load_store_ratio:.1f}:1 - may be memory-bound")

    # Insight: Tiling
    tile_sizes = []
    for k in kernels:
        if "XBLOCK" in k.get("tiling_params", {}):
            tile_sizes.append(k["tiling_params"]["XBLOCK"])

    if tile_sizes and all(s >= 256 for s in tile_sizes):
        insights.append("Large tile sizes (>=256) - good for memory coalescing")

    return insights


def _format_failed_fusions(decisions: list[FusionDecision]) -> str:
    """Format failed fusion attempts with reasons and suggestions."""
    if not decisions:
        return "  None"

    lines = []
    for i, d in enumerate(decisions, 1):
        lines.append(f"\n  {i}. {d.node1} + {d.node2}")
        lines.append(f"     Reason: {d.reason}")

        # Add fix suggestions based on reason
        suggestion = _suggest_fusion_fix(d.reason)
        if suggestion:
            lines.append(f"     Fix: {suggestion}")

    if len(decisions) > 5:
        lines.append(f"\n  ... and {len(decisions) - 5} more")

    return "\n".join(lines)


def _suggest_fusion_fix(reason: str) -> str | None:
    """Suggest fixes for fusion failures."""
    reason_lower = reason.lower()

    if 'device' in reason_lower:
        return "Ensure all tensors on same device (CPU vs CUDA)"
    elif 'iteration' in reason_lower:
        return "Operations have incompatible shapes or broadcast patterns"
    elif 'dependency' in reason_lower:
        return "Reorder operations to break circular dependency"
    elif 'extern' in reason_lower:
        return "External kernel (matmul, conv) blocks fusion - this is expected"
    elif 'memory' in reason_lower:
        return "Reduce intermediate buffer sizes or split computation"
    else:
        return None
