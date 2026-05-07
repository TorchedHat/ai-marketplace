"""
AOT Autograd Stage Analyzers

Parsers for:
- Functionalization (mutation removal)
- Joint graph (forward + backward)
- Partitioning (splitting joint graph)
- Post-grad passes (optimization effects)
"""

import re
from collections import Counter
from pathlib import Path
from typing import Any

# Import post-grad pass analyzer from dedicated module
from .aot_post_grad import analyze_post_grad_passes as _analyze_post_grad_passes

try:
    from pydantic import BaseModel

    class PartitionInfo(BaseModel):
        """Partitioning analysis result."""
        saved_activations: list[str]
        recomputed_ops: list[str]
        memory_saved: int | None = None
except ImportError:
    # Pydantic not available or incompatible
    PartitionInfo = None


# Common in-place operations to detect
INPLACE_OPS = {
    'add_', 'sub_', 'mul_', 'div_', 'pow_',
    'abs_', 'neg_', 'sqrt_', 'exp_', 'log_',
    'floor_', 'ceil_', 'round_', 'trunc_',
    'relu_', 'sigmoid_', 'tanh_',
    'copy_', 'fill_', 'zero_', 'masked_fill_',
    'clamp_', 'clamp_min_', 'clamp_max_',
    'normal_', 'uniform_',
    'transpose_', 'permute_',
    'scatter_', 'masked_scatter_',
    'index_add_', 'index_copy_', 'index_fill_',
}

# Map in-place ops to functional equivalents
FUNCTIONAL_EQUIVALENT = {
    'add_': 'add',
    'sub_': 'sub',
    'mul_': 'mul',
    'div_': 'div',
    'pow_': 'pow',
    'abs_': 'abs',
    'neg_': 'neg',
    'sqrt_': 'sqrt',
    'exp_': 'exp',
    'log_': 'log',
    'floor_': 'floor',
    'ceil_': 'ceil',
    'round_': 'round',
    'trunc_': 'trunc',
    'relu_': 'relu',
    'sigmoid_': 'sigmoid',
    'tanh_': 'tanh',
    'copy_': 'clone',
    'fill_': 'full_like',
    'zero_': 'zeros_like',
    'clamp_': 'clamp',
    'clamp_min_': 'clamp_min',
    'clamp_max_': 'clamp_max',
    'normal_': 'normal',
    'uniform_': 'uniform',
    'transpose_': 'transpose',
    'permute_': 'permute',
}


async def analyze_functionalization(graph_path: str) -> str:
    """
    Analyze functionalization results in AOT graph.

    Parses graph file to detect in-place operations, verify conversion to
    functional equivalents, and show transformation status.

    Args:
        graph_path: Path to AOT graph file (fx_graph_readable.py)

    Returns:
        Formatted analysis string showing functionalization status

    Raises:
        FileNotFoundError: If graph file doesn't exist
    """
    graph_file = Path(graph_path)

    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    content = graph_file.read_text()

    # Parse operations from graph
    inplace_ops = _find_inplace_operations(content)
    all_ops = _find_all_operations(content)
    tensor_types = _extract_tensor_types(content)

    # Analyze functionalization status
    is_clean = len(inplace_ops) == 0
    total_ops = len(all_ops)

    # Build report
    lines = [
        "Functionalization Analysis:",
        "",
        f"File: {graph_file.name}",
        f"Path: {graph_path}",
        "",
        "=" * 60,
        "OPERATION SUMMARY",
        "=" * 60,
        f"Total Operations: {total_ops}",
        f"In-Place Operations: {len(inplace_ops)}",
        f"Functional Operations: {total_ops - len(inplace_ops)}",
        ""
    ]

    # Show in-place operations found
    if inplace_ops:
        lines.extend([
            "=" * 60,
            "IN-PLACE OPERATIONS DETECTED",
            "=" * 60,
            ""
        ])

        # Group by operation type
        op_counts = Counter(op['op_name'] for op in inplace_ops)

        lines.append("By Type:")
        for op_name, count in op_counts.most_common():
            functional = FUNCTIONAL_EQUIVALENT.get(op_name, '?')
            lines.append(f"  ✗ {op_name:20s} → {functional:20s} ({count}x)")
        lines.append("")

        lines.append("Detailed Occurrences:")
        for op in inplace_ops:
            lines.append(
                f"  Line {op['line']:3d}: {op['var_name']:15s} = "
                f"{op['op_name']}(...) : {op['type']}"
            )
        lines.append("")

    # Show verification status
    lines.extend([
        "=" * 60,
        "FUNCTIONALIZATION VERIFICATION",
        "=" * 60,
        ""
    ])

    if is_clean:
        lines.extend([
            "✓ All in-place operations converted to functional equivalents",
            "✓ Graph is properly functionalized",
            ""
        ])
    else:
        lines.extend([
            f"✗ Found {len(inplace_ops)} in-place operation(s)",
            "✗ Functionalization incomplete or graph shows pre-functionalization state",
            "",
            "Recommended Actions:",
        ])

        # Show transformations needed
        unique_ops = set(op['op_name'] for op in inplace_ops)
        for op_name in sorted(unique_ops):
            functional = FUNCTIONAL_EQUIVALENT.get(op_name, 'unknown')
            lines.append(f"  - Replace {op_name} → {functional}")
        lines.append("")

    # Show tensor type summary
    if tensor_types:
        lines.extend([
            "=" * 60,
            "TENSOR TYPE SUMMARY",
            "=" * 60,
            ""
        ])

        type_counts = Counter(tensor_types.values())
        lines.append(f"Unique Tensor Types: {len(type_counts)}")
        for dtype, count in type_counts.most_common(10):
            lines.append(f"  - {dtype:30s} ({count}x)")
        lines.append("")

    # Show operation distribution
    lines.extend([
        "=" * 60,
        "OPERATION DISTRIBUTION",
        "=" * 60,
        ""
    ])

    op_types = Counter()
    for op in all_ops:
        # Extract operation family (e.g., aten.add.Tensor → add)
        op_name = op['op_name']
        # Remove variant suffix if present
        base_op = op_name.split('_')[0] if '_' in op_name else op_name
        op_types[base_op] += 1

    lines.append("Top Operations:")
    for op_name, count in op_types.most_common(15):
        pct = (count / total_ops * 100) if total_ops > 0 else 0
        lines.append(f"  {op_name:20s} {count:4d}x ({pct:5.1f}%)")
    lines.append("")

    return "\n".join(lines)


def _find_inplace_operations(content: str) -> list[dict[str, Any]]:
    """
    Find in-place operations in graph file.

    Args:
        content: Graph file content

    Returns:
        List of dicts with operation details
    """
    inplace_ops = []

    # Pattern: var_name: "type" = torch.ops.aten.op_name_(...)
    # Example: mul_: "f32[10, 100]" = torch.ops.aten.mul_.Tensor(arg0_1, 2)
    pattern = r'^(\s*)(\w+):\s*"([^"]+)"\s*=\s*torch\.ops\.(\w+)\.(\w+_)\.(\w+)\('

    for line_num, line in enumerate(content.split('\n'), 1):
        match = re.match(pattern, line)
        if match:
            indent, var_name, var_type, namespace, op_name, variant = match.groups()

            # Check if this is actually an in-place operation
            # (ends with '_' and is in our known list)
            if op_name in INPLACE_OPS:
                inplace_ops.append({
                    'line': line_num,
                    'var_name': var_name,
                    'type': var_type,
                    'namespace': namespace,
                    'op_name': op_name,
                    'variant': variant,
                    'full_op': f'{namespace}.{op_name}.{variant}',
                })

    return inplace_ops


def _find_all_operations(content: str) -> list[dict[str, Any]]:
    """
    Find all operations in graph file.

    Args:
        content: Graph file content

    Returns:
        List of dicts with operation details
    """
    operations = []

    # Pattern: var_name: "type" = torch.ops.namespace.op_name.variant(...)
    pattern = r'^(\s*)(\w+):\s*"([^"]+)"\s*=\s*torch\.ops\.(\w+)\.(\w+)\.(\w+)\('

    for line_num, line in enumerate(content.split('\n'), 1):
        match = re.match(pattern, line)
        if match:
            indent, var_name, var_type, namespace, op_name, variant = match.groups()

            operations.append({
                'line': line_num,
                'var_name': var_name,
                'type': var_type,
                'namespace': namespace,
                'op_name': op_name,
                'variant': variant,
                'full_op': f'{namespace}.{op_name}.{variant}',
            })

    return operations


def _extract_tensor_types(content: str) -> dict[str, str]:
    """
    Extract tensor variable names and their types.

    Args:
        content: Graph file content

    Returns:
        Dict mapping variable names to types
    """
    tensor_types = {}

    # Pattern: var_name: "type" = ...
    pattern = r'^(\s*)(\w+):\s*"([^"]+)"\s*='

    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            indent, var_name, var_type = match.groups()
            tensor_types[var_name] = var_type

    return tensor_types


async def analyze_joint_graph(joint_graph_path: str) -> str:
    """
    Analyze joint forward+backward graph or separate forward/backward graphs.

    Extracts:
    - Forward/backward operations
    - Input/output tensors
    - Saved activations for backward
    - Data flow between operations
    - Operation counts and types
    """
    import re
    from pathlib import Path

    # Check file exists
    graph_file = Path(joint_graph_path)
    if not graph_file.exists():
        return f"""Joint Graph Analysis:

File: {joint_graph_path}

Error: File not found

Please ensure the path points to a valid fx_graph_readable.py file.
"""

    # Read file content
    content = graph_file.read_text()

    # Determine if forward or backward graph
    is_forward = "forward" in joint_graph_path.lower()
    is_backward = "backward" in joint_graph_path.lower()
    graph_type = "Forward" if is_forward else ("Backward" if is_backward else "Joint")

    # Parse function signature to get inputs
    inputs: list[tuple[str, str]] = []
    input_pattern = r'def forward\(self,\s*([^)]+)\)'
    input_match = re.search(input_pattern, content)
    if input_match:
        input_params = input_match.group(1)
        # Parse individual parameters
        param_pattern = r'(\w+):\s*"([^"]+)"'
        inputs = [(m.group(1), m.group(2)) for m in re.finditer(param_pattern, input_params)]

    # Parse operations (call_function lines)
    operations: list[dict[str, str]] = []
    op_pattern = r'(\w+):\s*"([^"]+)"\s*=\s*torch\.ops\.(\S+)\(([^)]*)\)'
    for match in re.finditer(op_pattern, content, re.MULTILINE):
        var_name = match.group(1)
        var_type = match.group(2)
        op_name = match.group(3)
        args_str = match.group(4)

        operations.append({
            "var": var_name,
            "type": var_type,
            "op": op_name,
            "args": args_str.strip(),
        })

    # Parse return statement to get outputs
    outputs: list[str] = []
    return_pattern = r'return\s*\(([^)]+)\)'
    return_match = re.search(return_pattern, content)
    if return_match:
        outputs = [s.strip() for s in return_match.group(1).split(',')]

    # Build data flow by tracking variable usage
    data_flow: list[tuple[str, str, str]] = []
    for op in operations:
        # Extract variable references from args
        arg_vars = re.findall(r'\b(\w+)\b', op["args"])
        # Filter to actual variables (exclude numbers and keywords)
        arg_vars = [v for v in arg_vars if not v.isdigit() and v not in ["True", "False", "None"]]

        for arg_var in arg_vars:
            data_flow.append((arg_var, op["op"], op["var"]))

    # Identify saved activations (for forward graphs)
    saved_activations: list[str] = []
    if is_forward and outputs:
        # First output is user result, rest are saved for backward
        if len(outputs) > 1:
            saved_activations = outputs[1:]

    # Identify gradient inputs (for backward graphs)
    gradient_inputs: list[str] = []
    if is_backward and inputs:
        # Look for "tangents" in input names
        gradient_inputs = [name for name, _ in inputs if "tangent" in name]

    # Count operation types
    op_types: dict[str, int] = {}
    for op in operations:
        op_name = op["op"].split(".")[-1]  # Get last part (e.g., "default" -> actual op)
        full_op = op["op"]
        # Extract actual operation name
        if ".aten." in full_op:
            op_base = full_op.split(".aten.")[1].split(".")[0]
        else:
            op_base = full_op.split(".")[-2] if "." in full_op else full_op

        op_types[op_base] = op_types.get(op_base, 0) + 1

    # Format results
    result_parts = [
        "Joint Graph Analysis:",
        "",
        f"File: {joint_graph_path}",
        f"Type: {graph_type} Graph",
        "",
        "=== Inputs ===",
    ]

    if inputs:
        for name, dtype in inputs:
            input_type = "Gradient Input" if "tangent" in name else ("Parameter" if "primal" in name else "Input")
            result_parts.append(f"  {name}: {dtype} ({input_type})")
    else:
        result_parts.append("  (none)")

    result_parts.extend([
        "",
        f"=== Operations: {len(operations)} total ===",
    ])

    if operations:
        for i, op in enumerate(operations, 1):
            result_parts.append(f"  {i}. {op['var']} = {op['op']}")
    else:
        result_parts.append("  (none)")

    result_parts.extend([
        "",
        "=== Operation Types ===",
    ])
    for op_name, count in sorted(op_types.items()):
        result_parts.append(f"  {op_name}: {count}")

    result_parts.extend([
        "",
        "=== Outputs ===",
    ])
    if outputs:
        for i, out in enumerate(outputs):
            if i == 0:
                result_parts.append(f"  {out} (result)")
            else:
                result_parts.append(f"  {out} (saved for backward)")
    else:
        result_parts.append("  (none)")

    if saved_activations:
        result_parts.extend([
            "",
            f"=== Saved Activations for Backward: {len(saved_activations)} ===",
        ])
        for act in saved_activations:
            result_parts.append(f"  {act}")

    if gradient_inputs:
        result_parts.extend([
            "",
            f"=== Gradient Inputs: {len(gradient_inputs)} ===",
        ])
        for grad_in in gradient_inputs:
            result_parts.append(f"  {grad_in}")

    result_parts.extend([
        "",
        f"=== Data Flow: {len(data_flow)} edges ===",
    ])
    if data_flow:
        # Show first 10 data flow edges
        for src, op, dst in data_flow[:10]:
            result_parts.append(f"  {src} -> {op.split('.')[-2] if '.' in op else op} -> {dst}")
        if len(data_flow) > 10:
            result_parts.append(f"  ... and {len(data_flow) - 10} more edges")
    else:
        result_parts.append("  (none)")

    return "\n".join(result_parts)


async def analyze_partitioning(
    forward_path: str,
    backward_path: str
) -> str:
    """
    Analyze partitioning decisions.

    Compares forward and backward graphs to identify:
    - What activations are saved from forward for backward
    - Input/output correspondence between graphs
    - Memory impact of saved activations
    - Gradient flow

    Args:
        forward_path: Path to forward graph file
        backward_path: Path to backward graph file

    Returns:
        Analysis report string

    Raises:
        FileNotFoundError: If either graph file doesn't exist
    """
    from pathlib import Path

    # Validate files exist
    fwd_file = Path(forward_path)
    bwd_file = Path(backward_path)

    if not fwd_file.exists():
        raise FileNotFoundError(f"Forward graph not found: {forward_path}")
    if not bwd_file.exists():
        raise FileNotFoundError(f"Backward graph not found: {backward_path}")

    # Read both graphs
    forward_content = fwd_file.read_text()
    backward_content = bwd_file.read_text()

    # Parse forward graph outputs
    forward_outputs = _parse_forward_outputs(forward_content)

    # Parse backward graph inputs
    backward_inputs = _parse_backward_inputs(backward_content)

    # Identify saved activations (forward outputs used in backward)
    saved_activations = _identify_saved_activations(
        forward_outputs, backward_inputs
    )

    # Identify user output (first forward output, not used in backward)
    user_output = forward_outputs[0] if forward_outputs else None

    # Count operations in each graph
    forward_ops = _count_operations(forward_content)
    backward_ops = _count_operations(backward_content)

    # Build analysis report
    lines = [
        "Partitioning Analysis:",
        "",
        f"Forward Graph: {fwd_file.name}",
        f"Backward Graph: {bwd_file.name}",
        "",
        "=" * 60,
        "FORWARD GRAPH",
        "=" * 60,
        f"Operations: {forward_ops}",
        f"Outputs: {len(forward_outputs)}",
        ""
    ]

    if user_output:
        lines.extend([
            "User Output (returned to caller):",
            f"  - {user_output['name']} : {user_output['type']}",
            ""
        ])

    if saved_activations:
        lines.extend([
            f"Saved Activations (for backward): {len(saved_activations)}",
        ])
        for act in saved_activations:
            lines.append(f"  - {act['name']} : {act['type']}")
        lines.append("")
    else:
        lines.extend([
            "Saved Activations: None",
            ""
        ])

    lines.extend([
        "=" * 60,
        "BACKWARD GRAPH",
        "=" * 60,
        f"Operations: {backward_ops}",
        f"Inputs: {len(backward_inputs)}",
        ""
    ])

    # Categorize backward inputs
    saved_inputs = []
    tangent_inputs = []

    for inp in backward_inputs:
        if "tangent" in inp['name'].lower():
            tangent_inputs.append(inp)
        else:
            saved_inputs.append(inp)

    if saved_inputs:
        lines.extend([
            f"Saved Activations (from forward): {len(saved_inputs)}",
        ])
        for inp in saved_inputs:
            lines.append(f"  - {inp['name']} : {inp['type']}")
        lines.append("")

    if tangent_inputs:
        lines.extend([
            f"Gradient Inputs (tangents): {len(tangent_inputs)}",
        ])
        for inp in tangent_inputs:
            lines.append(f"  - {inp['name']} : {inp['type']}")
        lines.append("")

    # Analyze correspondence
    lines.extend([
        "=" * 60,
        "PARTITION BOUNDARY ANALYSIS",
        "=" * 60,
        ""
    ])

    # Check forward-backward correspondence
    fwd_saved_names = {act['name'] for act in saved_activations}
    bwd_saved_names = {inp['name'] for inp in saved_inputs}

    matched = fwd_saved_names & bwd_saved_names
    fwd_only = fwd_saved_names - bwd_saved_names
    bwd_only = bwd_saved_names - fwd_saved_names

    if matched:
        lines.extend([
            f"Matched Activations: {len(matched)}",
        ])
        for name in sorted(matched):
            lines.append(f"  ✓ {name}")
        lines.append("")

    if fwd_only:
        lines.extend([
            f"Forward-only (not used in backward): {len(fwd_only)}",
        ])
        for name in sorted(fwd_only):
            lines.append(f"  - {name}")
        lines.append("")

    if bwd_only:
        lines.extend([
            f"Backward-only (not in forward outputs): {len(bwd_only)}",
        ])
        for name in sorted(bwd_only):
            lines.append(f"  - {name}")
        lines.append("")

    # Memory analysis
    lines.extend([
        "=" * 60,
        "MEMORY IMPACT",
        "=" * 60,
        ""
    ])

    total_saved = len(saved_activations)
    if total_saved > 0:
        lines.extend([
            f"Total Saved Activations: {total_saved}",
            f"Memory Overhead: {total_saved} tensors saved for backward pass",
            "",
            "Note: Each saved activation consumes memory until backward completes.",
            "Consider gradient checkpointing for very deep models.",
            ""
        ])
    else:
        lines.extend([
            "No activations saved (inference mode or optimized out)",
            ""
        ])

    return "\n".join(lines)


def _parse_forward_outputs(content: str) -> list[dict]:
    """
    Parse forward graph return statement to extract outputs.

    Returns:
        List of dicts with 'name' and 'type' keys
    """
    import re

    # Find return statement
    # Example: return (relu, primals_1, primals_3, le)
    return_match = re.search(r'return\s+\(([^)]+)\)', content)
    if not return_match:
        return []

    return_vars = return_match.group(1)
    var_names = [v.strip() for v in return_vars.split(',')]

    # Find type annotations for each variable
    outputs = []
    for var_name in var_names:
        # Look for variable assignment with type
        # Example: relu: "f32[5, 10]" = ...
        type_match = re.search(
            rf'{re.escape(var_name)}:\s*"([^"]+)"\s*=',
            content
        )
        var_type = type_match.group(1) if type_match else "unknown"
        outputs.append({
            'name': var_name,
            'type': var_type
        })

    return outputs


def _parse_backward_inputs(content: str) -> list[dict]:
    """
    Parse backward graph forward signature to extract inputs.

    Returns:
        List of dicts with 'name' and 'type' keys
    """
    import re

    # Find forward method signature
    # Example: def forward(self, primals_1: "f32[10, 10]", primals_3: "f32[5, 10]", ...)
    sig_match = re.search(
        r'def\s+forward\s*\([^)]+\):',
        content,
        re.MULTILINE
    )

    if not sig_match:
        return []

    sig = sig_match.group(0)

    # Parse parameters (skip 'self')
    # Pattern: name: "type"
    param_pattern = r'(\w+):\s*"([^"]+)"'
    params = re.findall(param_pattern, sig)

    return [
        {'name': name, 'type': type_str}
        for name, type_str in params
    ]


def _identify_saved_activations(
    forward_outputs: list[dict],
    backward_inputs: list[dict]
) -> list[dict]:
    """
    Identify which forward outputs are saved for backward.

    Saved activations are forward outputs (excluding first user output)
    that appear as backward inputs.

    Returns:
        List of saved activation dicts
    """
    # User output is typically the first forward output
    # Remaining outputs are saved activations
    if len(forward_outputs) <= 1:
        return []

    saved = forward_outputs[1:]  # All but first

    # Verify these appear in backward inputs
    backward_names = {inp['name'] for inp in backward_inputs}

    # Filter to only those actually used in backward
    return [
        act for act in saved
        if act['name'] in backward_names
    ]


def _count_operations(content: str) -> int:
    """
    Count operations in a graph.

    Returns:
        Number of operations (call_function nodes)
    """
    import re

    # Count call_function occurrences
    ops = re.findall(r'torch\.ops\.\w+\.\w+', content)
    return len(ops)


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
    # Delegate to implementation in aot_post_grad module
    return await _analyze_post_grad_passes(backward_graph_before, backward_graph_after)
