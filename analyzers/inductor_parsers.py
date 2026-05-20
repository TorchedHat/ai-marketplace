"""
Inductor Stage Parsers

Parse TORCH_LOGS output and debug files from Inductor compilation stage.
"""

import re


async def parse_fusion_decisions(log_content: str) -> str:
    """
    Parse fusion decisions from stdout (TORCH_LOGS="fusion,schedule").

    Args:
        log_content: Stdout from running with TORCH_LOGS="fusion,schedule"

    Returns:
        Formatted analysis of fusion decisions
    """
    # Look for fusion patterns
    fused = len(re.findall(r"(?:fused|fusing)", log_content, re.IGNORECASE))
    not_fused = len(re.findall(r"(?:cannot fuse|not fusing)", log_content, re.IGNORECASE))

    # Extract fusion reasons
    reasons = re.findall(
        r"(?:fuse|fusion).*?(?:because|reason):\s*(.+)", log_content, re.IGNORECASE
    )

    result = "## Fusion Decisions Analysis\n\n"
    result += f"**Fused:** ~{fused} mentions\n"
    result += f"**Not fused:** ~{not_fused} mentions\n\n"

    if reasons:
        result += "**Fusion reasons:**\n"
        for r in reasons[:10]:
            result += f"  - {r.strip()}\n"

    # Look for kernel mentions
    kernels = re.findall(r"kernel\d+", log_content)
    if kernels:
        unique_kernels = len(set(kernels))
        result += f"\n**Kernels generated:** ~{unique_kernels}\n"

    return result


async def parse_ir_post_fusion(ir_content: str) -> str:
    """
    Parse IR post-fusion file (ir_post_fusion_*.txt from TORCH_LOGS="ir_post_fusion").

    Args:
        ir_content: Content of ir_post_fusion_*.txt file

    Returns:
        Formatted analysis of LoopBody operations
    """
    # Count ops.* operations
    load_ops = len(re.findall(r"ops\.load", ir_content))
    store_ops = len(re.findall(r"ops\.store", ir_content))
    compute_ops = len(re.findall(r"ops\.(?:add|mul|sub|div|maximum|minimum)", ir_content))
    index_ops = len(re.findall(r"ops\.index_expr", ir_content))

    # Find reduction operations
    reductions = re.findall(r"Reduction.*?dim=(\d+)", ir_content)

    result = "## LoopBody IR Analysis\n\n"
    result += "**Operations:**\n"
    result += f"  - Load: {load_ops}\n"
    result += f"  - Store: {store_ops}\n"
    result += f"  - Compute: {compute_ops}\n"
    result += f"  - Index: {index_ops}\n"

    if reductions:
        result += f"\n**Reductions:** {len(reductions)}\n"
        for dim in set(reductions):
            result += f"  - Reducing over dimension {dim}\n"

    return result


async def parse_output_code(code_content: str) -> str:
    """
    Parse generated kernel code (output_code.py from TORCH_LOGS="output_code").

    Args:
        code_content: Content of output_code.py file

    Returns:
        Formatted analysis of generated kernels
    """
    # Detect kernel type (Triton vs C++)
    is_triton = "@triton.jit" in code_content or "import triton" in code_content
    is_cpp = "#include" in code_content or 'extern "C"' in code_content

    # Count kernels
    if is_triton:
        kernels = re.findall(r"@triton\.jit\s+def\s+(\w+)", code_content)
        kernel_type = "Triton (GPU)"
    elif is_cpp:
        kernels = re.findall(r"kernel_(\w+)", code_content)
        kernel_type = "C++ (CPU)"
    else:
        kernels = []
        kernel_type = "Unknown"

    result = "## Generated Kernel Analysis\n\n"
    result += f"**Kernel type:** {kernel_type}\n"
    result += f"**Kernels generated:** {len(kernels)}\n"

    if kernels:
        result += "\n**Kernel names:**\n"
        for k in kernels[:10]:
            result += f"  - {k}\n"
        if len(kernels) > 10:
            result += f"  - ... and {len(kernels) - 10} more\n"

    if is_triton:
        # Triton-specific analysis
        tiled = len(re.findall(r"XBLOCK|YBLOCK|RBLOCK", code_content))
        result += f"\n**Triton tiling blocks:** {tiled}\n"

    return result
