"""Analyzers for each compilation stage."""

# Import submodules
from analyzers import aot_trace, cross_stage_trace, dynamo_trace, inductor_trace

# Re-export all analyzer functions for easy importing
from analyzers.aot_trace import (
    analyze_functionalization,
    analyze_joint_graph,
    analyze_partitioning,
    analyze_post_grad_passes,
)
from analyzers.cross_stage_trace import search_ir, trace_operation
from analyzers.dynamo_trace import (
    analyze_fx_graph,
    analyze_pre_grad_passes,
    parse_graph_breaks,
)
from analyzers.inductor_trace import (
    analyze_loopbody,
    analyze_lowering,
    analyze_triton_codegen,
    parse_fusion_decisions,
)

__all__ = [
    # Modules
    "aot_trace",
    "cross_stage_trace",
    "dynamo_trace",
    "inductor_trace",
    # Dynamo functions
    "parse_graph_breaks",
    "analyze_fx_graph",
    "analyze_pre_grad_passes",
    # AOT functions
    "analyze_functionalization",
    "analyze_joint_graph",
    "analyze_partitioning",
    "analyze_post_grad_passes",
    # Inductor functions
    "parse_fusion_decisions",
    "analyze_triton_codegen",
    "analyze_lowering",
    "analyze_loopbody",
    # Cross-stage functions
    "trace_operation",
    "search_ir",
]
