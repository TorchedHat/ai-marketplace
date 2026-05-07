#!/usr/bin/env python3
"""
Debug Tracer MCP Server

Parsers for each torch.compile pipeline stage:
- Dynamo: Graph breaks, FX graphs, pre-grad passes
- AOT: Functionalization, joint graph, partitioning, post-grad
- Inductor: Lowering, fusion, loopbody, codegen
"""

import asyncio

from mcp.server import Server
from mcp.types import TextContent, Tool

app = Server("debug-tracer")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available debug tools for each compilation stage."""
    return [
        # ============================================================
        # Dynamo Stage Tools
        # ============================================================
        Tool(
            name="parse_graph_breaks",
            description="Parse TORCH_LOGS graph_breaks output (Dynamo stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_content": {"type": "string", "description": "TORCH_LOGS=graph_breaks output"}
                },
                "required": ["log_content"]
            }
        ),
        Tool(
            name="analyze_fx_graph",
            description="Analyze FX graph structure (Dynamo stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "fx_graph_path": {"type": "string", "description": "Path to fx_graph_readable.py"}
                },
                "required": ["fx_graph_path"]
            }
        ),
        Tool(
            name="analyze_pre_grad_passes",
            description="Analyze pre-grad pass effects (Dynamo stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "before_path": {"type": "string", "description": "fx_graph_readable.py"},
                    "after_path": {"type": "string", "description": "fx_graph_transformed.py"}
                },
                "required": ["before_path", "after_path"]
            }
        ),

        # ============================================================
        # AOT Stage Tools
        # ============================================================
        Tool(
            name="analyze_functionalization",
            description="Analyze mutation removal (AOT stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_path": {"type": "string", "description": "Path to AOT forward graph"}
                },
                "required": ["graph_path"]
            }
        ),
        Tool(
            name="analyze_joint_graph",
            description="Analyze joint forward+backward graph (AOT stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "joint_graph_path": {"type": "string", "description": "model__*__joint_*.py"}
                },
                "required": ["joint_graph_path"]
            }
        ),
        Tool(
            name="analyze_partitioning",
            description="Analyze AOT partitioning decisions (AOT stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "joint_path": {"type": "string", "description": "Joint graph file"},
                    "forward_path": {"type": "string", "description": "Forward graph file"},
                    "backward_path": {"type": "string", "description": "Backward graph file"}
                },
                "required": ["joint_path", "forward_path"]
            }
        ),
        Tool(
            name="analyze_post_grad_passes",
            description="Analyze post-grad optimization effects (AOT stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_content": {"type": "string", "description": "TORCH_LOGS=post_grad_graphs output"}
                },
                "required": ["log_content"]
            }
        ),

        # ============================================================
        # Inductor Stage Tools
        # ============================================================
        Tool(
            name="analyze_lowering",
            description="Analyze ATen → IR node lowering (Inductor stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "ir_file": {"type": "string", "description": "ir_*.txt path"}
                },
                "required": ["ir_file"]
            }
        ),
        Tool(
            name="parse_fusion_decisions",
            description="Parse fusion decisions and explain (Inductor stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_content": {"type": "string", "description": "TORCH_LOGS=fusion,schedule output"}
                },
                "required": ["log_content"]
            }
        ),
        Tool(
            name="analyze_loopbody",
            description="Analyze LoopBody IR (ops.* operations, Inductor stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "ir_post_fusion_path": {"type": "string", "description": "ir_post_fusion_*.txt path"}
                },
                "required": ["ir_post_fusion_path"]
            }
        ),
        Tool(
            name="analyze_triton_codegen",
            description="Analyze generated Triton kernel (Inductor stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_code_path": {"type": "string", "description": "output_code.py path"}
                },
                "required": ["output_code_path"]
            }
        ),

        # ============================================================
        # Cross-Stage Tools
        # ============================================================
        Tool(
            name="trace_operation",
            description="Trace operation through all compilation stages",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "Operation name (e.g., torch.relu)"},
                    "debug_dir": {"type": "string", "description": "Compilation output directory"}
                },
                "required": ["operation", "debug_dir"]
            }
        ),
        Tool(
            name="search_ir",
            description="Search IR files for patterns across compilation stages with line numbers and context",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "stage": {
                        "type": "string",
                        "enum": ["dynamo", "aot", "inductor", "loopbody", "all"],
                        "description": "Stage to search: dynamo (FX graphs), aot (AOT autograd), inductor (IR), loopbody (ops.*), or all"
                    },
                    "debug_dir": {"type": "string", "description": "Path to torch_compile_debug directory"}
                },
                "required": ["pattern", "stage", "debug_dir"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route tool calls to appropriate analyzers."""

    # Import analyzers
    from analyzers import dynamo_trace, aot_trace, inductor_trace, cross_stage_trace

    # Dynamo stage tools
    if name == "parse_graph_breaks":
        result = await dynamo_trace.parse_graph_breaks(arguments["log_content"])
    elif name == "analyze_fx_graph":
        result = await dynamo_trace.analyze_fx_graph(arguments["fx_graph_path"])
    elif name == "analyze_pre_grad_passes":
        result = await dynamo_trace.analyze_pre_grad_passes(
            arguments["before_path"], arguments["after_path"]
        )

    # AOT stage tools
    elif name == "analyze_functionalization":
        result = await aot_trace.analyze_functionalization(arguments["graph_path"])
    elif name == "analyze_joint_graph":
        result = await aot_trace.analyze_joint_graph(arguments["joint_graph_path"])
    elif name == "analyze_partitioning":
        result = await aot_trace.analyze_partitioning(
            arguments["joint_path"],
            arguments["forward_path"],
            arguments.get("backward_path")
        )
    elif name == "analyze_post_grad_passes":
        result = await aot_trace.analyze_post_grad_passes(arguments["log_content"])

    # Inductor stage tools
    elif name == "analyze_lowering":
        result = await inductor_trace.analyze_lowering(arguments["ir_file"])
    elif name == "parse_fusion_decisions":
        result = await inductor_trace.parse_fusion_decisions(arguments["log_content"])
    elif name == "analyze_loopbody":
        result = await inductor_trace.analyze_loopbody(arguments["ir_post_fusion_path"])
    elif name == "analyze_triton_codegen":
        result = await inductor_trace.analyze_triton_codegen(arguments["output_code_path"])

    # Cross-stage tools
    elif name == "trace_operation":
        result = await cross_stage_trace.trace_operation(
            arguments["operation"], arguments["debug_dir"]
        )
    elif name == "search_ir":
        result = await cross_stage_trace.search_ir(
            arguments["pattern"], arguments["stage"], arguments["debug_dir"]
        )

    else:
        raise ValueError(f"Unknown tool: {name}")

    return [TextContent(type="text", text=result)]


async def main():
    """Run MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
