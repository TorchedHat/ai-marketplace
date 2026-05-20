#!/usr/bin/env python3
"""
torch-compile-ai MCP Server

9 parsers aligned with torch.compile IR levels:
- Dynamo: Graph breaks, FX graphs, pre-grad passes
- AOT: Joint graph, partitioned graphs, post-grad passes
- Inductor: Fusion decisions, IR post-fusion, output code
"""

import asyncio

from mcp.server import Server
from mcp.types import TextContent, Tool

app = Server("torch-compile-ai")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List 9 IR-level parsers for torch.compile debug output."""
    return [
        # ================================================================
        # Dynamo Stage (3 tools)
        # ================================================================
        Tool(
            name="parse_graph_breaks",
            description="Parse TORCH_LOGS='graph_breaks' stdout - identifies graph breaks and reasons",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_content": {
                        "type": "string",
                        "description": "Stdout from TORCH_LOGS='graph_breaks'",
                    }
                },
                "required": ["log_content"],
            },
        ),
        Tool(
            name="parse_fx_graph",
            description="Parse FX graph file content (fx_graph_readable.py from TORCH_LOGS='dynamo')",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_content": {
                        "type": "string",
                        "description": "Content of fx_graph_readable.py file",
                    }
                },
                "required": ["graph_content"],
            },
        ),
        Tool(
            name="parse_pre_grad_passes",
            description="Parse pre-grad pass effects (before/after FX graphs from TORCH_LOGS='pre_grad_graphs')",
            inputSchema={
                "type": "object",
                "properties": {
                    "before_content": {
                        "type": "string",
                        "description": "Content of fx_graph_readable.py (before passes)",
                    },
                    "after_content": {
                        "type": "string",
                        "description": "Content of fx_graph_transformed.py (after passes)",
                    },
                },
                "required": ["before_content", "after_content"],
            },
        ),
        # ================================================================
        # AOT Stage (3 tools)
        # ================================================================
        Tool(
            name="parse_aot_joint_graph",
            description="Parse AOT joint graph file (model__*__joint_*.py from TORCH_LOGS='aot_joint_graph')",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_content": {
                        "type": "string",
                        "description": "Content of joint graph file",
                    }
                },
                "required": ["graph_content"],
            },
        ),
        Tool(
            name="parse_aot_graphs",
            description="Parse partitioned AOT graphs (forward/backward from TORCH_LOGS='aot_graphs')",
            inputSchema={
                "type": "object",
                "properties": {
                    "forward_content": {
                        "type": "string",
                        "description": "Content of forward graph file",
                    },
                    "backward_content": {
                        "type": "string",
                        "description": "Content of backward graph file (optional)",
                    },
                },
                "required": ["forward_content"],
            },
        ),
        Tool(
            name="parse_post_grad_passes",
            description="Parse post-grad pass output (TORCH_LOGS='post_grad_graphs' stdout or files)",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_content": {
                        "type": "string",
                        "description": "Stdout or file content from post_grad_graphs logging",
                    }
                },
                "required": ["log_content"],
            },
        ),
        # ================================================================
        # Inductor Stage (3 tools)
        # ================================================================
        Tool(
            name="parse_fusion_decisions",
            description="Parse fusion decisions from stdout (TORCH_LOGS='fusion,schedule')",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_content": {
                        "type": "string",
                        "description": "Stdout from TORCH_LOGS='fusion,schedule'",
                    }
                },
                "required": ["log_content"],
            },
        ),
        Tool(
            name="parse_ir_post_fusion",
            description="Parse LoopBody IR (ir_post_fusion_*.txt from TORCH_LOGS='ir_post_fusion')",
            inputSchema={
                "type": "object",
                "properties": {
                    "ir_content": {
                        "type": "string",
                        "description": "Content of ir_post_fusion_*.txt file",
                    }
                },
                "required": ["ir_content"],
            },
        ),
        Tool(
            name="parse_output_code",
            description="Parse generated kernel code (output_code.py from TORCH_LOGS='output_code')",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_content": {
                        "type": "string",
                        "description": "Content of output_code.py file",
                    }
                },
                "required": ["code_content"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route tool calls to appropriate parsers."""

    # Import parsers
    from analyzers import (
        aot_parsers,
        dynamo_parsers,
        inductor_parsers,
    )

    # Dynamo stage
    if name == "parse_graph_breaks":
        result = await dynamo_parsers.parse_graph_breaks(arguments["log_content"])
    elif name == "parse_fx_graph":
        result = await dynamo_parsers.parse_fx_graph(arguments["graph_content"])
    elif name == "parse_pre_grad_passes":
        result = await dynamo_parsers.parse_pre_grad_passes(
            arguments["before_content"], arguments["after_content"]
        )

    # AOT stage
    elif name == "parse_aot_joint_graph":
        result = await aot_parsers.parse_aot_joint_graph(arguments["graph_content"])
    elif name == "parse_aot_graphs":
        result = await aot_parsers.parse_aot_graphs(
            arguments["forward_content"], arguments.get("backward_content")
        )
    elif name == "parse_post_grad_passes":
        result = await aot_parsers.parse_post_grad_passes(arguments["log_content"])

    # Inductor stage
    elif name == "parse_fusion_decisions":
        result = await inductor_parsers.parse_fusion_decisions(arguments["log_content"])
    elif name == "parse_ir_post_fusion":
        result = await inductor_parsers.parse_ir_post_fusion(arguments["ir_content"])
    elif name == "parse_output_code":
        result = await inductor_parsers.parse_output_code(arguments["code_content"])

    else:
        raise ValueError(f"Unknown tool: {name}")

    return [TextContent(type="text", text=result)]


async def main():
    """Run MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
