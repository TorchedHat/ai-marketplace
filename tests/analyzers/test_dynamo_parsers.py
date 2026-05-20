"""Tests for Dynamo stage parsers."""

import pytest

from analyzers.dynamo_parsers import (
    parse_fx_graph,
    parse_graph_breaks,
    parse_pre_grad_passes,
)


class TestParseGraphBreaks:
    """Test parse_graph_breaks parser."""

    @pytest.mark.asyncio
    async def test_basic_graph_breaks(self):
        """Test parsing basic graph break output."""
        log = """
Graph break: tensor.item()
  Reason: data-dependent operation
  User code: test.py:10 in fn

Graph break: print(x)
  Reason: call_function print in skip list
  User code: test.py:15 in fn
"""
        result = await parse_graph_breaks(log)
        assert "Graph Breaks Analysis" in result
        assert "2" in result or "Total breaks: 2" in result
        assert "tensor.item()" in result or "item()" in result

    @pytest.mark.asyncio
    async def test_no_graph_breaks(self):
        """Test with no graph breaks."""
        log = "No graph breaks in this output"
        result = await parse_graph_breaks(log)
        assert "No graph breaks" in result


class TestParseFXGraph:
    """Test parse_fx_graph parser."""

    @pytest.mark.asyncio
    async def test_basic_fx_graph(self):
        """Test parsing FX graph content."""
        graph = """
def forward(self, x: torch.Tensor):
    %placeholder = placeholder[target=x]
    %relu = torch.ops.aten.relu(%placeholder)
    %add = torch.ops.aten.add(%relu, 1.0)
    return [%add]
"""
        result = await parse_fx_graph(graph)
        assert "FX Graph Analysis" in result
        assert "Operations:" in result
        assert "relu" in result or "aten.relu" in result


class TestParsePreGradPasses:
    """Test parse_pre_grad_passes parser."""

    @pytest.mark.asyncio
    async def test_optimization_applied(self):
        """Test when optimizations reduce operation count."""
        before = """
%a = torch.ops.aten.relu(x)
%b = torch.ops.aten.add(%a, 1)
%c = torch.ops.aten.mul(%b, 2)
"""
        after = """
%a = torch.ops.aten.relu(x)
%c = torch.ops.aten.fused_add_mul(%a, 1, 2)
"""
        result = await parse_pre_grad_passes(before, after)
        assert "Pre-Grad Pass Analysis" in result
        assert "Before:" in result
        assert "After:" in result
