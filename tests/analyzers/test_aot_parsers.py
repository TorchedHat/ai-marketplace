"""Tests for AOT stage parsers."""

import pytest
from analyzers.aot_parsers import (
    parse_aot_graphs,
    parse_aot_joint_graph,
    parse_post_grad_passes,
)


class TestParseAOTJointGraph:
    """Test parse_aot_joint_graph parser."""

    @pytest.mark.asyncio
    async def test_basic_joint_graph(self):
        """Test parsing joint graph with forward and backward operations."""
        graph = """
def forward(self, primals_1, primals_2):
    # forward operations
    %mul = torch.ops.aten.mul(primals_1, primals_2)
    %relu = torch.ops.aten.relu(%mul)

    # backward operations
    %grad_relu = torch.ops.aten.threshold_backward(%relu, 0)
    %grad_mul = torch.ops.aten.mul(%grad_relu, primals_2)

    return [%relu, %grad_mul]
"""
        result = await parse_aot_joint_graph(graph)
        assert "AOT Joint Graph" in result
        assert "Total operations:" in result
        assert "operations" in result.lower()

    @pytest.mark.asyncio
    async def test_joint_graph_op_counts(self):
        """Test that operation counting works."""
        graph = """
%a = torch.ops.aten.add(x, y)
%b = torch.ops.aten.mul(%a, 2)
%c = torch.ops.aten.relu(%b)
"""
        result = await parse_aot_joint_graph(graph)
        assert "3" in result  # Should count 3 operations


class TestParseAOTGraphs:
    """Test parse_aot_graphs parser."""

    @pytest.mark.asyncio
    async def test_forward_only(self):
        """Test parsing forward graph only."""
        forward = """
def forward(self, x):
    %relu = torch.ops.aten.relu(x)
    %add = torch.ops.aten.add(%relu, 1.0)
    return [%add]
"""
        result = await parse_aot_graphs(forward)
        assert "Forward Graph:" in result
        assert "Operations:" in result

    @pytest.mark.asyncio
    async def test_forward_and_backward(self):
        """Test parsing both forward and backward graphs."""
        forward = """
def forward(self, x, y):
    %mm = torch.ops.aten.mm(x, y)
    return [%mm]
"""
        backward = """
def backward(self, grad_out, x, y):
    %grad_x = torch.ops.aten.mm(grad_out, y.t())
    %grad_y = torch.ops.aten.mm(x.t(), grad_out)
    return [%grad_x, %grad_y]
"""
        result = await parse_aot_graphs(forward, backward)
        assert "Forward Graph:" in result
        assert "Backward Graph:" in result
        assert "saved activations" in result.lower() or "inputs" in result.lower()

    @pytest.mark.asyncio
    async def test_memory_analysis(self):
        """Test that saved activation count is reported."""
        forward = "def forward(self, x): return [x]"
        backward = """
def backward(self, grad, saved_x, saved_y, saved_z):
    %placeholder1 = placeholder[target=saved_x]
    %placeholder2 = placeholder[target=saved_y]
    %placeholder3 = placeholder[target=saved_z]
    return []
"""
        result = await parse_aot_graphs(forward, backward)
        assert "Backward Graph:" in result
        # Should mention saved activations/inputs
        assert "3" in result or "inputs" in result.lower()


class TestParsePostGradPasses:
    """Test parse_post_grad_passes parser."""

    @pytest.mark.asyncio
    async def test_basic_post_grad_logs(self):
        """Test parsing post-grad pass logs."""
        log = """
Running pass: ConstantFolding
Running pass: DeadCodeElimination
Running pass: CommonSubexpressionElimination

Fused matmul operations
Eliminated 3 dead nodes
Replaced duplicate computations
"""
        result = await parse_post_grad_passes(log)
        assert "Post-Grad Pass" in result
        assert "Passes run:" in result or "pass" in result.lower()

    @pytest.mark.asyncio
    async def test_optimization_messages(self):
        """Test that optimization messages are captured."""
        log = """
Running pass: FusionPass
Fused 2 operations into single kernel
Eliminated redundant computation
Optimized memory layout
"""
        result = await parse_post_grad_passes(log)
        assert "Optimizations" in result or "optimization" in result.lower()
        assert "Fused" in result or "fused" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_post_grad_logs(self):
        """Test with no post-grad information."""
        log = "No optimization information here"
        result = await parse_post_grad_passes(log)
        assert "No post-grad" in result or "not found" in result.lower()
