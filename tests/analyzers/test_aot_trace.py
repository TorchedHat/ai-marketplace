"""Tests for AOT stage analyzers."""

import pytest

from analyzers.aot_trace import (
    analyze_functionalization,
    analyze_joint_graph,
    analyze_partitioning,
    analyze_post_grad_passes,
)


# Real paths from training run
FORWARD_PATH = "torch_compile_debug/run_2026_05_07_20_40_41_161544-pid_3044825/torchinductor/model__3_forward_4.3/fx_graph_readable.py"
BACKWARD_PATH = "torch_compile_debug/run_2026_05_07_20_40_41_161544-pid_3044825/torchinductor/model__3_backward_6.4/fx_graph_readable.py"


class TestAnalyzeFunctionalization:
    """Tests for analyze_functionalization function."""

    @pytest.mark.asyncio
    async def test_functionalization_clean_graph(self):
        """Test with properly functionalized graph (no inplace ops)."""
        result = await analyze_functionalization(FORWARD_PATH)
        
        assert "Functionalization" in result
        assert "Operations:" in result

    @pytest.mark.asyncio
    async def test_functionalization_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            await analyze_functionalization("/nonexistent/file.py")


class TestAnalyzeJointGraph:
    """Tests for analyze_joint_graph function."""

    @pytest.mark.asyncio
    async def test_joint_graph_forward(self):
        """Test analyzing forward graph."""
        result = await analyze_joint_graph(FORWARD_PATH)
        
        assert "Joint Graph Analysis" in result or "Graph Analysis" in result
        assert "Operations:" in result

    @pytest.mark.asyncio
    async def test_joint_graph_backward(self):
        """Test analyzing backward graph."""
        result = await analyze_joint_graph(BACKWARD_PATH)
        
        assert "Operations:" in result
        assert "tangents" in result.lower() or "gradient" in result.lower()


class TestAnalyzePartitioning:
    """Tests for analyze_partitioning function."""

    @pytest.mark.asyncio
    async def test_partitioning_basic(self):
        """Test basic partitioning analysis."""
        result = await analyze_partitioning(FORWARD_PATH, BACKWARD_PATH)
        
        assert "Partitioning Analysis" in result
        assert "Forward" in result
        assert "Backward" in result

    @pytest.mark.asyncio
    async def test_partitioning_identifies_saved_activations(self):
        """Test that saved activations are identified."""
        result = await analyze_partitioning(FORWARD_PATH, BACKWARD_PATH)
        
        assert "Saved Activations" in result or "saved" in result.lower()

    @pytest.mark.asyncio
    async def test_partitioning_memory_analysis(self):
        """Test memory analysis section."""
        result = await analyze_partitioning(FORWARD_PATH, BACKWARD_PATH)
        
        assert "Memory" in result or "memory" in result.lower()


class TestAnalyzePostGradPasses:
    """Tests for analyze_post_grad_passes function."""

    @pytest.mark.asyncio
    async def test_post_grad_passes_basic(self):
        """Test basic post-grad pass analysis."""
        result = await analyze_post_grad_passes(BACKWARD_PATH)
        
        assert "Post-grad" in result or "Optimization" in result or "Operations:" in result
