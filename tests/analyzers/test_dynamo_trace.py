"""Tests for Dynamo stage analyzers."""

import pytest

from analyzers.dynamo_trace import analyze_fx_graph, analyze_pre_grad_passes, parse_graph_breaks


class TestParseGraphBreaks:
    """Tests for parse_graph_breaks function."""

    @pytest.mark.asyncio
    async def test_parse_basic_graph_breaks(self):
        """Test parsing basic graph break format."""
        log = """
Graph break: y.sum().item()
  Reason: call_method aten.item.default - data-dependent operation
  User code: /path/to/file.py:36 in model_with_breaks
  Graph Count: 2

Graph break: print(f"Input shape: {x.shape}")
  Reason: call_function print in skip list
  User code: /path/to/file.py:29 in model_with_breaks
  Graph Count: 1
"""
        result = await parse_graph_breaks(log)
        
        assert "Total Breaks: 2" in result
        assert "Data-dependent operation" in result
        assert "Unsupported operation (skip list)" in result
        assert "Recommendations:" in result


class TestAnalyzeFXGraph:
    """Tests for analyze_fx_graph function."""

    @pytest.mark.asyncio
    async def test_analyze_fx_graph_basic(self):
        """Test basic FX graph analysis."""
        # Use real file from torch_compile_debug
        fx_file = "torch_compile_debug/run_2026_05_07_20_40_41_161544-pid_3044825/torchinductor/model__0_inference_0.0/fx_graph_readable.py"
        
        result = await analyze_fx_graph(fx_file)
        
        assert "Operation Summary:" in result or "Operations:" in result
        assert "aten." in result  # Should have aten operations

    @pytest.mark.asyncio
    async def test_analyze_fx_graph_missing_file(self):
        """Test handling of missing file."""
        result = await analyze_fx_graph("/nonexistent/path.py")
        assert "Error" in result


class TestAnalyzePreGradPasses:
    """Tests for analyze_pre_grad_passes function."""

    @pytest.mark.asyncio
    async def test_pre_grad_passes_basic(self):
        """Test pre-grad pass detection."""
        fx_file = "torch_compile_debug/run_2026_05_07_20_40_41_161544-pid_3044825/torchinductor/model__0_inference_0.0/fx_graph_readable.py"
        
        result = await analyze_pre_grad_passes(fx_file)
        
        assert "Pre-Grad Pass Analysis" in result or "Optimization" in result
